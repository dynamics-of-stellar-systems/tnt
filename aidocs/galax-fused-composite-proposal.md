# RFC: fuse the Gauss–Legendre quadrature across composite-potential children

*Proposal seeking feedback before any PR. Observations are against galax
`84d4730b` (`0.0.4.dev2`).*

## Problem

`AbstractCompositePotential._potential` sums each child's independently-evaluated
potential. Several builtin potentials evaluate their `_potential` as a *fixed-order
Gauss–Legendre quadrature of the same Chandrasekhar ellipsoidal `τ`-integral*, over
the same `[0, 1]` interval with the same node set:

- `TriaxialGaussianPotential`
- `AxisymmetricGaussianPotential`  (the `q1 = 1` special case)
- `TriaxialNFWPotential`

A composite of `N` such components — the common case being a Multi-Gaussian
Expansion, 10–20 `TriaxialGaussianPotential` terms, optionally plus a
`TriaxialNFWPotential` halo — runs `N` separate quadratures where one shared
quadrature would do. Since integration is linear and the nodes are identical,

    Σ_i  prefactor_i · ∫ integrand_i(s) ds   ≡   ∫  Σ_i prefactor_i · integrand_i(s)  ds

("sum of integrals" → "integral of sum") is exact, and the RHS is one reduction
over the node axis instead of `N`.

## Why it's worth doing

An MGE potential's `gradient()` (i.e. acceleration) is the call an orbit
integrator makes at every Runge–Kutta stage, under `vmap` over a batch of
particles. Benchmarks on that workload (laptop CPU, a 256-core cluster CPU node,
A40, A100; `N ∈ {5, 25, 125}`; batch `1 → 10⁷`):

| | `gradient()`, batched |
|---|---|
| **GPU (A40 / A100)** | fused is **~1.7–2×** faster than the summed composite (large-batch geomean ~1.95× A40, ~1.67× A100; A100 faster in every cell tested) |
| **CPU** | roughly a wash; mild regression at very large batch |

The cost is memory: the fused intermediate is `O(nodes · N · batch)` where the
summed composite is `O(N + batch)`, so a large enough `N × batch` can exhaust
device memory (an A40 with 48 GB OOMs around `N=125, batch=10⁷, f64`). That makes
fusion an **opt-in fast path**, not a new default. Batch-dimension chunking to
bound the footprint is a natural follow-up.

(A `lax.scan` over the nodes keeps the footprint flat but serialises the loop,
which reverse-mode autodiff then replays — it loses everywhere and is not worth
shipping. The win is specifically the array-fused `jnp.sum` over one
`(nodes, …, N)` tensor.)

The fused kernel is the same quadrature at the same order, so the result is
identical to the per-child path up to floating-point reduction order.

*(Full benchmark write-up:
<https://github.com/dynamics-of-stellar-systems/tnt/discussions/34>.)*

## The three `_potential` methods differ only trivially

Every one of them is:

```python
def _potential(self, xyz, t, /):
    xyz = u.Q.from_(xyz, self.units["length"])[None]     # add the integration batch axis
    t   = u.Q.from_(t, self.units["time"])
    batchdims = xyz.ndim - 2
    # ... compute r_s, rho0, q1, q2 ...
    def integrand(s):                                    # nested closure, not exposed
        s2 = s.reshape(s.shape + (1,) * batchdims) ** 2
        denom = jnp.sqrt(((q1sq - 1)*s2 + 1) * ((q2sq - 1)*s2 + 1))
        return delta_psi_factor(s2) / denom
    integral = self._integrator(integrand)
    return ((-2*jnp.pi * G * rho0 * r_s**2 * q1 * q2) * integral).ustrip(self.units["specific energy"])
```

The **only** differences across the three:

| | `delta_psi_factor(ξ²)` | prefactor | `_ellipsoid_surface` |
|---|---|---|---|
| `TriaxialGaussianPotential` | `2·exp(−ξ²/2)` | `−2πG ρ₀ r_s² q₁ q₂` | full |
| `AxisymmetricGaussianPotential` | `2·exp(−ξ²/2)` | `−2πG ρ₀ r_s² q₂` | `q₁ = 1` |
| `TriaxialNFWPotential` | `2/(1 + ξ)` | `−2πG ρ₀ r_s² q₁ q₂` | full |

The denominator, the ellipsoidal coordinate `ξ²(s)`, and the prefactor *form* are
shared. The integrand closure is the only thing that must vary per component — and
it's currently unreachable from outside `_potential`.

## Design

### Step 1 — expose a quadrature-integrand protocol (pure refactor, no behaviour change)

Introduce `AbstractQuadraturePotential(AbstractSinglePotential)` (or a mixin / a
duck-typed protocol — maintainers' call) carrying the `_integrator` field and two
hooks:

```python
class AbstractQuadraturePotential(AbstractSinglePotential):
    _integrator: GaussLegendreIntegrator

    def _quadrature_integrand(self, xyz, t, /) -> Callable[[Real[Array, "N"]], Real[Array, "N *batch"]]:
        """`f` such that `self._integrator(f)` is the dimensionless integral."""

    def _quadrature_prefactor(self, t, /) -> u.Quantity["specific energy"]:
        """Scalar (per leading batch) multiplying the integral to give the potential."""

    def _potential(self, xyz, t, /):
        integral = self._integrator(self._quadrature_integrand(xyz, t))
        return (self._quadrature_prefactor(t) * integral).ustrip(self.units["specific energy"])
```

The 3 potentials move their integrand closure and prefactor into these hooks; their
`_potential` becomes the inherited 2-liner. Existing tests are untouched.

**Contract to pin down:**
- integrand signature `f(s: Real[Array, "N"]) -> Real[Array, "N *batch"]`, matching
  what `GaussLegendreIntegrator.__call__` already feeds/expects;
- the leading integration axis convention (`xyz[None]` today);
- integrand returns dimensionless; the prefactor carries all units.

### Step 2 — a fused composite that uses the protocol

```python
class FusedCompositePotential(AbstractCompositePotential):
    # accepted iff every child is an AbstractQuadraturePotential sharing integration_order
    def _potential(self, xyz, t, /):
        parts = [(c._quadrature_prefactor(t), c._quadrature_integrand(xyz, t))
                 for c in self.values()]
        fused = lambda s: sum(pf * ig(s) for pf, ig in parts)
        return self._integrator(fused).ustrip(self.units["specific energy"])

    _gradient = <jax.grad of the fused _potential>   # not AbstractCompositePotential's per-child sum
```

Profile-agnostic: no `isinstance` dispatch, mixed Gaussian + NFW children fuse,
and any future quadrature potential is covered for free. `_density` /
`_laplacian` / `_hessian` stay inherited (correct, just not fused).

Delivery form — three options, roughly increasing invasiveness:
- **(a)** a standalone `FusedCompositePotential` subclass (opt-in by construction);
- **(b)** `CompositePotential(..., fuse=True)` / a `.fused()` view;
- **(c)** auto-fuse inside `AbstractCompositePotential._potential` when the children
  qualify — cleanest API but changes the memory profile silently, so probably not.

(a) or (b), given the memory trade-off.

## Open questions for maintainers

1. Interest at all? If galax would rather not carry composite-fusion logic, this
   stays a downstream extension (TNT already has a working `FusedCompositePotential`
   along the lines of Step 2 — happy to upstream it, or not).
2. Public or private protocol names (`_quadrature_integrand` vs `quadrature_integrand`)?
3. New `AbstractQuadraturePotential` base, a mixin, or a runtime-checkable `Protocol`?
4. Delivery form (a) / (b) / (c) above.
5. Is batch-dimension chunking in scope here, or left to callers?

## Reference implementation

TNT (public: <https://github.com/dynamics-of-stellar-systems/tnt>) carries a
working `FusedCompositePotential` — zero non-`galax` deps beyond jax/unxt/equinox,
written as a `galax.potential` extension. It does Step 2 with an `isinstance`
dispatch standing in for the missing Step 1 protocol: accepts the 3 types, fuses
homogeneous or mixed children, `_potential` + `_gradient` only, checked against
`CompositePotential` across batch shapes and profile mixes. Happy to open a PR
here along the two steps above if there's interest.
