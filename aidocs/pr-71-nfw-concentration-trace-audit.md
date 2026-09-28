# PR 71 audit: NFW concentration conversion traceable in batches

## Follow-up (2026-09-28)

F1 has been fixed in the working tree: the concentration solver now uses an
implicit JAX derivative, and tests compare inverse gradients with finite
differences for scalar and batched inputs. The focused NFW tests and the full
potential unit-test module passed. At this follow-up, the changes were not yet
committed or pushed; the review below describes the earlier commit named in
"Reviewed repository state."

The inverse's current internal callers recompute raw parameters while
generating models and reporting their values. The audit demonstrated incorrect
gradients through the inverse API, but did not establish that the current
model-fitting path differentiates through that API.

One pre-existing range limitation remains: a true concentration outside the
solver's `[1e-6, 1e6]` bracket is returned as a clamped endpoint. The new
implicit derivative is valid for roots inside the bracket; it does not
describe the clamped result. The solver docstring now states this limit.

Audit date: 2026-09-28. Reviewer: Claude, at Prash's request.

**Recommendation: do not merge yet.** One correctness finding needs
addressing: the PR fixes and tests only the *forward* `(c, M_200) -> (m,
r_s)` converter's differentiability, but the *inverse* converter this same
module exposes -- `(m, r_s) -> (c, M_200)`, needed after
`GalaxPotentialComponent.rescale()` per that function's own docstring --
remains silently non-differentiable under `jax.grad`, unchanged by this PR
and untested by it.

## Reviewed repository state

- PR: [#71 -- Make NFW concentration conversion traceable in batches](https://github.com/dynamics-of-stellar-systems/tnt/pull/71).
- Reviewed branch: `codex/nfw-concentration-trace`.
- Reviewed commit: `1b8b656aa8facd47f530ae7ece279fd6962b3206`.
- Base: `main`, commit `7429509d1fb59d45a599e17170ac78c3793e48c2`.
- Reviewed in an isolated `git worktree` checkout of the PR branch, not the
  primary working tree (which had unrelated in-progress work on another
  branch); no files outside this audit doc were touched.
- GitHub reports the PR as `MERGEABLE`.

## Scope and completed work

- [x] Read the PR's diff in full (`tnt/potential/nfw.py`,
  `tests/unit_tests/test_potential.py`, `aidocs/KNOWLEDGE.md`).
- [x] Read the surrounding, unmodified code in `tnt/potential/nfw.py`
  (`_solve_nfw_concentration`, `_nfw_concentration_m200_inverse`) that the
  PR's stated goal ("traceable in batches") bears on but doesn't touch.
- [x] Independently reproduced the finding below by direct execution, not
  just static reading.
- [x] Ran the new tests the PR adds.

## Findings requiring changes

### F1 -- P1: `_solve_nfw_concentration`'s bisection drops gradients to zero

**Location:** `tnt/potential/nfw.py:71-88` (`_solve_nfw_concentration`),
consumed by `_nfw_concentration_m200_inverse` (`:94-124`). Both are
pre-existing code this PR does not modify.

`_solve_nfw_concentration` is a fixed-80-iteration bisection:

```python
too_low = h(mid) < target
lower = jnp.where(too_low, mid, lower)
upper = jnp.where(too_low, upper, mid)
```

`too_low` is a boolean comparison; `jnp.where` on it routes no gradient
back from `lower`/`upper`/the returned `c` to `target` through that
branch. `jax.grad` therefore sees `d(c)/d(target)` as identically `0.0`,
not the true nonzero derivative a bisection's *fixed point* actually has
(the implicit-function derivative of `h(c) = target`).

Verified directly, on this branch:

```
>>> jax.grad(_solve_nfw_concentration)(50.0)
0.0
>>> central finite difference at target=50.0
0.0302578049193869
```

The same corruption propagates through `_nfw_concentration_m200_inverse`.
Differentiating its `M_200` output w.r.t. `(m, r_s)` at `m=1e12 Msun,
r_s=20 kpc` (`H=70 km/s/Mpc`):

```
autodiff:      (1.6561716807634490, 0.0)
finite diff:   (2.0000465975341797, -51581237554.93165)
```

The `r_s` partial is dropped to exactly `0.0` where the true derivative is
large and negative; the `m` partial's magnitude is off by ~20% too (both
partials pass through `_solve_nfw_concentration`, so both inherit its
zeroed derivative w.r.t. `target`).

**Why this is in scope for this PR, not just a pre-existing issue to note
separately:** the PR's own title and commit message are "traceable in
batches" / "so JAX can batch and differentiate the converter", and its new
test (`test_nfw_concentration_m200_traces_with_gradients`) proves exactly
that property -- but only for the forward converter. The inverse
converter's docstring (`_nfw_concentration_m200_inverse`, unchanged by
this PR) explains it exists specifically to recompute `(c, M_200)`
correctly after `GalaxPotentialComponent.rescale()` -- a rescale that, in
this codebase, is naturally exercised inside gradient-based fitting code.
Silently zeroed/wrong gradients through that exact path, with no error and
no test either direction, is the kind of failure this PR's own stated
purpose commits to ruling out.

**Failure scenario:** any gradient-based code (an optimizer, a sensitivity
analysis, a fit) that differentiates through
`_nfw_concentration_m200_inverse` after a rescale gets wrong -- frequently
exactly-zero -- gradients with no error raised, no warning, and nothing in
this PR's own new coverage that would catch it.

**Required outcome:** make `_solve_nfw_concentration` differentiable
(e.g. `jax.custom_jvp`/`custom_vjp` supplying the implicit-function
derivative of `h(c) = target`, or an equivalent closed-form correction on
the returned `c`), and add a test mirroring
`test_nfw_concentration_m200_traces_with_gradients` but for
`_nfw_concentration_m200_inverse`, asserting finite, correctly-valued
gradients w.r.t. `(m, r_s)` under both `jax.grad` and `jax.vmap`.

## Verification commands used

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from tnt.potential.nfw import _solve_nfw_concentration

g = jax.grad(_solve_nfw_concentration)(50.0)
eps = 1e-4
fd = (_solve_nfw_concentration(50.0 + eps) - _solve_nfw_concentration(50.0 - eps)) / (2 * eps)
# g == 0.0, fd == 0.0302578049193869
```

```python
import jax
jax.config.update("jax_enable_x64", True)
from tnt.potential.nfw import _nfw_concentration_m200_inverse
from unxt import Quantity

def f(m_val, rs_val):
    native = {"m": Quantity(m_val, "Msun"), "r_s": Quantity(rs_val, "kpc")}
    out = _nfw_concentration_m200_inverse(
        native, {"M_200": "Msun"}, {"H": Quantity(70.0, "km/(s*Mpc)")}
    )
    return out["M_200"].ustrip("Msun")

m0, rs0 = 1e12, 20.0
g = jax.grad(f, argnums=(0, 1))(m0, rs0)
eps_m, eps_rs = m0 * 1e-6, rs0 * 1e-6
fd_m = (f(m0 + eps_m, rs0) - f(m0 - eps_m, rs0)) / (2 * eps_m)
fd_rs = (f(m0, rs0 + eps_rs) - f(m0, rs0 - eps_rs)) / (2 * eps_rs)
# g == (1.6561716807634490, 0.0)
# (fd_m, fd_rs) == (2.0000465975341797, -51581237554.93165)
```

Run from this worktree's checkout, matching the reviewed commit exactly.
