# Potential

TNT potentials are assembled from one or more named components (e.g. a
stellar mass distribution, a dark-matter halo, a central black hole, ...).
A component is specified by its `type` and, optionally, a
`parameterization` -- both discussed below.

This part of TNT is under active development. Some of what's described below
is signature-only scaffolding rather than a working implementation -- see
[What's implemented today](#whats-implemented-today) and `tnt.potential`'s
own module docstring for exactly what raises `NotImplementedError`.

## Component types

TNT potential components are backed by [`galax`](https://github.com/GalacticDynamics/galax), a JAX library for galactic dynamics. `potential.<name>.type` names one of a
curated set of 25 `galax.potential` classes (e.g. `"NFWPotential"`);
see `tnt.potential._SUPPORTED_GALAX_TYPES` for the exact list.

Some `galax.potential` classes are not supported: abstract/base classes;
pre-packaged multi-component bundles with no free parameters of their own,
like `MilkyWayPotential` (its `disk`/`bulge`/`halo`/`nucleus` fields are
themselves sub-potentials, not `Quantity` parameters); wrapper/transform
decorators needing a required nested potential object; and classes needing
a required non-`Quantity` hyperparameter (e.g. `MultipolePotential`'s
`l_max: int`) -- none representable by a component's scalar
`parameters.<name>.value` schema. Each curated class also carries a
mass-rescale exponent per native parameter (see [What's implemented
today](#whats-implemented-today)).

In addition to `galax` potential components, TNT provides MGE-based
component types -- triaxial (`"TriaxialLightMGEPotential"`/
`"TriaxialMassMGEPotential"`) and oblate axisymmetric
(`"OblateLightMGEPotential"`/`"OblateMassMGEPotential"`) (see
[MGE composite types](#mge-composite-types)).

## Parameterizations (optional)

By default, TNT assumes a component's own native `galax` parameterization --
however other parameterizations may be preferred. NFW is the
motivating example: `galax.potential.NFWPotential`'s native parameters are
a characteristic mass `m` and scale radius `r_s`, but it may be more
appropriate to search over (concentration, $M_{200c}$). The optional `parameterization`
field can be used to specify alternatives:

- **Omitted**: `parameters` must match the native `galax` names exactly.
- **Given**: names a registered conversion from some other parameter
  convention into the native `galax` fields. Today, NFW registers
  `concentration_m200`, converting a concentration `c` and $M_{200c}$ (mass
  enclosed within the radius where the mean density is 200 times the
  critical density) into native `(m, r_s)`. Some parameterizations need more
  than a component's own `parameters` to convert -- `concentration_m200`
  also needs `H` from the resolved configuration's
  `cosmological_parameters` section (the Hubble parameter at the halo's own
  epoch, not necessarily today's H0) to compute the critical density.

  A parameterization converts only within one component's own raw
  parameters; it can't depend on another component's resolved state (e.g. a
  mass ratio to another component's total mass) -- components are resolved
  independently of each other. That kind of cross-component relationship
  belongs to a separate, not-yet-designed "prior" concept, consumed by the
  parameter generator/search space rather than by potential construction.

Every registered parameterization also converts back: `AllModels`' table
always reports a component's parameters the way its configuration actually
specified it (`dh.c`/`dh.M_200` under `concentration_m200`, not `dh.m`/
`dh.r_s`), even after `parameter_space_settings.potential_rescalings`
rescales the underlying `galax` potential -- rescaling only knows how to
scale native parameters (see [What's implemented today](#whats-implemented-today)),
so the raw parameterization's values are recomputed from the rescaled
native ones, not carried through unchanged.

## Configuration reference

```yaml
potential:
  bh:
    type: "PlummerPotential"      # a galax.potential class name
    parameters:
      m_tot:                      # galax's own native kwarg name
        value: 5.0
        unit: "Msun"
        fixed: true
      r_s:
        value: 1.0e-3
        unit: "kpc"
        fixed: true

  dh:
    type: "NFWPotential"
    parameterization: "concentration_m200"   # non-native
    parameters:
      c: {value: 8.0, fixed: true}
      M_200: {value: 1.0e12, unit: "Msun", fixed: true}
```

- `type` (required): a `galax.potential` class name, or one of the four MGE
  composite type names.
- `parameterization` (optional): a named conversion registered for `type`.
  Omit it to use `type`'s native parameters directly.
- `parameters` (required): one entry per parameter the resolved
  `type`/`parameterization` pair expects -- every native field, including
  ones with a `galax` constructor default (e.g. `TriaxialHernquistPotential`'s
  `q1`/`q2`) -- each with a `value` and, if it carries physical units, a
  `unit`.

Every declared component is part of the assembled potential -- to leave one
out, remove or comment out its entry.

Configuration preparation checks parameter names and declared physical
dimensions. When a proposed parameter set is turned into a runtime potential,
TNT additionally requires every value to be a scalar, finite `Quantity` and
applies the physical-domain rules owned by that component type. Raw
parameterization values are checked before conversion; the resulting native
parameters are checked again afterward. This catches invalid inputs such as
non-positive masses or scale lengths, non-positive NFW `c`/`M_200`, profile
slopes outside their analytic domains, invalid axis ordering, and invalid MGE
normalizations before a `galax` potential or orbit library is constructed.
Masses, MGE normalizations, and every scale length are strictly positive;
zero is not a disabled-component convention or an accepted limiting case.
`MonariEtAl2016BarPotential.alpha` and its pattern speed `Omega` remain signed
parameters, so either rotation direction is representable.
Comparisons convert compatible units locally but do not normalize or replace a
parameter's declared unit. MGE deprojection continues to own the more complex
viewing-geometry checks that depend on the MGE data itself.

### MGE composite types

All four build a potential from a named Multi-Gaussian Expansion (MGE) --
TNT provides these directly, since `galax.potential` has no "sum of
Gaussians" potential of its own to name for the MGE-specific
deprojection/composition wiring. All require an `mge` field naming a
registered MGE (see [Configuration preparation](configuration.md)), and
split along two independent axes:

- **Mass parameterization**: light types (`ml`, a mass-to-light ratio) vs.
  mass types (`mge_mass_scale`, a pure multiplicative scale factor on an
  already mass-calibrated MGE) -- TNT's own parameter names, not `galax`'s.
- **Deprojection convention**: triaxial types (`theta`/`phi`/`psi`, the
  global viewing angles the MGE is deprojected under -- see
  `AbstractMGE.deproject_triaxial`) vs. oblate axisymmetric types (a single
  `inclination`, in `(0, 90]` degrees -- see `AbstractMGE.deproject_oblate`,
  which also requires the named MGE's `PA_twist` to be zero for every
  component, an axisymmetric system having no isophote twist). The
  axisymmetric deprojection is the oblate convention only (`p = B/A = 1`,
  `q = C/A <= 1`); a prolate spheroid needs a different relation and would
  get its own `Prolate...` types.

```yaml
potential:
  stars:
    type: "TriaxialLightMGEPotential"
    mge: "mge_lum"
    parameters:
      ml: {value: 5.0, unit: "Msun / Lsun"}   # a mass type uses mge_mass_scale instead
      theta: {value: 1.0, unit: "rad"}
      phi: {value: 0.5, unit: "rad"}
      psi: {value: 0.0, unit: "rad"}

  bulge:
    type: "OblateLightMGEPotential"
    mge: "mge_bulge"
    parameters:
      ml: {value: 3.0, unit: "Msun / Lsun"}
      inclination: {value: 90.0, unit: "deg"}   # edge-on; must be in (0, 90] deg
```

## What's implemented today

- **Every curated type** (`parameterization` omitted): the component
  resolves and `to_galax()` (building the actual `galax` potential object)
  works for every class in `tnt.potential._SUPPORTED_GALAX_TYPES` -- 25
  classes, from ordinary single-component potentials (`PlummerPotential`,
  `NFWPotential`, `HernquistPotential`, ...) to triaxial and bar potentials.
  `rescale()` works for every native parameter of every curated class,
  each with an individually confirmed mass-rescale exponent -- curated
  rather than derived from dimension, since dimension alone doesn't
  determine a parameter's role: `MonariEtAl2016BarPotential`'s `Omega`
  (bar pattern speed, stays fixed) and `v0`
  (sets the potential's amplitude, scales) share the same time-power but
  play opposite roles. Adding a new class to the curated set requires the
  same kind of direct verification against `galax`'s own potential formula
  for every one of its parameters.
  Runtime construction also validates the scalar/finite contract and each
  curated parameter's physical domain before constructing the component.
- **NFW's `concentration_m200` parameterization**: implemented and verified
  against `galax`'s own enclosed-mass function. Converts a concentration `c`
  and $M_{200c}$ into native `(m, r_s)` via the critical-density definition
  of $M_{200c}$, using the configured Hubble parameter $H$ (the value at the
  halo's epoch, not necessarily the present-day $H_0$):
  $\rho_\mathrm{crit} = 3 H^2 / (8\pi G)$,
  $r_{200} = (3 M_{200} / (4\pi \cdot 200 \rho_\mathrm{crit}))^{1/3}$,
  $r_s = r_{200} / c$, $m = M_{200} / (\ln(1+c) - c/(1+c))$. The reverse
  conversion, native `(m, r_s)` back to `(c, M_200)`, is implemented too --
  used to build `AllModels`' table columns -- but has no closed form:
  `rescale()` scales `m` while holding `r_s` fixed, which is *not* the same
  as holding `c` fixed and scaling `M_200`, so recovering `c` after a
  rescale means solving a transcendental equation. Solved numerically via
  bisection, verified by confirming the round trip is self-consistent
  (converting the recovered `(c, M_200)` forward again reproduces the same
  rescaled `(m, r_s)`), since there's no independent closed-form answer to
  check against.
- **All four MGE composite types**: implemented. The named MGE is
  deprojected -- triaxial types under `theta`/`phi`/`psi`
  (`AbstractMGE.deproject_triaxial`), oblate axisymmetric types under a
  single `inclination` (`AbstractMGE.deproject_oblate`) -- once, when the
  component itself is built from a proposed point in parameter space, not
  lazily inside `to_galax()`, so an invalid viewing geometry
  (`tnt.mge.MGEDeprojectionError`, for a deprojection with no real solution
  or intrinsic axial ratios outside TNT's `0 < q <= p <= 1` convention)
  surfaces there, before anything downstream is attempted. `to_galax()` then
  sums one `galax.potential.TriaxialGaussianPotential` /
  `AxisymmetricGaussianPotential` per Gaussian component. TNT uses these
  native `galax` Gaussian potentials and `CompositePotential` directly rather
  than defining a custom `galax.potential.AbstractPotential` subclass; each
  density matches its
  `Deprojected3DMGE` counterpart term for term (`r_s <-> sigma`,
  `q1 <-> p`, `q2 <-> q`; an oblate axisymmetric deprojection's `p` is always 1),
  giving a direct, verified `m_tot = I * p * q * (2*pi)**1.5 * sigma**3`
  conversion per component. `(theta, phi, psi)` / `inclination` are native;
  the alternative `(p, q, u)` shape/compression parameterization closer to
  the triaxial-Schwarzschild-modeling / DYNAMITE-successor literature isn't
  registered yet -- converting it to `(theta, phi, psi)` needs a formula
  that hasn't been confirmed.
- **`Potential.generate_orbit_library`**: not implemented -- blocked on
  `tnt.orbit_library`, itself still a full scaffold.
