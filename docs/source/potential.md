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
component types, `"triaxial_light_mge"` and `"triaxial_mass_mge"` (see
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
  also needs `H0` from the resolved configuration's
  `cosmological_parameters` section to compute the critical density.

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
    include: true
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
    include: true
    parameters:
      c: {value: 8.0, fixed: true}
      M_200: {value: 1.0e12, unit: "Msun", fixed: true}
```

- `type` (required): a `galax.potential` class name, or one of the two MGE
  composite type names.
- `parameterization` (optional): a named conversion registered for `type`.
  Omit it to use `type`'s native parameters directly.
- `parameters` (required unless `include: false`): one entry per parameter
  the resolved `type`/`parameterization` pair expects, each with a `value`
  and, if that parameter carries physical units, a `unit`.
- `include` (required): whether this component participates in the
  assembled potential.

### MGE composite types

`"triaxial_light_mge"` and `"triaxial_mass_mge"` build a potential from a
named Multi-Gaussian Expansion (MGE) -- TNT provides these two types
directly, since `galax.potential` has no "sum of triaxial Gaussians"
potential of its own to name. Both require an `mge` field naming a
registered MGE (see [Configuration preparation](configuration.md)):

```yaml
potential:
  stars:
    type: "triaxial_light_mge"
    include: true
    mge: "mge_lum"
    parameters:
      ml: {value: 5.0, unit: "Msun / Lsun"}   # triaxial_mass_mge uses mge_mass_scale instead
```

`triaxial_light_mge`'s `ml` (mass-to-light ratio) and `triaxial_mass_mge`'s
`mge_mass_scale` (a pure multiplicative scale factor on an already
mass-calibrated MGE) are TNT's own parameter names for these two types.

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
- **NFW's `concentration_m200` parameterization**: implemented and verified
  against `galax`'s own enclosed-mass function. Converts a concentration `c`
  and $M_{200c}$ into native `(m, r_s)` via the critical-density definition
  of $M_{200c}$: $\rho_\mathrm{crit} = 3 H_0^2 / (8\pi G)$,
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
- **The two MGE composite types**: `from_settings` resolves the component
  and its named MGE, but `to_galax()` raises `NotImplementedError` -- no
  native `galax.potential` class exists for a sum-of-triaxial-Gaussians
  potential, so building one needs a custom `galax.potential.AbstractPotential`
  subclass, the same difficulty tier as `AbstractMGE.get_projected_mass`'s
  from-scratch implementation. The MGE `stars` component's own viewing-geometry
  parameterization (`q`, `p`, `u` -> `theta`, `phi`, `psi`) needs a formula
  from the triaxial-Schwarzschild-modeling / DYNAMITE-successor literature
  that hasn't been confirmed yet.
- **`Potential.generate_orbit_library`**: not implemented -- blocked on
  `tnt.orbit_library`, itself still a full scaffold.
