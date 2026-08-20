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

TNT potential components are backed by [`galax`](https://github.com/GalacticDynamics/galax),
a JAX library for galactic dynamics. `potential.<name>.type` should name a
`galax.potential` class directly -- `"NFWPotential"`, `"PlummerPotential"`,
`"TriaxialNFWPotential"`, and so on; see `galax.potential`'s
[`__init__.py`](https://github.com/GalacticDynamics/galax/blob/247a33556809398f2f7c34c1c3fee74f4e46ba45/src/galax/potential/__init__.py)
for every class it currently provides.

In addition to `galax` potential components, TNT provides MGE-based
component types, `"triaxial_light_mge"` and `"triaxial_mass_mge"` (see
[MGE composite types](#mge-composite-types)).

## Parameterizations (optional)

By default, TNT assumes a component's own native `galax` parameterization --
however other parameterizations may be preferred. NFW is the
motivating exception: `galax.potential.NFWPotential`'s native parameters are
a characteristic mass `m` and scale radius `r_s`, but it may be more
appropriate to search over concentration and mass fraction. The optional `parameterization` field can be used to specify alternatives:

- **Omitted**: `parameters` must match the native `galax` names exactly.
- **Given**: names a registered conversion from some other parameter
  convention into the native `galax` fields. Today, only one such conversion
  is registered anywhere (NFW's `concentration_mass_ratio`), and it isn't
  implemented yet -- see below.

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
    parameterization: "concentration_mass_ratio"   # non-native; not yet implemented
    include: true
    parameters:
      c: {value: 3.0, fixed: true}
      f: {value: 1.0, fixed: true}
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

- **Any native-galax-parameterized type** (`parameterization` omitted): the
  component resolves and `to_galax()` (building the actual `galax`
  potential object) works for every `galax.potential` class -- verified
  against `PlummerPotential`, `NFWPotential`, and `TriaxialNFWPotential`.
  `rescale()` works the same way for parameters with a confirmed
  mass-rescale exponent (mass, length, angle, dimensionless, speed --
  verified including `LogarithmicPotential`'s velocity-parameterized
  `v_c`); each addition to that confirmed set requires the same kind of
  direct verification, since e.g. a bar's pattern speed
  (`MonariEtAl2016BarPotential`'s `Omega`) shares `v_c`'s dimension but has
  to stay fixed under a mass rescale rather than scale with it.
- **NFW's `concentration_mass_ratio` parameterization**: registered, but not
  implemented -- converting `(c, f)` into native `(m, r_s)` needs a formula
  from the triaxial-Schwarzschild-modeling / DYNAMITE-successor literature
  that hasn't been confirmed yet. Building a component with this
  parameterization raises `NotImplementedError` naming exactly what's
  missing.
- **The two MGE composite types**: `from_settings` resolves the component
  and its named MGE, but `to_galax()` raises `NotImplementedError` -- no
  native `galax.potential` class exists for a sum-of-triaxial-Gaussians
  potential, so building one needs a custom `galax.potential.AbstractPotential`
  subclass, the same difficulty tier as `AbstractMGE.get_projected_mass`'s
  from-scratch implementation. The MGE `stars` component's own viewing-geometry
  parameterization (`q_min`, `p_min`, `u` -> `theta`, `phi`, `psi`) is a
  second, separate unconfirmed formula, for the same reason as NFW's.
- **`Potential.generate_orbit_library`**: not implemented -- blocked on
  `tnt.orbit_library`, itself still a full scaffold.
