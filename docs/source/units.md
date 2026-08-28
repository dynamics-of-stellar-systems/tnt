# Units

TNT uses [`unxt`](https://unxt.readthedocs.io/) to validate units and define
two related unit systems:

- `units.internal` names the base units of the unit system TNT hands to
  `galax` when it constructs a real potential object for orbit integration
  (and for prior plugins that need one) -- see
  `Potential.to_galax()`. It is *not* a normalization applied to declared
  configuration values or to data read from files: those keep the units they
  are declared in.
- `units.display` controls presentation preferences. Any dimension not
  overridden there inherits its internal unit.

The packaged defaults currently use:

```yaml
units:
  internal:
    length: "kpc"
    time: "Myr"
    mass: "Msun"
    angle: "rad"
  display:
    angle: "arcsec"
    speed: "km / s"
```

`units.internal` requires exactly four keys -- `length`, `time`, `mass`, and
`angle` -- the dimensions `galax`'s potential types use that `unxt` cannot
derive on its own. `unxt` builds `power`, `speed`, `frequency`, ...
automatically from mass/length/time, so declaring them is redundant and now
rejected; `angle` is dimensionally independent (`unxt` cannot decompose
`rad` into the mechanical bases) and is a real native parameter dimension
for some `galax` types, so it must be stated. `power` used to be required
and has been dropped -- a breaking change from the earlier five-key schema,
handled like other config renames with no back-compat shim; besides being
redundant, requiring it let a config declare a `power` unit inconsistent
with the others that `galax` silently ignores. `units.display` still accepts
`power` as an optional presentation override, alongside `speed` and the four
base dimensions.

TNT checks that every unit describes the dimension named by its key. For
example, using `Myr` as the internal length unit is an error. The resolved
configuration object exposes the constructed systems as
`config.unit_systems.internal` and `config.unit_systems.display`.

## Quantity syntax

Every unitful value must state its unit, even when that unit matches a
configured internal unit and even when the value is zero. Standalone
quantities use a `value` and `unit` mapping:

```yaml
system_attributes:
  distance: {value: 39960.0, unit: "kpc"}

kinematic_data:
  central_spectroscopy:
    histogram:
      center: {value: 0.0, unit: "km / s"}

cosmological_parameters:
  H:
    value: 70.0
    unit: "km / (s Mpc)"
```

The flow-style and block-style mappings have identical meaning. A two-item
YAML sequence such as `[39.96, Mpc]` is deliberately not supported: naming
`value` and `unit` makes the schema clearer and permits precise validation
errors.

Unitful parameters require one sibling `unit` applying to their value and
search range:

```yaml
parameters:
  a:
    unit: "pc"
    value: 500.0
    generator_settings:
      lower_bound: 100.0
      upper_bound: 1000.0
      step: 100.0
      minimum_step: 10.0
```

For a potential parameter, `value`, bounds, `step`, and `minimum_step` are all
coordinates in its declared unit. Parameter generators retain that unit, and
potential construction transforms a proposed `Quantity` only when a registered
parameterization needs it; the value first reaches a shared unit system inside
`Potential.to_galax()`, where `galax`'s own constructor converts it.

Dimensionless fields remain plain numbers and must not add a `unit`. Examples
include axial ratios, Gauss-Hermite coefficients, relative error factors, and
unitless warning thresholds.

## Dimension validation, not conversion

Dimension validation and conversion into orbit-integration units are
deliberately separate concerns. Every declared unit is checked for
*dimensional* correctness against a fixed per-dimension reference
(`tnt.units._REFERENCE_UNITS`) that is independent of any run's chosen unit
system; the value then keeps its declared or source unit.

Configuration preparation dimensionally validates:

- `cosmological_parameters.H` as inverse time;
- `system_attributes.distance` as length;
- explicit kinematics histogram `width` and `center` as speed;
- Gauss-Hermite systematic uncertainties `v` and `sigma` as speed;
- each configured potential component's raw parameters by their resolved
  dimension (e.g. `PlummerPotential`'s `m_tot` and `r_s` as mass and
  length -- see [Potential](potential.md)); and
- light-MGE potential parameter `ml` as mass divided by power.

It does not convert these quantities or remove their units. Each per-run
`config_repository/runs/<run_id>/resolved_config.yaml` retains the resolved
`{value, unit}` structures and unitful parameter declarations. The submitted
user profile remains transient input: TNT does not archive its source path or
bytes, but its declarations survive after defaults have been applied.

`ModelIterator.from_configuration()` wraps `cosmological_parameters`,
including `H`, into `Quantity` objects for runtime consumers such as NFW's
`concentration_m200` parameterization; those consumers let `unxt` handle any
conversion rather than assuming a declared unit. System distance remains a
declared quantity until a runtime consumer needs it; compatibility checks
compare its physical value directly from the preserved declaration.

Every run receives its own immutable resolved configuration, while resume
compatibility compares physical meaning. It recognizes atomic `{value, unit}`
declarations and converts one value to the other's unit only while comparing
that field. It does not normalize the complete configuration. Consequently,
declarations such as `1 kpc` and `1000 pc` compare equal, incompatible
dimensions compare unequal, and malformed declarations raise a compatibility
error. This exact comparison uses host-side numerical conversion rather than
JAX-backed `unxt.Quantity` equality, so its result is independent of the
configured JAX runtime precision. In particular, selecting 32-bit JAX
calculations cannot round away a small change in a preserved configuration
declaration.

## Runtime objects keep their source units

MGE contents and quantities read from observational data files are not
converted during configuration preparation -- those files are not opened at
that stage -- and they are not converted into a unit system when the runtime
objects are built either. Each constructor checks its columns' or metadata's
declared units for the right dimension and then keeps them:

- **MGE** (`tnt.mge`): the `I`, `sigma`, `q`, and `PA_twist` columns are read
  in the units the ECSV file declares. `build_mges()` still projects each MGE
  to physical units with `AbstractMGE.angular_to_physical()`, which works for
  any angular unit `sigma`/`I` were declared in.
- **Gauss-Hermite kinematics**: the `v`/`dv`/`sigma`/`dsigma` columns keep
  their ECSV units; the velocity column's unit is the local reference the
  quadrature errors and the auto-sized histogram are computed in.
- **Bayesian LOSVD**: the ECSV metadata's `velocity_unit` (applying to
  `vcent` and `dv`) is kept.
- **Proper motions**: the NPZ archive's scalar `velocity_unit` (applying to
  `vxrange` and `vyrange`) is kept.
- **Populations**: each `property`/`dproperty` pair must declare
  dimensionally equivalent units when unitful (any dimension is allowed); the
  uncertainty is converted into the *value column's* declared unit and that
  unit is kept. A pair without declared units is treated as dimensionless.
- **Projected spatial binning**: `min_x`, `min_y`, `x_extent`, `y_extent`,
  and `PA` are validated as angular quantities and kept in their declared
  units; `ProjectedBinning` does its grid geometry on demand in `min_x`'s
  unit, and `build_spatial_binnings()` then projects to physical units the
  same way `build_mges()` does.

Dimensionless moments, distributions, and relative uncertainties never carry
a unit.

The one place a shared, explicit unit system is genuinely needed is
`Potential.to_galax()` and its callers: `galax`'s potential classes require
an explicit `units=` argument, and converting every parameter and physical
constant into one fixed system once, outside the JIT-compiled hot loop, is
what that argument is for.
