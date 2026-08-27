# Units

TNT uses [`unxt`](https://unxt.readthedocs.io/) to validate units and define
two related unit systems:

- `units.internal` defines the canonical units used by runtime converters and
  later numerical calculations. Potential parameter proposals are an explicit
  exception: they retain each parameter's declared unit until potential
  construction consumes them, as described below.
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
    power: "Lsun"
  display:
    angle: "arcsec"
    speed: "km / s"
```

TNT checks that every unit describes the dimension named by its key. For
example, using `Myr` as the internal length unit is an error. The resolved
configuration object exposes the constructed systems as
`config.unit_systems.internal` and `config.unit_systems.display`.

## Quantity syntax

Every unitful value must state its unit, even when that unit is the configured
internal unit and even when the value is zero. Standalone quantities use a
`value` and `unit` mapping:

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
coordinates in its declared unit. Parameter generators retain that unit rather
than eagerly converting the coordinates into the internal unit system;
potential construction converts or transforms a proposed `Quantity` only when
its native component or registered parameterization needs it.

Dimensionless fields remain plain numbers and must not add a `unit`. Examples
include axial ratios, Gauss-Hermite coefficients, relative error factors, and
unitless warning thresholds.

## Configuration validation and runtime conversion

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

Runtime constructors convert the settings they consume. Kinematics builders
convert explicit histogram width/center and Gauss-Hermite velocity systematic
uncertainties. Potential parameters are different: they keep their own
declared unit all the way through `AbstractParameterGenerator` and
`Potential` construction -- nothing coerces them into a shared internal unit
system, since `galax`'s own potential classes already convert generically at
evaluation time. Configuration preparation still validates that a declared
unit's *dimension* is correct (as listed above), but that check is against
each dimension's own fixed reference unit, not the configured internal unit
system (see [Potential](potential.md)). The resolved configuration itself
remains unchanged. `ModelIterator.from_configuration()` converts
`cosmological_parameters`, including `H`, into `Quantity` objects for runtime
consumers such as NFW's `concentration_m200` parameterization. System distance
remains a declared quantity until a runtime consumer needs it; compatibility
checks compare its physical value directly from the preserved declaration.

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

MGE (multi-Gaussian expansion) contents and quantities read from observational
data files are intentionally not converted during configuration preparation.
Those files are not opened at this stage. MGE, kinematics, and population
constructors validate their declared units and convert arrays into the internal
unit system.

Gauss-Hermite velocity columns declare units directly in ECSV. Bayesian LOSVD
ECSV metadata declares a `velocity_unit` applying to `vcent` and `dv`.
Proper-motion NPZ archives contain a scalar string `velocity_unit` applying to
`vxrange` and `vyrange`. Dimensionless moments, distributions, and relative
uncertainties do not carry a unit.

Population ECSV files may mix properties with different physical dimensions.
Each `property`/`dproperty` pair must declare equivalent units when unitful;
the pair is converted to the matching internal unit. A pair without declared
units is treated as dimensionless.

Projected spatial-binning geometry is also converted during runtime-object
construction rather than preparation. Its `min_x`, `min_y`, `x_extent`,
`y_extent`, and `PA` fields still use explicit `{value, unit}` mappings in the
resolved configuration. `ProjectedBinning.from_settings()` validates that
these are angular quantities and converts them to the internal angle unit when
`build_spatial_binnings()` loads the binning registry.
