# Units

TNT uses [`unxt`](https://unxt.readthedocs.io/) to validate units and define
two related unit systems:

- `units.internal` controls the canonical units used by resolved
  configuration values and later numerical calculations.
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

kinematics:
  central_spectroscopy:
    histogram:
      center: {value: 0.0, unit: "km / s"}

cosmological_parameters:
  H0:
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

For a linear parameter, TNT converts the value, bounds, step, and minimum step.
For a logarithmic parameter, the declared unit is the reference unit for the
logarithm: TNT shifts the value and bounds into the internal reference unit,
while logarithmic step sizes remain unchanged.

Dimensionless fields remain plain numbers and must not add a `unit`. Examples
include axial ratios, Gauss-Hermite coefficients, relative error factors, and
unitless warning thresholds.

## Currently unit-aware fields

Configuration preparation currently normalizes:

- `cosmological_parameters.H0` to inverse internal time;
- `system_attributes.distance` to internal length;
- explicit kinematics histogram `width` and `center` to internal speed;
- Gauss-Hermite systematic uncertainties `v` and `sigma` to internal speed;
- Plummer parameters `m` and `a` to internal mass and length; and
- the system parameter `ml` to internal mass divided by internal power.

The byte-for-byte `user_config.yaml` archive retains the user's notation. The
generated `resolved_config.yaml` contains plain numbers in internal units, so
runtime code does not need to repeat conversions and the YAML remains portable
and easy to serialize.

MGE (multi-Gaussian expansion) contents and quantities read from observational
data files are intentionally not converted during configuration preparation.
Those files are not opened at this stage; their unit handling belongs to the
later object-construction and data-loading phase.
