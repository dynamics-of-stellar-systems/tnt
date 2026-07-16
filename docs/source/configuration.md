# Configuration preparation

TNT handles configuration in two separate phases. The preparation phase reads
and resolves configuration data. A later execution phase will construct the
physical system and other runtime objects from that resolved data.

## Preparing a configuration

```python
import tnt

config = tnt.Configuration()
config.read("user_config.yaml")
```

This low-level API respects logging configured by the calling application but
does not start TNT's own handlers. Standalone workflows that want preparation
messages in the TNT logfile should use the lifecycle API:

```python
with tnt.configuration_session("user_config.yaml") as config:
    # Construct and execute the model while TNT logging remains active.
    ...
```

The lifecycle API loads the YAML and defaults once, bootstraps logging from the
output and logging settings, and then performs the same complete preparation.

`Configuration.read()` performs the following operations:

1. Loads the packaged `default_config.yaml` profile.
2. Recursively merges the user profile over the packaged profile.
3. Applies common defaults to every dynamically named component, parameter,
   and kinematics data set.
4. Applies defaults selected by each kinematics data set's `type`.
5. Validates the resolved data without constructing runtime objects.
6. Writes the fully resolved configuration to
   `<output_directory>/config_repository/resolved_config.yaml`.

Mapping values are merged recursively. A user value replaces a default scalar
or list. User values always take precedence over applicable defaults.

The schema-only `dynamic_object_defaults` and `kinematics_type_defaults`
sections are applied during preparation and omitted from the resolved file.
Consequently, the generated YAML is a self-contained runtime configuration.

If a kinematics data set explicitly supplies complete histogram metadata
(`width`, `center`, and `bins`), that metadata replaces the type's histogram
derivation policy. Supplying only part of this explicit metadata is an error.

## Per-data-set kinematics settings

Observational error policies and fitting order belong to each named kinematics
data set because observations from different instruments may require different
assumptions. Type-specific defaults supply neutral settings, which a data set
can override:

```yaml
kinematics:
  central_spectroscopy:
    type: "gauss_hermite"
    maximum_gh_order: 4
    observational_errors:
      systematic_uncertainties:
        v: 0.0
        sigma: 0.0
        h3: 0.0
        h4: 0.0

  proper_motion_catalogue:
    type: "proper_motions"
    observational_errors:
      variance_scale: 1.0
```

Gauss-Hermite systematic uncertainties are named explicitly and added in
quadrature to the corresponding observational uncertainties. Their keys must
cover `v`, `sigma`, and every coefficient through `maximum_gh_order`.
Proper-motion `variance_scale` multiplies error variances and must be positive;
its neutral value is `1.0`.

The former global `number_GH`, `GH_sys_err`, and `PM_sys_err_factor` fields are
not weight-solver settings in TNT and are rejected as unknown fields.

## Validation

Preparation rejects duplicate YAML keys, unknown fields, missing required
fields, incorrect value types, unsupported enumerated values, and inconsistent
tagged thresholds. It also checks data-only numerical constraints, including
parameter bounds, positive worker counts, orbit-grid limits, and positive odd
explicit histogram bin counts. Per-kinematics checks cover Gauss-Hermite order,
complete systematic-uncertainty mappings, and proper-motion variance scaling.

Errors identify the configuration path containing the invalid value. The
resolved file is written only after every preparation-stage check succeeds.

These checks do not instantiate system components, inspect observational data
or MGE files, or verify optional runtime dependencies. Those checks belong to
the later execution phase. TNT also does not reproduce warnings for deprecated
configuration fields from predecessor software; unsupported fields are
reported as errors instead.

## Paths and side effects

The user profile must define non-empty `io_settings.input_directory` and
`io_settings.output_directory` strings. Relative paths are interpreted from
the process working directory and stored as absolute paths in the resolved
configuration, so the execution phase does not depend on a later working
directory.

Configuration preparation creates the output directory and its
`config_repository` subdirectory when necessary. It atomically replaces
`resolved_config.yaml` on each successful read. It does not instantiate
components, load observational data, or execute modelling code.
