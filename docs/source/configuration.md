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

Input and output paths are interpreted relative to a workspace root. Supply
one explicitly when the data should be rooted somewhere other than the
invoking script:

```python
config.read("user_config.yaml", workspace_root="/scratch/project/NGC6278")
```

This low-level API respects logging configured by the calling application but
does not start TNT's own handlers. Standalone workflows that want preparation
messages in the TNT logfile should use the lifecycle API:

```python
with tnt.configuration_session(
    "user_config.yaml",
    workspace_root="/scratch/project/NGC6278",
) as config:
    # Construct and execute the model while TNT logging remains active.
    ...
```

The lifecycle API loads the YAML and defaults once, bootstraps logging from the
output and logging settings, and then performs the same complete preparation.

### `Configuration.read()` versus `configuration_session()`

Both interfaces apply the same defaults and validation, return the same kind
of resolved `Configuration` object, and write the same configuration-repository
artifacts. Their difference is ownership of the logging lifecycle:

| Behavior | `Configuration.read()` | `configuration_session()` |
| --- | --- | --- |
| Resolves and validates configuration | Yes | Yes |
| Writes the configuration repository | Yes | Yes |
| Starts TNT logfile and terminal handlers | No | Yes |
| Logs configuration preparation | Only through logging already configured by the caller | Yes |
| Logs exceptions from subsequent model execution | No | Yes, while execution remains inside the `with` block |
| Cleans up TNT-created logging handlers | Not applicable | Yes, when the `with` block ends |

Use `Configuration.read()` when TNT is embedded in an application that owns
its logging configuration, or when preparation does not need a TNT logfile.
Use `configuration_session()` for a normal standalone TNT run so that
configuration preparation and model execution share one logfile. Code that
constructs and executes the model should remain inside the session's `with`
block.

`Configuration.read()` performs the following operations:

1. Loads the packaged `default_config.yaml` profile.
2. Recursively merges the user profile over the packaged profile.
3. Applies common defaults to every dynamically named component, parameter,
   and kinematics data set.
4. Applies defaults selected by each kinematics data set's `type`.
5. Validates the configured unit systems and converts supported unitful
   quantities to internal units.
6. Validates the resolved data without constructing runtime objects.
7. Preserves the original user YAML, portable resolved configuration, and run
   manifest below `<output_directory>/config_repository/`.

Mapping values are merged recursively. A user value replaces a default scalar
or list. User values always take precedence over applicable defaults.

The schema-only `dynamic_object_defaults` and `kinematics_type_defaults`
sections are applied during preparation and omitted from the resolved file.
Consequently, the generated YAML contains all scientific and numerical
settings needed by TNT without depending on the package defaults used during
the original preparation.

If a kinematics data set explicitly supplies complete histogram metadata
(`width`, `center`, and `bins`), that metadata replaces the type's histogram
derivation policy. Supplying only part of this explicit metadata is an error.

See [Units](units.md) for the internal and display unit systems, accepted
quantity syntax, and the fields currently normalized during preparation.

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
        v: {value: 0.0, unit: "km / s"}
        sigma: {value: 0.0, unit: "km / s"}
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
`io_settings.output_directory` strings. Both are interpreted relative to the
workspace root. If `workspace_root` is omitted, TNT uses the directory that
contains the invoking Python script. Interactive sessions, which have no
invoking script file, fall back to the current working directory. A relative
workspace-root argument itself is interpreted from the current working
directory.

For example:

```yaml
io_settings:
  input_directory: "input"
  output_directory: "output"
```

With a workspace root of `/scratch/project/NGC6278`, TNT uses
`/scratch/project/NGC6278/input` and
`/scratch/project/NGC6278/output` at runtime. Explicit absolute input and output
paths remain supported; TNT expresses them relative to the workspace root in
the portable snapshot. Choosing a common parent of input and output as the
workspace root therefore gives the most useful archived configuration.

`Configuration.data` and `Configuration.as_dict()` contain materialized
absolute input and output paths for runtime consumers. The corresponding
portable values are available through `Configuration.portable_data` and
`Configuration.as_portable_dict()`.

## Configuration repository

After successful validation, TNT writes three files atomically into
`<output_directory>/config_repository/`:

- `user_config.yaml` is a byte-for-byte copy of the submitted file, including
  its comments and formatting.
- `resolved_config.yaml` has all TNT defaults applied and stores input and
  output paths relative to the workspace root. It is intended to be moved to
  another machine and reused with a different workspace root.
- `run_manifest.yaml` records the absolute paths used for this preparation,
  checksums of both configuration files, TNT and dependency versions, the Git
  commit and dirty-working-tree state when available, Python and platform
  details, hostname, scheduler job identifiers when available, and random-seed
  state.

A negative configured orbit-library seed still means that execution must
generate a seed. Until that happens, the preparation manifest records the
effective seed as `null` with status `pending_generation`; the future execution
stage must update it once the actual seed is known.

Configuration preparation creates the output directory and its
`config_repository` subdirectory when necessary. The three repository files
are replaced on each successful preparation. It does not instantiate
components, load observational data, checksum observational inputs, or execute
modelling code.
