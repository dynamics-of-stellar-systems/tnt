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

Both interfaces apply the same defaults and validation and return the same kind
of resolved `Configuration` object. Neither allocates a run or writes the
configuration repository; that happens only when a successfully constructed
`ModelIterator` starts `run()`. Their difference is ownership of the logging
lifecycle:

| Behavior | `Configuration.read()` | `configuration_session()` |
| --- | --- | --- |
| Resolves and validates configuration | Yes | Yes |
| Writes the configuration repository | No | No |
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
3. Applies common defaults to every dynamically named potential component and
   parameter.
4. Applies defaults selected by each kinematics data set's `type`.
5. Validates the configured unit systems and the dimensions of supported
   unitful quantities without converting or stripping their declarations.
6. Validates the generic resolved schema and registry references without
   constructing runtime objects. Type-specific kinematics and population-file
   validation is deferred to runtime construction.
7. Applies the resolved process-wide JAX precision policy before later runtime
   construction.
8. Retains runtime and portable forms in memory for later runtime construction
   and run archiving.

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
quantity syntax, and the runtime conversion boundaries.

## JAX numerical precision

TNT makes its JAX precision policy explicit under `numerics_settings`:

```yaml
numerics_settings:
  jax_enable_x64: true
```

The packaged default is `true`, enabling 64-bit JAX values for TNT's
scientific calculations. Setting it to `false` selects JAX's 32-bit mode,
which can reduce memory use and improve throughput on some accelerators at the
cost of numerical precision.

JAX precision is process-wide rather than local to one model or
`Configuration` object. Importing `tnt` establishes the packaged default
before importing TNT's JAX-backed modules. A successful `Configuration.read()`
or `configuration_session()` then applies the resolved setting after complete
configuration validation and before runtime-object construction. Construct
TNT runtime objects only after preparing their configuration. JAX arrays that
another library created earlier in the process retain their original dtype;
changing the policy cannot convert existing arrays.

Repeated configuration reads and multiple `ModelIterator.run()` calls may
share one process when they use the same precision policy. Once one resolved
TNT configuration has established the process policy, reading a configuration
with the opposite value raises an error; start a new Python process for that
configuration instead. The setting is also resume-critical, so a model set
created under one precision policy cannot be resumed under the other.

This runtime policy does not control comparison of preserved configuration
quantity declarations. Resume compatibility deliberately performs that exact,
host-side comparison independently of JAX precision, so selecting 32-bit
runtime calculations cannot hide a small configuration edit.

## Scientific input registries

TNT keeps reusable data definitions separate from model components. Entries
refer to one another by name:

```yaml
MGEs:
  stellar_light: "mge_lum.ecsv"

spatial_binnings:
  central_bins:
    min_x: {value: -29.5, unit: "arcsec"}
    min_y: {value: -26.5, unit: "arcsec"}
    x_extent: {value: 58.0, unit: "arcsec"}
    y_extent: {value: 52.0, unit: "arcsec"}
    PA: {value: 126.0, unit: "deg"}
    bins_file: "bins.npy"

potential:
  stars:
    type: "TriaxialLightMGEPotential"
    mge: "stellar_light"
    parameters:
      ml:
        value: 5.0
        unit: "Msun / Lsun"
      theta: {value: 1.0, unit: "rad"}
      phi: {value: 0.5, unit: "rad"}
      psi: {value: 0.0, unit: "rad"}

kinematic_data:
  central_spectroscopy:
    type: "gauss_hermite"
    data_file: "gauss_hermite.ecsv"
    binning: "central_bins"
    mge: "stellar_light"

population_data:
  stellar_populations:
    data_file: "populations.ecsv"
    binning: "central_bins"
```

`MGEs` is the named multi-Gaussian expansion file registry.
`spatial_binnings` lets kinematic and population data share one projected
aperture definition. The four coordinate fields define a regular rectangular
pixel grid, while `PA` gives the galaxy major-axis position angle measured
counterclockwise from the grid's y-axis (the Cappellari/van den Bosch
convention also used by an MGE's `PA_twist`), not the grid's x-axis. Each is
an explicit angular
`{value, unit}` quantity. `bins_file` is resolved relative to
`io_settings.input_directory` and must contain a two-dimensional NumPy array
with shape `(npix_x, npix_y)`. Its non-negative integers assign pixels to bins;
ID 0 marks pixels that are not assigned to a bin, and the positive IDs must be
contiguous, running `1, 2, ..., n_bins` with no gaps. The pixel counts are
inferred from the array shape. A kinematic data set may optionally reference
an MGE; this is not required for proper-motion data.

Population observations must always be supplied through their own
`population_data.<name>.data_file`. TNT does not support population columns
embedded in a kinematics data file, even when both data sets use the same
`spatial_binnings` entry.

The supported potential types are `TriaxialLightMGEPotential`,
`TriaxialMassMGEPotential`, `AxisymmetricLightMGEPotential`,
`AxisymmetricMassMGEPotential`, and a curated set of `galax.potential` class
names (see [Potential](potential.md)). A light-MGE potential requires an
`ml` mass-to-light parameter. A mass-MGE potential must not declare `ml`,
because its input MGE already represents mass. Triaxial MGE types also
require `theta`/`phi`/`psi`, the global viewing angles the named MGE is
deprojected under; axisymmetric MGE types require a single `inclination`
instead. MGE contents and their physical units are inspected only in the
later object-construction phase.

## Loading configured MGEs

Configuration preparation validates the named `MGEs` registry but does not
open MGE files or construct scientific objects. After resolving a
configuration, load the registered MGEs explicitly:

```python
from tnt import Configuration
from tnt.mge import build_mges
from tnt.units import resolve_system_distance

config = Configuration().read("configuration.yaml")
resolved = config.as_dict()

mges = build_mges(
    resolved["MGEs"],
    resolved["io_settings"]["input_directory"],
    config.unit_systems.internal,
    resolve_system_distance(resolved["system_attributes"]),
)
```

The returned dictionary uses the configured MGE names as keys. Each value is
a `LightMGE` or `MassMGE`, inferred from the physical unit of the ECSV `I`
column, and already converted from angular to physical units via
`angular_to_physical(distance)`. File contents and units are validated
during this runtime-loading step, so loading can fail even after
configuration preparation succeeded.

## Loading configured spatial binnings

Like MGEs, spatial binnings become scientific objects only after configuration
preparation. Build the named registry explicitly from the resolved settings:

```python
from tnt import Configuration
from tnt.spatial_binnings import build_spatial_binnings
from tnt.units import resolve_system_distance

config = Configuration().read("configuration.yaml")
resolved = config.as_dict()

binnings = build_spatial_binnings(
    resolved["spatial_binnings"],
    resolved["io_settings"]["input_directory"],
    config.unit_systems.internal,
    resolved["mge_settings"]["projected_mass_quad_order"],
    resolve_system_distance(resolved["system_attributes"]),
)
```

The returned dictionary uses the configured binning names as keys and contains
`ProjectedBinning` values. This loading step opens each `.npy` file, validates
the exact entry schema and inline geometry, and rejects empty, negative, or
non-contiguous pixel-to-bin arrays. It converts the geometry to the internal
angle unit, then to physical units via `angular_to_physical(distance)` --
matching `build_mges` above, so that MGEs and spatial binnings loaded this way
are always dimensionally consistent -- and precomputes pixel edges and
quadrature nodes.

`AbstractMGE.get_projected_mass()` integrates an MGE over a
`ProjectedBinning`; both must be angular or both physical.

The fixed Gauss-Legendre quadrature orders are configured under
`mge_settings`. `intrinsic_mass_quad_order` applies to intrinsic
three-dimensional integration on a `SphericalGrid`, and
`projected_mass_quad_order` applies to projected pixel integration. Both must
be positive integers; higher orders trade additional computation for greater
integration accuracy.

## Per-data-set kinematics settings

Observational error policies and fitting order belong to each named kinematics
data set because observations from different instruments may require different
assumptions. Type-specific defaults supply neutral settings, which a data set
can override:

```yaml
kinematic_data:
  central_spectroscopy:
    type: "gauss_hermite"
    binning: "central_bins"
    data_file: "gauss_hermite.ecsv"
    maximum_gh_order: 4
    observational_errors:
      systematic_uncertainties:
        v: {value: 0.0, unit: "km / s"}
        sigma: {value: 0.0, unit: "km / s"}
        h3: 0.0
        h4: 0.0

  proper_motion_catalogue:
    type: "proper_motions"
    binning: "central_bins"
    data_file: "proper_motions.ecsv"
    observational_errors:
      variance_scale: 1.0
```

Gauss-Hermite systematic uncertainties are named explicitly and added in
quadrature to the corresponding observational uncertainties. Their keys must
cover `v`, `sigma`, and every coefficient through `maximum_gh_order`.
Proper-motion `variance_scale` multiplies error variances and must be positive;
its neutral value is `1.0`.

## Constructing kinematics

After configuration preparation, build the observational runtime objects from
the resolved registries:

```python
from tnt.kinematics import build_kinematics
from tnt.mge import build_mges
from tnt.spatial_binnings import build_spatial_binnings
from tnt.units import resolve_system_distance

input_directory = config.data["io_settings"]["input_directory"]
unit_system = config.unit_systems.internal
distance = resolve_system_distance(config.data["system_attributes"])
mges = build_mges(config.data["MGEs"], input_directory, unit_system, distance)
spatial_binnings = build_spatial_binnings(
    config.data["spatial_binnings"],
    input_directory,
    unit_system,
    config.data["mge_settings"]["projected_mass_quad_order"],
    distance,
)

kinematics = build_kinematics(
    config.data["kinematic_data"],
    input_directory,
    unit_system,
    spatial_binnings,
    mges,
)
```

Each kinematics object retains the referenced shared objects rather than
reopening or duplicating them. An MGE reference remains optional; a
spatial-binning reference is required. Runtime construction strictly requires
each binning reference to resolve to a `ProjectedBinning` and each MGE
reference to resolve to a `LightMGE` or `MassMGE`.

Gauss-Hermite and Bayesian LOSVD inputs are ECSV files. Gauss-Hermite files
contain `bin_id`, unitful `v`, `dv`, `sigma`, and `dsigma` columns, followed by
dimensionless `hN` and `dhN` pairs. Bayesian LOSVD files contain
`bin_id`, `bin_flux`, and paired `losvd_N`/`dlosvd_N` columns; their
metadata declares `vcent`, `dv`, and `velocity_unit`.

Proper-motion inputs use an NPZ archive containing `PM_2dhist`,
`PM_2dhist_sigma`, `bin_id`, `nstarbin`, `vxrange`, `vyrange`, and the
scalar string `velocity_unit`. TNT validates odd two-dimensional velocity-bin
counts and positive uncertainties, normalizes each spatial-bin distribution,
and applies `variance_scale` to its error variances.

Every kinematics input must cover the referenced `ProjectedBinning` exactly:
`bin_id` is a positive, unique integer vector whose values are the complete
set ``1, 2, ..., n_bins``. Row order is unrestricted because `bin_id`
identifies each row. ID 0 is reserved for unbinned pixels in the bin map and
must not occur in observational data.

## Constructing populations

Population runtime objects use the same already-built spatial-binning
registry, but they do not use the MGE registry:

```python
from tnt.populations import build_populations

populations = build_populations(
    config.data["population_data"],
    input_directory,
    unit_system,
    spatial_binnings,
)
```

Each returned `Populations` object retains its shared `ProjectedBinning` and
stores its observations as JAX-backed `unxt.Quantity` arrays. Under the same
convention as kinematics inputs, a population ECSV file requires a `bin_id`
column whose positive, unique integer values cover every positive ID in the
referenced binning exactly once. It also requires at least one paired
population property and uncertainty, for example `age`/`dage` or
`metallicity`/`dmetallicity`. Property names are otherwise unrestricted.
Declared units on each pair must be equivalent and are converted into the
internal unit system; columns without declared units are dimensionless. All
values must be finite and all uncertainties must be strictly positive.

Population objects do not contain an MGE. A population file must also be
different from every configured kinematics file; sharing only the
`spatial_binnings` entry is supported.

## Validation

Preparation rejects duplicate YAML keys, generic unknown fields, missing
required registry fields, incorrect generic value types, unsupported type
identifiers owned by the preparation schema, and inconsistent tagged
thresholds. A potential component's curated `galax` class name and optional
parameterization are resolved later by `tnt.potential` during runtime-object
construction. Preparation also checks non-kinematics data-only numerical
constraints, including parameter bounds, positive worker counts, and orbit-grid
limits. Concrete kinematics and population constructors
check explicit histogram bin counts, Gauss-Hermite order and
systematic-uncertainty mappings, Bayesian LOSVD policies, proper-motion
variance scaling and warning thresholds, population value/uncertainty pairs,
and all observational file contents. Preparation still checks that references
from potentials and observational data resolve to existing MGE and
spatial-binning entries.

Preparation collects spatial-binning names for those cross-reference checks,
but deliberately leaves each entry's geometry, units, `bins_file`, and loaded
array validation to `ProjectedBinning.from_settings()` and
`build_spatial_binnings()`. Consequently, a malformed spatial-binning entry can
pass preparation, but runtime construction rejects non-mapping entries,
missing or unknown fields, invalid `bins_file` values, malformed quantities,
and empty or invalid bin arrays with a named-entry error.

Execution worker counts are currently reserved settings: preparation validates
and records them, but they do not yet control runtime scheduling. Only
`execution_settings.model_processing_order: model_by_model` is executable;
`stage_by_stage` raises `NotImplementedError`. See
[Model search](model_search.md) for the current execution-setting support.

## Potential rescaling

Potential rescaling is configured independently of the ordinary `ml`
parameter:

```yaml
parameter_space_settings:
  potential_rescalings:
    enabled: true
    range_count: 10
    mass_scale_range:
      minimum: 0.1
      maximum: 10.0
    spacing: "logarithmic"
```

The scaling factor applies to the complete assembled potential, after its
ordinary component parameters—including `ml`—have been applied. `spacing`
accepts `linear` or `logarithmic`; the minimum and maximum are inclusive.

When `enabled` is false, TNT retains and validates the remaining settings, but
the later execution phase produces only the unscaled model. See
[Model search](model_search.md) for how rescaled potentials are evaluated and
recorded at runtime.

Errors identify the configuration path containing the invalid value. No
resolved file or run manifest is written during preparation.

Preparation does not instantiate system components, inspect observational data
or MGE files, or verify optional runtime dependencies. Those checks belong to
the later execution phase. Unsupported fields are reported as errors.

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
the archived resolved configuration. Choosing a common parent of input and output as the
workspace root therefore gives the most useful archived configuration.

`Configuration.data` and `Configuration.as_dict()` contain materialized
absolute input and output paths for runtime consumers. The corresponding
portable values are available through `Configuration.portable_data` and
`Configuration.as_portable_dict()`. `Configuration.logfile_path` records the
active session logfile for later inclusion in a run manifest, or is `None`
when the caller owns logging.

## Configuration repository

After `ModelIterator.from_configuration()` has successfully constructed every
runtime object, each subsequent `run()` invocation that passes its state and
resume preflight publishes immutable artifacts under
`<output_directory>/config_repository/`:

```text
config_repository/
├── runs/
│   ├── 0000/
│   │   ├── run_manifest.yaml
│   │   └── resolved_config.yaml
│   └── 0001/
│       ├── run_manifest.yaml
│       └── resolved_config.yaml
└── run_config_log.ecsv
```

- `runs/` contains one numbered, immutable directory per TNT run. Its
  `resolved_config.yaml` has all defaults applied, preserves declared units,
  and stores input and output paths relative to the workspace root. Its
  `run_manifest.yaml` records the run ID, the resolved-configuration path, TNT
  and dependency versions, Git state, Python/platform/host context, scheduler
  identifiers, logfile location, and random-seed state. Repeated identical
  configurations are archived independently because run directories are
  intentionally not deduplicated.
- The submitted user profile, its source path, and its bytes are not archived.
  TNT also does not create configuration-content or scientific-input hashes.

Numeric directory names provide stable human-readable run IDs. After `run()`
starts, `ModelIterator.run_manifest` identifies the immutable bundle and
`ModelIterator.run_id` contains its numeric run ID. Both are `None` before the
first invocation and identify the latest invocation after sequential calls.
Earlier provenance remains accessible through `RunConfigLog` and the
repository. `Configuration.source_path` remains available only in memory for
the active process.

TNT supports exactly one coordinating process writing a given output directory
at a time. Model calculations may use multiple workers, but workers must return
their results to that coordinator rather than modifying shared repository or
checkpoint files. Concurrent coordinators using the same output directory are
unsupported.

(run-identity)=
### Run identity

One invocation of `ModelIterator.run()` is exactly one TNT run. A run identity
is allocated only after `ModelIterator.from_configuration()` has successfully
constructed every configured MGE, spatial binning, observational data set,
population, potential component, and model-search service. A configuration
that fails runtime construction is therefore never archived. Sequential
`run()` calls on the same iterator are supported; every call repeats the
state/resume preflight, receives a fresh run ID, and archives the resolved
configuration again. This intentional duplication distinguishes execution
attempts even when they share one configuration and process.

`run_config_log.ecsv` is created when the model-search caller explicitly
persists `AllModels` and `RunConfigLog` through the coordinated
`ModelSearchState` writer; configuration preparation itself does not create or
update the log. It maps each cumulative search iteration to a run ID. The
corresponding immutable run directory contains that run's resolved
configuration and execution provenance. Its derived ECSV metadata
records `total_runs` and `run_ids_without_iterations`; callers must persist the
state even when a run produces no iteration so that these summaries are
updated.

A negative configured orbit-library seed means that execution must
generate a seed. Until that happens, the run manifest records the effective
seed as `null` with status `pending_generation`; the future execution stage
must update it once the actual seed is known.

`Configuration.read()` does not create the output directory or
`config_repository`; `configuration_session()` may create the output and log
directories for its logging lifecycle. `ModelIterator.run()` creates the
repository when needed and atomically publishes its manifest and resolved
configuration as one run directory after preflight state and
resume-compatibility checks succeed. Existing artifacts are never replaced.
