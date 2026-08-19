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
3. Applies common defaults to every dynamically named potential component and
   parameter.
4. Applies defaults selected by each kinematics data set's `type`.
5. Validates the configured unit systems and converts supported unitful
   quantities to internal units.
6. Validates the generic resolved schema and registry references without
   constructing runtime objects. Type-specific kinematics and population-file
   validation is deferred to runtime construction.
7. Preserves the portable resolved configuration and a run-specific manifest
   below
   `<output_directory>/config_repository/`.

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
    type: "triaxial_light_mge"
    mge: "stellar_light"
    parameters:
      ml:
        value: 5.0
        unit: "Msun / Lsun"

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

The supported potential types are `triaxial_light_mge`,
`triaxial_mass_mge`, `nfw`, and `plummer`. A light-MGE potential requires an
`ml` mass-to-light parameter. A mass-MGE potential must not declare `ml`,
because its input MGE already represents mass. MGE contents and their physical
units are inspected only in the later object-construction phase.

## Loading configured MGEs

Configuration preparation validates the named `MGEs` registry but does not
open MGE files or construct scientific objects. After resolving a
configuration, load the registered MGEs explicitly:

```python
from tnt import Configuration
from tnt.mge import build_mges

config = Configuration().read("configuration.yaml")
resolved = config.as_dict()

mges = build_mges(
    resolved["MGEs"],
    resolved["io_settings"]["input_directory"],
    config.unit_systems.internal,
)
```

The returned dictionary uses the configured MGE names as keys. Each value is
a `LightMGE` or `MassMGE`, inferred from the physical unit of the ECSV `I`
column. File contents and units are validated during this runtime-loading
step, so loading can fail even after configuration preparation succeeded.

## Loading configured spatial binnings

Like MGEs, spatial binnings become scientific objects only after configuration
preparation. Build the named registry explicitly from the resolved settings:

```python
from tnt import Configuration
from tnt.spatial_binnings import build_spatial_binnings

config = Configuration().read("configuration.yaml")
resolved = config.as_dict()

binnings = build_spatial_binnings(
    resolved["spatial_binnings"],
    resolved["io_settings"]["input_directory"],
    config.unit_systems.internal,
    resolved["mge_settings"]["projected_mass_quad_order"],
)
```

The returned dictionary uses the configured binning names as keys and contains
`ProjectedBinning` values. This loading step opens each `.npy` file, validates
the exact entry schema and inline geometry, and rejects empty, negative, or
non-contiguous pixel-to-bin arrays. It converts the geometry to the internal
angle unit and precomputes pixel edges and quadrature nodes.

`AbstractMGE.get_projected_mass()` integrates an MGE over a
`ProjectedBinning` and returns the total in each positive bin ID. Convert both
objects with their `angular_to_physical()` methods before integration when
working in physical rather than angular coordinates.

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

input_directory = config.data["io_settings"]["input_directory"]
unit_system = config.unit_systems.internal
mges = build_mges(config.data["MGEs"], input_directory, unit_system)
spatial_binnings = build_spatial_binnings(
    config.data["spatial_binnings"],
    input_directory,
    unit_system,
    config.data["mge_settings"]["projected_mass_quad_order"],
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
contain `vbin_id`, unitful `v`, `dv`, `sigma`, and `dsigma` columns, followed by
dimensionless `hN` and `dhN` pairs. Bayesian LOSVD files contain
`binID_dynamite`, `bin_flux`, and paired `losvd_N`/`dlosvd_N` columns; their
metadata declares `vcent`, `dv`, and `velocity_unit`.

Proper-motion inputs use an NPZ archive containing `PM_2dhist`,
`PM_2dhist_sigma`, `binID_dynamite`, `nstarbin`, `vxrange`, `vyrange`, and the
scalar string `velocity_unit`. TNT validates odd two-dimensional velocity-bin
counts and positive uncertainties, normalizes each spatial-bin distribution,
and applies `variance_scale` to its error variances.

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
stores its observations as JAX-backed `unxt.Quantity` arrays. A population
ECSV file requires a positive unique `vbin_id` column and at least one paired
population property and uncertainty, for example `age`/`dage` or
`metallicity`/`dmetallicity`. Property names are otherwise unrestricted.
Declared units on each pair must be equivalent and are converted into the
internal unit system; columns without declared units are dimensionless. All
values must be finite and all uncertainties must be strictly positive.

Population bin IDs may cover a subset of the positive IDs in the referenced
binning, but they cannot refer to IDs absent from it. Population objects do not
contain an MGE. A population file must also be different from every configured
kinematics file; sharing only the `spatial_binnings` entry is supported.

The former global `number_GH`, `GH_sys_err`, and `PM_sys_err_factor` fields are
not weight-solver settings in TNT and are rejected as unknown fields.

## Validation

Preparation rejects duplicate YAML keys, generic unknown fields, missing
required registry fields, incorrect generic value types, unsupported type
identifiers, and inconsistent tagged thresholds. It also checks non-kinematics
data-only numerical constraints, including parameter bounds, positive worker
counts, and orbit-grid limits. Concrete kinematics and population constructors
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

Errors identify the configuration path containing the invalid value. The
resolved file is written only after every preparation-stage check succeeds.

Preparation does not instantiate system components, inspect observational data
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

After successful validation, TNT publishes immutable artifacts under
`<output_directory>/config_repository/`:

```text
config_repository/
├── configurations/
│   └── 0000-a81c09f3/
│       ├── resolved_config.yaml
│       └── compatibility_signature.yaml
├── manifests/
│   └── 0000-run_manifest.yaml
└── run_config_log.ecsv
```

- `configurations/` holds semantic configuration versions. Each resolved file
  has all TNT defaults applied and stores input and output paths relative to
  the workspace root. TNT hashes a canonical representation in which mapping
  order and YAML presentation do not matter, while list order and values do.
  An existing snapshot is reused when this semantic hash matches.
  Runtime construction later adds `compatibility_signature.yaml` beside the
  resolved file, after TNT has loaded and validated the referenced scientific
  inputs. The signature records the versioned resume contract,
  compatibility-critical resolved settings, and SHA-256 hashes of MGE,
  spatial-binning, kinematics, and population files. Configuration preparation
  itself does not open or checksum those inputs.
- `manifests/` receives one new manifest for every TNT run. A manifest records
  its run ID and configuration-snapshot identifiers, paths relative to the
  configuration-repository root, resolved-file and semantic configuration
  hashes, TNT and dependency versions, Git state, Python and platform details,
  hostname, scheduler identifiers, logfile location, and random-seed state.
  The submitted user profile, its source path, and its byte hash are not
  persisted.

Numeric prefixes provide stable human-readable identifiers. Short hash
prefixes in artifact names aid inspection; manifests retain the complete
SHA-256 hashes. `Configuration.resolved_path` and
`Configuration.run_manifest_path` identify the artifacts selected or created
by the current run. `Configuration.run_id` records the numeric run ID.
`Configuration.source_path` remains available only in memory for the active
process.

(run-identity)=
### Run identity

A successful `Configuration.read()` defines the start of a TNT run for
provenance purposes: it creates a new immutable run manifest and assigns the
next run ID. Reusing that prepared `Configuration` and its `ModelIterator`
across multiple `ModelIterator.run()` calls retains the same run ID. Reading a
configuration again creates another run manifest and assigns another run ID,
even when the resolved configuration is semantically identical and therefore
reuses an existing configuration snapshot.

Configuration snapshots and run manifests consequently have different
identities. A snapshot represents one distinct resolved configuration; a run
manifest represents one TNT execution session that used a snapshot. The
future top-level execution layer should preserve this boundary by reading the
configuration once at the beginning of each invocation and using the resulting
run ID for every model-search iteration in that invocation.

`run_config_log.ecsv` is created when the model-search caller explicitly
persists `AllModels` and `RunConfigLog` through the coordinated
`ModelSearchState` writer; configuration preparation itself does not create or
update the log. It maps each cumulative search iteration to a run ID. The
corresponding immutable run manifest links that ID to its resolved
configuration snapshot and execution provenance. Its derived ECSV metadata
records `total_runs` and `run_ids_without_iterations`; callers must persist the
state even when a run produces no iteration so that these summaries are
updated.

A negative configured orbit-library seed still means that execution must
generate a seed. Until that happens, the preparation manifest records the
effective seed as `null` with status `pending_generation`; the future execution
stage must update it once the actual seed is known.

Configuration preparation creates the output directory and its
`config_repository` subdirectories when necessary. Existing artifacts are
never replaced. Repeating an identical run configuration therefore reuses the
same resolved configuration file while creating a new run manifest. It does
not instantiate components, load observational data, checksum observational
inputs, or execute modelling code.
