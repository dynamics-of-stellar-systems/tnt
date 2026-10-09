# Project Knowledge

## Python style

- TNT requires Python 3.12 or newer. `.python-version` selects the 3.12
  baseline for local development, and CI tests supported Python versions
  beginning with 3.12.
- Follow the PEP 8 style guide whenever practical.
- Use double quotes for Python strings.
- Add type hints for function parameter types and return types.
- Use the Google style for function and class docstrings.
- Keep individual methods to 100 lines or fewer as a soft limit; exceeding it
  slightly is acceptable when necessary.

## Module layout

- `tnt/validation.py` holds shared helpers for validating
  resolved configuration data (mapping/required-field/reject-unknown/
  string/number checks, named cross-reference resolution, bin-ID reading).
  If a helper like this ends up reimplemented in more than one module,
  move the shared logic there instead of leaving the copies to drift --
  see its module docstring for the full rationale and existing contents
  before adding a near-duplicate.
- `tnt/configuration/` groups configuration resolution and preservation
  (`core.py`), preparation-time schema validation (`validation.py`), and
  resume compatibility (`compatibility.py`).
- `tnt/registry.py` provides the small shared class-registration mechanic used
  by configured runtime-object families. Each family owns its registry,
  public lookup helpers, and domain-specific decorator; the shared helper
  only enforces an explicitly declared, unique, non-empty `_type`.
- `tnt/kinematics/` keeps shared base objects in `base.py`, one concrete data
  family per module, explicit registration in `registry.py`, and construction
  orchestration in `__init__.py`.
- `tnt/potential/` separates curated type metadata (`registry.py`), NFW
  parameterization mathematics (`nfw.py`), the component hierarchy
  (`components.py`), and whole-potential orchestration (`core.py`). Its
  `__init__.py` defines the intended package-level API; implementation-specific
  names remain in their owning submodules.
- TNT's own potential-component types (as opposed to curated native `galax`
  types) register themselves with `tnt.potential.registry.register_component`,
  applied directly to each concrete `AbstractPotentialComponent` subclass's
  definition (e.g. `tnt.potential.triaxial_mge`). The decorator registers the
  class under its own `_type`; schema access reads `_raw_dimensions` from that
  registered class. Runtime dispatch uses its lookup helper, while
  configuration validation uses its public type predicate and schema accessors
  rather than reading private registry state. A class participates only when
  it is explicitly decorated.
  `tnt/potential/__init__.py` must import every concrete component module so
  its decorators run during normal package initialization.
- `tnt.configuration`'s "does not construct scientific objects" boundary
  permits imports needed to read static registration metadata. Import-time
  registration is allowed; configuration preparation must not instantiate
  scientific objects, load scientific input data, or begin scientific
  execution.

## Angular reference frames

- An MGE's photometric orientation and an observational data grid's
  orientation are independently measurable and frequently different
  (kinematic/photometric misalignment is a real triaxiality signature, not
  noise). TNT keeps them as separate config fields rather than one shared
  position angle: `MGEs.<name>.major_axis_pa` (the on-sky PA
  `PA_twist` is measured from, in `[0, 180)` degrees -- an axis, not a
  direction) and `spatial_binnings.<name>.y_axis_pa` (the on-sky PA of the
  grid's +y axis, in `[0, 360)` degrees). Both domains are half-open and
  enforced by rejection (not normalization -- declarations are preserved
  verbatim for resume compatibility) via `tnt.units.validate_position_angle`,
  at the config boundary (`_validate_mges`, `ProjectedBinning.from_settings`)
  and at runtime construction (`AbstractMGE.from_qtable`). See
  `docs/source/data_preparation.md` for the full geometric picture.
- `ProjectedBinning` declares only `y_axis_pa` (in `[0, 360)` degrees), not
  an x-axis PA too: TNT fixes the grid's parity by convention -- the positive
  x-axis is always 90 degrees east of positive y
  (`AbstractMGE.get_projected_mass`'s `alpha = y_axis_pa + pi/2 -
  major_axis_pa - PA_twist`). Data with the opposite parity (x pointing west
  of y -- the traditional FITS-display convention, north up/east left) must
  be converted before use: reverse `bins` along axis 0, set `min_x` to
  `-(min_x + x_extent)`, and negate every x-directed vector quantity (e.g.
  proper-motion `vx`). See `docs/source/data_preparation.md`.
- `AbstractMGE.major_axis_pa` is a required field, not optional: `PA_twist`
  is defined as a twist *away from* `major_axis_pa`, so an MGE without one has
  no absolute orientation, the same way it wouldn't make sense to construct
  one without `sigma`. `LightMGE.read`/`MassMGE.read`/`read_mge` all require
  it as an explicit argument too (it isn't read from the ECSV); only
  `tnt.mge.build_mges` reads it from configuration.
- UNVERIFIED: the NGC6278 integration fixture
  (`tests/integration_tests/fixtures/bins.npy`, with `y_axis_pa: 0`) has an
  unconfirmed on-sky parity. It was rasterised from DYNAMITE
  `aperture.dat`/`bins.dat`, and DYNAMITE stores the bin map in the input-data
  frame without derotation (the aperture `angle` records only the major-axis
  PA; the legacy `90 - PA` relation implies grid +y is north). CALIFA DR3
  cubes are delivered north-up/east-left (Sanchez et al. 2016, Fig. 5), i.e.
  the *opposite* parity to TNT -- so if the map kept the cube's native pixel
  grid it needs the `bins[::-1, :]` + `min_x` conversion and currently does
  not have it. Whether it does depends on the sign convention chosen when the
  kinematic map was built, which isn't recorded. No current test asserts
  projected masses, so this is inert today; check the fixture as a possible
  error source when first comparing TNT projections against DYNAMITE output.

## Linux development container

- `Dockerfile` and `compose.yaml` provide the reproducible Linux `x86_64`
  development environment used from Intel macOS. The host checkout is mounted
  at `/workspace`; its macOS `.venv` is never used in the container because
  `UV_PROJECT_ENVIRONMENT` points to `/opt/tnt-venv` inside the image.
- For this checkout's local macOS workflow, use Docker's `colima` context.
  The local Colima VM has 4 GB of memory. Run scientific test suites
  sequentially, preferably in separate processes; do not run multiple JAX
  test processes in parallel. Accumulated compiled graphs can also exhaust
  memory within one process. The native MGE gradient tests clear JAX's
  compilation caches between cases to bound their memory use.
  Example: `docker --context colima compose run --rm dev pytest -q
  tests/unit_tests/test_mge.py`.
- Run `docker compose build` after dependency or container-definition changes.
  Normal source edits are immediately visible without rebuilding.
- Use `docker compose run --rm dev <command>` for Linux validation, for example
  `pytest -q`, `ruff check .`, or
  `sphinx-build -E -b html -W docs/source docs/build/html`. Omitting the command
  opens an interactive shell with the TNT environment on `PATH`.

## Configuration defaults

- The packaged base profile is `tnt/defaults/default_config.yaml`.
- TNT will recursively merge the base profile with a user profile before
  constructing a model.
- Configuration preparation is implemented by `tnt.Configuration`. Its
  `read()` method loads the user YAML, recursively merges package defaults,
  resolves dynamic and kinematics-type defaults, validates the resulting
  data, and retains runtime and portable representations in memory. It does
  not instantiate scientific runtime objects, allocate a run ID, or write the
  configuration repository.
- Preparation-stage validation rejects duplicate keys, unknown or missing
  fields in preparation-owned schemas, invalid types and enumerations,
  malformed tagged thresholds, and basic numerical inconsistencies before the
  resolved file is written. Spatial-binning entry fields are an explicit
  exception: preparation collects their names for cross-reference validation,
  while `ProjectedBinning.from_settings()` and `build_spatial_binnings()` own
  their exact entry schema, geometry, units, `bins_file`, and loaded-array
  validation. Runtime construction rejects non-mapping entries, missing and
  unknown fields, invalid filenames, and empty or otherwise invalid bin maps.
  Concrete kinematics and population constructors likewise validate their
  observational file contents and type-specific runtime rules. Other
  runtime-object, MGE-content, and optional-dependency checks remain the
  responsibility of the execution phase.
  Unknown fields raise validation errors rather than being ignored.
- TNT uses `unxt` for configuration units. `units.internal` is *only* the
  unit system handed to `galax` in `Potential.to_galax()` -- not a
  normalization applied to declared values or file data, which keep their
  own units. It requires exactly `length`, `time`, `mass`, and `angle`:
  the dimensions `galax`'s potential types need that `unxt` can't derive
  for you. `unxt` builds `power`/`speed`/`frequency`/... from
  mass/length/time automatically, so derived dimensions must not be declared
  in `units.internal`. `angle` is dimensionally independent (`unxt` can't
  decompose `rad`) and a real native parameter dimension for some `galax`
  types, so it is required. `units.display` may override `power` and `speed`
  for presentation. `Configuration.unit_systems` exposes both constructed
  systems.
- Every known unitful configuration value must state its unit explicitly;
  internal units and zero values are not implicit exceptions. Standalone
  quantities use `{value: ..., unit: ...}`. Unitful parameter definitions use
  a required sibling `unit` applying to the parameter value and generator
  range. Dimensionless values remain plain numbers and reject a `unit`.
  Configuration preparation validates their dimensions without converting or
  stripping them; per-run resolved configurations preserve the `{value, unit}`
  declarations. The submitted profile is transient input and is not archived
  by TNT.
- The first unit-aware schema covers `cosmological_parameters.H`,
  `system_attributes.distance`, explicit kinematics histogram width and center,
  Gauss-Hermite `v` and `sigma` systematic uncertainties,
  `PlummerPotential`'s native `m_tot` and `r_s`, and light-MGE potential `ml`.
  Their runtime handling is consumer-specific rather than one blanket
  normalization step; in particular, potential parameter values, bounds, and
  steps remain expressed in their declared unit.
- Runtime kinematics construction validates configured histogram quantities
  and Gauss-Hermite velocity systematics and keeps their declared units.
  Computations that combine equivalent quantities convert them on demand into
  the data set's local reference unit. Potential parameters likewise keep
  their declared unit through `AbstractParameterGenerator` and `Potential`
  construction -- `ModelIterator.from_configuration()` does not create a
  shared internal-unit copy. `galax` converts potential parameters when
  `Potential.to_galax()` constructs its runtime potential, so they do not need
  pre-normalization (see `tnt.potential`'s module docstring).
  `ModelIterator.from_configuration()` also converts
  `cosmological_parameters`, including `H`, into `Quantity` objects for
  runtime consumers such as NFW's `concentration_m200` parameterization.
  System distance remains a declared quantity until a runtime consumer needs
  it.
- `tnt.configuration.compatibility._critical_configuration` projects the
  preserved resolved configuration without normalizing it. Its recursive
  comparator treats complete `{value, unit}` mappings as atomic quantities,
  converts one value to the other's unit only for that comparison, and requires
  exact numerical equality after conversion. Incompatible dimensions are
  differences; malformed declarations raise `ConfigurationCompatibilityError`.
  At the run boundary, `Configuration.read()` does not archive, and
  `ModelIterator.from_configuration()` successfully constructs every
  runtime object before `run()` can publish the configuration. Malformed
  runtime-owned fields such as `spatial_binnings.*.min_x` therefore fail before
  any run bundle exists.
- Validation ownership depends on the field rather than one blanket unit
  policy. `spatial_binnings` has no prep-time quantity check because runtime
  construction owns its complete entry schema. Kinematics histogram and
  systematic-uncertainty fields are checked independently at preparation and
  construction time. Potential-parameter dimensions are checked during
  preparation by
  `tnt.configuration.validation.validate_configuration_quantities`; parameter
  generation and potential construction then preserve the declared unit. No
  one of these construction paths normalizes values into `units.internal`.
- MGE contents and quantities inside observational files are deliberately
  deferred to the object-construction/data-loading phase. Configuration
  preparation does not open those files. `tnt.kinematics.build_kinematics`
  constructs named `GaussHermite`, `BayesLOSVD`, and `ProperMotions` objects;
  it validates each column's/metadata's declared unit for the right dimension
  (against a fixed reference, not any unit system) and keeps it, retaining
  JAX arrays in immutable Equinox modules. Population observations are loaded
  separately by `tnt.populations.build_populations()`.
- `tnt.mge.build_mges()` is the explicit runtime boundary that loads the
  resolved `MGEs` registry into named `LightMGE` and `MassMGE` objects.
  `Configuration` continues to contain no instantiated scientific objects.
  MGE ECSV `I` is physical surface brightness/density (`Lsun/pc2` or
  `Msun/pc2`, or equivalent units); `sigma` is angular, `q` dimensionless,
  and `PA_twist` angular. Runtime loading converts only `sigma` using the
  system distance and preserves `I`. For fixed angular widths, total
  luminosity/mass scales with distance squared. Direct constructors may
  already carry physical widths. See `docs/source/data_preparation.md`.
- Native MGE deprojection has paired eager and `with_validity` APIs.
  All four supported MGE potential types construct inside one JAX trace using
  native normalization and viewing angles. The complete proposal returns one
  scalar boolean; invalid intrinsic MGEs contain zeros and must not be used,
  including through `to_galax`, unless the flag is true. Eager and traced paths
  share intrinsic-axis, finite positive density/width/mass, and numerical
  accuracy checks. Native potential construction additionally probes positive
  finite values in local `Msun`/`kpc` units. It does not compute or certify
  construction derivatives on each proposal; regression tests cover gradients
  for representable proposals (equivalent units, integer columns,
  finite-difference comparisons) only. An extreme MGE column value can pass
  as valid with a non-finite gradient.
  Oblate cancellation and triaxial
  covariance-inversion conditioning/residual checks use `50*sqrt(eps)`
  relative error thresholds at active precision. Exactly circular projected
  rows use the analytic spherical result, avoiding roundoff beyond q=1 or u=1;
  see `docs/source/potential.md`.
  Surface intensity stays physical and projected total mass is conserved.
  `Deprojected3DMGE.component_masses` owns the intrinsic Gaussian mass
  calculation shared by validation and both Galax construction paths. It
  converts locally to `kpc` and `Msun` (or `Lsun` for luminosity), interleaves
  density/width products, and preserves the conversion boundary against
  compiler reassociation. Its shared `gaussian_widths` property supplies
  floating-point `kpc` widths to both Galax factories with the same conversion
  boundary. Stored MGE columns and proposal units remain unchanged.
  Gaussian-width arithmetic uses floating-point arrays
  even when fixed input columns contain integers. Equivalent physical widths
  in `pc`, `kpc`, and `km` must yield consistent validity, mass, potential
  values, and gradients; finite
  integrated mass alone does not guarantee representable derivatives.
  The native-MGE validity contract covers deprojection and construction
  values. By explicit scope decision, it does not certify Galax's
  fixed-order potential quadrature. Galax's 50-point Gaussian quadrature can
  be inaccurate for very thin Gaussians: at oblate q=0.001 the normalized
  central potential differs from the analytic arccos(q)/sqrt(1-q**2)
  reference by about 0.7%, and its q derivative by about 95%, even with a
  well-resolved float64 deprojection. Potential-quadrature accuracy needs
  separate work; do not treat a true construction flag as that guarantee.
  MGE `q_min`, `pqu`, and `T_maj_min` conversions are traceable. The standalone
  scientific methods and registry adapters share numerical candidates and
  predicates, including anchor twists, endpoint margins and round-trip checks.
  Their standalone `with_validity` counterparts use the same
  `_guard_construction` scaffold as potential construction. Numerical validity
  does not certify derivatives: edge-on `q_min` remains accepted despite its
  unbounded conversion derivative, and inclusive `pqu` compression endpoints
  retain their precision-margin clamp and clipped compression derivative.
  Regression tests cover interior recovered-shape identity derivatives and
  these endpoint limitations; per-proposal derivative checks remain removed.
  Shape constraints compare unit-free ratios, including percent declarations;
  inverse reporting restores the declared shape-coordinate units. In
  `ParameterConstraint`, `unit=""` means a unit-free comparison and only
  `unit=None` selects the proposed value's own unit.
  `Potential.build_with_validity` is the only potential construction interface;
  `build_potential`, `Potential.build`, `Potential.from_settings`, component
  `build`, and the four MGE `_build` factories have been removed.
  `ModelIterator._evaluate()` checks the scalar flag before any use of the
  potential, logs a generic numerical-validation rejection with raw parameters,
  and records an invalid model. Static setup errors propagate. Construction
  supports `vmap`; iterator batching remains deferred (prior integration is
  done -- see `tnt.priors`/`PriorSampler` below). The iterator still returns
  a variable-length `list[Model]`; orbit integration
  and weight solving remain scaffolding.
- Issue #72 fixes the execution target as one proposal evaluated inside a JAX
  trace. `ParameterConstraint.valid()` now exposes JAX scalar predicates for
  registered numeric bounds and same-component relationships; eager
  `violation()` uses those same predicates for its diagnostics.
  `ResolvedPotentialComponent._raw_parameters_valid()` checks static
  names/types/dimensions/shapes before tracing and returns a JAX scalar flag
  for finite raw values and registered bounds. A complete proposal must contain
  exactly the resolved component names. `Potential.build_with_validity()`
  composes native Galax and MGE components and their flags inside a JAX trace,
  including traceable registered conversions such as NFW's `concentration_m200`;
  a false flag requires JAX conditional execution before evaluating derived
  quantities. Registered conversions first check raw validity, then probe
  native finiteness and constraints without differentiation. Only proposals
  passing both checks run the differentiable conversion. This prevents
  invalid derived values from contaminating gradients even when raw values
  are finite and positive. Abstract evaluation supplies the output units and
  dtypes for zero placeholders used by invalid converted components; these
  are not usable physical models. Converters must themselves support JAX
  tracing, and their output names/types/dimensions/scalar shapes remain hard
  contract checks. Iterator batching is deferred (prior integration is
  done -- see `tnt.priors`/`PriorSampler` below).
- Eager constraint diagnostics and traced validity evaluate the same JAX
  predicates at the proposed value's active precision; converting eager values
  to Python floats would change half-open bound decisions in float32. For
  `StoneOstriker15Potential`, `r_h > r_c` also requires
  `(r_h - r_c) / max(abs(r_h), abs(r_c)) > eps**(1/5)` at that precision.
  The upstream potential formula subtracts nearly equal terms and otherwise
  yields unreliable gradients close to equal radii. This numerical guard is
  shared by ordinary and compiled calls to the same builder.
- Intel macOS is not a native TNT target because current JAX releases do not
  provide `jaxlib` wheels for that platform. Use the Linux `x86_64`
  development container there instead.
- Retrieve units from `unxt` unit systems by physical dimension rather than
  generated attribute names so TNT does not depend on convenience-name changes
  between `unxt` releases.
- Mapping values merge recursively, while user scalars and lists replace
  defaults. User values have final precedence. Schema-only
  `dynamic_object_defaults` and `kinematics_type_defaults` sections are
  removed after their values have been applied.
- Complete explicit kinematics histogram metadata (`width`, `center`, and
  `bins`) replaces the histogram derivation policy for that data set.
- Relative input and output paths are interpreted from an explicit workspace
  root. When omitted, the root is the invoking Python script's directory, with
  the process working directory used only for interactive sessions. Runtime
  configuration data materializes both paths as absolute; the resolved YAML
  stores them relative to the workspace root.
- `Configuration.data` and `as_dict()` expose runtime paths;
  `portable_data` and `as_portable_dict()` expose the portable form retained
  for later archiving. `Configuration.read()` requires both path strings but
  does not create the output directory or configuration repository;
  `configuration_session()` may create the output/log directory to start its
  logging lifecycle.
- The configuration repository stores one immutable bundle per TNT run under
  `runs/<run_id>/`, containing `run_manifest.yaml` and
  `resolved_config.yaml`. One invocation of `ModelIterator.run()` is exactly
  one run; it allocates and publishes the bundle only after
  `ModelIterator.from_configuration()` has successfully constructed all
  runtime objects and its state/resume preflight checks pass. Sequential calls
  on the same iterator are allowed, and each receives a fresh run ID and
  archive. `ModelIterator.run_id` and `run_manifest` identify the latest call;
  earlier provenance remains in `RunConfigLog` and the repository. Identical
  configurations are therefore archived again for separate calls; TNT
  performs no cross-run deduplication and stores no configuration or
  scientific-input hashes. The
  submitted user profile and source path remain transient. Each manifest
  records software versions, Git state, Python/platform/host context,
  scheduler identifiers, logfile location, and orbit random-seed state.
- Exactly one coordinating TNT process may write a given output directory.
  Parallel workers may calculate models, but only the coordinator may update
  shared repository or checkpoint files. Scientific input files must not be
  modified in place while an existing model set may be resumed; TNT does not
  hash their contents and therefore cannot detect such changes.
- `RunConfigLog` persists separately as
  `config_repository/run_config_log.ecsv`, with one row per cumulative
  model-search iteration. Rows map iterations to run IDs. Reads and atomic
  writes validate the referenced immutable run manifests; those manifests are
  the authoritative links to per-run resolved configurations and execution
  provenance. ECSV metadata derives `total_runs` and
  `run_ids_without_iterations` from all manifests and the iteration rows on
  every read or write. `ModelIterator.run()` returns both `AllModels` and this
  log without writing them, so the execution layer must load and save them
  together, including after a zero-iteration run. The log records provenance
  only; it does not implement the configuration-compatibility decision itself.
- `RunConfigLog` metadata refresh deliberately performs one O(M) scan of the
  M per-run manifests on every log read or write. Keep this scan unless
  profiling shows that it materially affects checkpoint
  time; run counts are expected to be small relative to model-calculation
  costs. If optimization becomes necessary, first make metadata refresh use a
  lightweight numeric run-directory scan instead of introducing a persistent
  index with additional synchronization and recovery rules.
- `ModelSearchState` is the coordinated persistence boundary for `AllModels`
  and `RunConfigLog`. It validates both temporary ECSV files, atomically
  replaces each file in run-log-first order, explicitly repairs a log-ahead
  crash state by truncating unpublished trailing rows, and rejects a
  models-ahead state because missing provenance is not recoverable. An initial
  zero-model checkpoint writes only the run log because `AllModels` has no
  column schema until its first model.
- Before resuming, runtime compares the current compatibility-critical
  configuration directly with the archived resolved configuration from the
  earliest run that contributed an iteration. The contract excludes
  operational/search/presentation fields and potential parameter
  values/units/ranges.
  It includes internal units, cosmology, physical system attributes except
  name, potential/parameter schema, MGE and observational settings including
  their configured file references, all `numerics_settings`, orbit-library
  settings, and weight-solver settings. `which_chi2` must be finite for every
  successful historical model, and the required potential parameter columns
  must exist. Negative configured orbit seeds are valid for fresh and
  continued runs; changing the configured seed between runs remains
  incompatible. Complete unit-bearing compatibility fields are compared by
  physical value on demand, so equivalent declarations such as `1 kpc` and
  `1000 pc` compare equal without an eager configuration-wide traversal.
  Comparison is exact, not tolerance-based, by design: the question this
  check answers is "did the human change anything," and a config field that
  changes unit between runs almost always changes value too, so exactness
  correctly flags real edits rather than hiding them. Floating-point
  unit-conversion noise (e.g. through angle units or composite units
  involving irrational factors) is a real property of the conversion
  arithmetic but not a practical risk here, since it would only bite two
  independently-authored declarations of the identical physical value in
  different units -- not how config files are actually edited between runs.
  The comparator intentionally keeps this conversion in host-side
  NumPy/Astropy `float64` arithmetic rather than constructing JAX-backed
  `unxt.Quantity` objects. Do not replace it with direct `Quantity` equality:
  `numerics_settings.jax_enable_x64: false` would then make compatibility
  comparison lose small declared differences to 32-bit rounding. Preserved
  configuration identity must remain independent of runtime precision.
- The compatibility check runs once at the start of each `run()` invocation,
  after runtime construction but before allocating that call's new run
  identity or modifying model-search state. It cannot run in
  `Configuration.read()` because the selected chi-square and model-table
  schema checks require the previous search state.
- A negative seed is recorded as `pending_generation` in the run manifest;
  the execution phase must update the effective seed.
- The user profile must define the physical system, dynamically named
  potential components and parameters, input directory, and output directory.
- TNT user profiles generally use snake-case type identifiers and field names.
  The established `MGEs` registry name is a current schema exception. A
  parameter's search-space declaration belongs under `prior` as
  `{distribution: "<numpyro.distributions class>", args: [...]}` (replaced
  `generator_settings`'s `lower_bound`/`upper_bound`/`step`/`minimum_step` --
  `step`/`minimum_step` had no consumer and were dropped rather than carried
  forward, matching how `logarithmic` was removed for the same reason, see
  below); display labels use `latex_label`. A `prior` is required whenever
  `fixed` is false: `_validate_parameters` (`tnt.configuration.validation`)
  raises if a `fixed: false` parameter has no `prior`, since
  `tnt.priors._build_model` would otherwise silently skip it (neither a
  sample site nor a fixed value), surfacing only as a missing-parameter
  error at `Potential.build_with_validity`. The packaged
  `dynamic_object_defaults.parameter.fixed` default stays `false`, so a user
  profile declares `fixed: true` or a `prior` on every parameter.
- Scientific inputs use independent named registries: `MGEs` maps names to
  `{file, major_axis_pa}` entries; `spatial_binnings` maps names to inline
  rectangular aperture geometry (`min_x`, `min_y`, `x_extent`, `y_extent`, and
  `y_axis_pa`) plus a `bins_file` containing a 2D NumPy pixel-to-bin map;
  `potential` defines potential components; `kinematic_data` references a
  binning and optionally an MGE; and `population_data` references a binning.
  Preparation validates all cross-references without opening the files.
- Population observations always use a separate
  `population_data.<name>.data_file`, even when the population and kinematics
  data share a spatial binning.
- `tnt.populations.build_populations()` loads each configured population ECSV
  into an immutable JAX/Equinox `Populations` object. It resolves a strictly
  typed `ProjectedBinning` but no MGE. Files require a `bin_id` column and one
  or more `property`/`dproperty` column pairs. Paired units must be
  dimensionally equivalent (any dimension); the uncertainty is converted into
  the value column's declared unit and that unit is kept, unitless columns
  remain dimensionless, and uncertainties must be positive. The shared
  observational bin-ID rule below applies.
- `tnt.spatial_binnings.build_spatial_binnings()` is the explicit runtime
  boundary that loads the resolved `spatial_binnings` registry into named
  `ProjectedBinning` objects. It validates the complete entry before file
  access, validates the loaded non-empty bin array, validates that each
  coordinate's declared unit is an angle and keeps it (the grid geometry is
  done on demand in `min_x`'s unit), and precomputes pixel quadrature.
- `AbstractMGE.get_projected_mass()` integrates projected MGE totals into the
  positive bin IDs of a `ProjectedBinning`; bin ID 0 is excluded. The MGE and
  binning coordinate units must be dimensionally consistent.
- `build_kinematics` requires already-built `ProjectedBinning` objects and
  optional `LightMGE`/`MassMGE` objects, resolves each data set's named
  references to those shared runtime objects, and returns a name-to-object
  mapping. Its runtime boundary rejects incorrectly typed registry values.
- `AbstractKinematics.binning` is strictly a `ProjectedBinning`; its optional
  `mge` is strictly a `LightMGE` or `MassMGE`. Each concrete kinematics class
  owns its configuration identifier in `_type` and explicitly registers via
  `tnt.kinematics.registry.register_kinematics`. Both runtime dispatch and
  configuration validation read the kinematics-owned registry through public
  accessors. Undecorated subclasses do not silently become configuration
  types, and duplicate or inherited `_type` declarations fail at registration.
- `AbstractKinematics.design_matrix()` defines the weight-solver projection
  boundary introduced by the model-architecture scaffold. It deliberately
  raises `NotImplementedError` until orbit integration and the concrete
  kinematics projections are implemented; observational values and
  uncertainties already have a shared base-class interface.
- Every kinematics and population input uses `bin_id`. Its positive, unique
  integer values must cover every positive ID in the referenced
  `ProjectedBinning` exactly once, although row order is unrestricted. ID 0
  represents unbinned pixels in the bin map and is invalid in observations.
- Gauss-Hermite ECSV files require `bin_id`, unitful `v`, `dv`, `sigma`, and
  `dsigma` columns plus dimensionless `hN`/`dhN` pairs. Configured systematic
  uncertainties are added in quadrature. Missing higher-order pairs are
  represented by zero coefficients only when the corresponding configured
  systematic uncertainty is positive.
- Bayesian LOSVD ECSV files use `bin_id`, `bin_flux`, `losvd_N`, and
  `dlosvd_N` columns. Metadata must contain `vcent`, `dv`, and an explicit
  `velocity_unit`; TNT converts the velocity grid and applies the configured
  flux-weighted systemic centering.
- Proper-motion NPZ input contains `PM_2dhist`, `PM_2dhist_sigma`,
  `bin_id`, `nstarbin`, `vxrange`, and `vyrange`, plus a required
  scalar `velocity_unit`. Construction validates and normalizes each 2D
  distribution, scales uncertainties by the square root of `variance_scale`,
  and emits configured sampling warnings.
- `potential.<name>.type` names one of a curated set of `galax.potential`
  classes (`tnt.potential._SUPPORTED_GALAX_TYPES`, e.g. `NFWPotential`,
  `PlummerPotential` -- 25 classes total), or one of four TNT-specific MGE
  composite types -- triaxial (`TriaxialLightMGEPotential`/
  `TriaxialMassMGEPotential`, `tnt/potential/triaxial_mge.py`) or oblate
  axisymmetric (`OblateLightMGEPotential`/`OblateMassMGEPotential`,
  `tnt/potential/oblate_mge.py`) -- provided directly by TNT since `galax`
  has no native class for the MGE-specific deprojection/composition wiring.
  A light type requires an `ml` parameter; a mass type requires
  `mge_mass_scale` instead (validation rejects `ml` on it, since its MGE
  already contains mass). Triaxial types also require `theta`/`phi`/`psi`;
  oblate types require a single `inclination` in `(0, 90]` deg. TNT's
  axisymmetric deprojection is oblate-only; prolate would get its own
  `Prolate...` types. Each type's exact
  parameter schema comes straight from its own registered `_raw_dimensions`
  (see `register_component`), not a second hand-written list.
  Deliberately curated rather than "any `AbstractPotential` subclass":
  `galax` also exports abstract/base classes, which cannot be constructed as
  concrete TNT potential components;
  pre-packaged multi-component bundles with no free parameters of their own
  like `MilkyWayPotential`/`LM10Potential` (their `disk`/`bulge`/`halo`/
  `nucleus` fields are themselves sub-potentials, not `ParameterField`s --
  redundant with TNT's own multi-component `potential:` section anyway),
  wrapper/transform decorators needing a required nested potential object
  (e.g. `TranslatedPotential`, `FlattenedInThePotential` -- these do carry
  their own `ParameterField`s, but the required nested potential still
  isn't representable), and classes needing a required non-`Quantity`
  hyperparameter (`MultipolePotential`'s `l_max: int`) -- none
  representable by the scalar `parameters.<name>.value` schema. Checked
  directly against every one of galax's 45
  `AbstractPotential` subclasses: 28 have every field either a scalar
  `ParameterField` or a galax-provided default (safe under the current
  schema); the curated 25 drops `HenonHeilesPotential`/`NullPotential`
  (not astrophysically relevant to TNT) and `AbstractCompositePotential`
  (an empty-parameter base class) from that 28.
- Config-prep validation (`_validate_potential`) checks the declared
  `parameters` are *exactly* the resolved `type`/`parameterization`'s set --
  missing or unexpected names rejected -- for curated native `galax` types
  and registered parameterizations as well as TNT's own MGE composite types.
  `registry.raw_parameter_dimensions(kind, parameterization)` supplies the
  authoritative set; `registry.parameter_schema_is_known(kind,
  parameterization)` gates it (an unrecognized type or an unimplemented
  parameterization is left to `resolve()`'s own error, not reported as
  "every parameter is extra"). Native fields with a `galax` constructor
  default (e.g. `TriaxialHernquistPotential`'s `q1`/`q2`) must still be
  declared -- intentional TNT policy, for a complete/reproducible
  model-table schema.
- Runtime potential construction owns value/domain validation.
  `ResolvedPotentialComponent.build_with_validity()` checks exactly named,
  dimensionally correct scalar `Quantity` inputs as static setup, then returns
  a flag for finiteness and physical-domain checks before/after conversion.
  It does not normalize declared units. MGE construction and standalone
  deprojection share `_guard_construction` for detached probing and conditional
  differentiable construction. Native `galax` constraints live beside metadata in
  `_SUPPORTED_GALAX_TYPES`; TNT composite constraints live on each component's
  `_constraints`; parameterization raw constraints live in the same registered
  `ParameterizationSpec` as its converters and schema. Registration rejects
  constraint names or relationships that disagree with their owning schema.
  `components._check_parameter_set_structure()` enforces the
  generator/converter structure contract. Each `ParameterConstraint.valid()`
  owns its numerical bound and relationship predicates; `violation()` uses
  those predicates for standalone diagnostics. Numerical proposal failures
  return `valid=False`, which `ModelIterator._evaluate()` records as an
  invalid model. Malformed names, shapes, types, and dimensions remain static
  setup errors and propagate to the caller, including
  `InvalidPotentialParametersError` and `TypeError`.
  TNT's chosen physical policy is strict positivity for every mass, MGE
  normalization (`ml`/`mge_mass_scale`), and scale length, including parameters
  such as Miyamoto-Nagai `a`; zero does not disable a component or select a
  limiting profile. `MonariEtAl2016BarPotential.alpha` and its pattern speed
  `Omega` deliberately remain signed, including negative values.
- Non-native parameterizations register via `registry.register_parameterization(
  type_name=, name=, convert=, invert=, raw_dimensions=, raw_constraints=)` --
  one call, from the module owning the numerics (`tnt.potential.nfw` for
  `concentration_m200`, `tnt.potential.triaxial_mge` for `pqu`), mirroring
  `register_component`. It bundles the forward/inverse converters, config
  parameter schema, and raw domain rules in a single `ParameterizationSpec`,
  so validation and runtime resolution can't disagree on which
  parameterizations exist. Read back via `get_parameterization(type, name)` /
  `parameterization_names(type)`. `type_name` may be a curated native `galax`
  type OR a registered TNT component type (`is_registered_component_type`).
  Config validation, `resolve()`, and inverse dispatch are all generic (they
  key `_PARAMETERIZATION_REGISTRY` by `(type, name)` with no galax
  assumption; issue #65). `AbstractPotentialComponent.raw_parameters` looks
  itself up via `_registry_type_name()` (`self._type` by default,
  `GalaxPotentialComponent` overrides it to `self.galax_type`) and runs the
  registered `invert` converter -- so a *new* type picks up correct
  `AllModels` reporting the moment a parameterization is registered for it,
  no per-type override needed. `ForwardConverter`/`InverseConverter` take a
  trailing optional `mge` arg -- `None` for a curated native `galax` type,
  else the component's own `tnt.mge` MGE (every MGE composite type stores it
  in a field named `mge`) -- supplied generically, not per type:
  `ResolvedPotentialComponent.build_with_validity` passes `extra_fields.get("mge")` to the
  forward converter, `AbstractPotentialComponent.raw_parameters` passes
  `getattr(self, "mge", None)` -- the same value, off the built component --
  to the inverse one. `pqu` uses it for `q' = min(component q)` and the
  anchor twist; `concentration_m200` ignores it.
- `pqu` (the two triaxial MGE types): `(p, q, u)` intrinsic axis ratios /
  compression <-> `(theta, phi, psi)` viewing angles, van den Bosch et al.
  2008 MNRAS 385, 647 (= DYNAMITE `triax_pqu2tpp`). Both directions live on
  `AbstractMGE`: `triaxial_viewing_angles(p, q, u) -> (theta, phi, psi)` and
  its inverse `triaxial_intrinsic_shape(theta, phi, psi) -> (p, q, u)` (the
  anchor slice of `deproject_triaxial`). Anchor `q' = min` component `q`; the
  anchor Gaussian's `PA_twist` is folded out of `psi` by
  `triaxial_viewing_angles` and back in by `triaxial_intrinsic_shape`, so
  `(p, q, u)` keep their meaning for a twisted MGE. Ties for min `q'` break by
  component order. `_pqu_to_tpp` / `_tpp_to_pqu` in `tnt.potential.triaxial_mge`
  share traceable numerical candidates with the standalone scientific methods.
  Failed forward conversions produce nonfinite angles for the shared builder
  to reject with `valid=False`; standalone methods retain diagnostic exceptions.
  A `pqu` config and its equivalent
  `(theta, phi, psi)` config build an identical potential. Data-independent
  bounds (`0 < q <= p <= 1`, `p < u <= 1`) are `ParameterConstraint`s;
  `triaxial_viewing_angles` additionally rejects a value violating `q < p`
  (prolate) or `max(q/q', p) < u <= min(p/q', 1)`, a degenerate weight, and a
  domain so narrow it has no representable interior point. The de Zeeuw &
  Franx weights are singular exactly on the `u` boundaries; the *lower*
  endpoints (`u = p`, `u = q/q'`) are excluded, the *upper* endpoints
  (`u = 1`, `u = p/q'`) are inclusive limiting geometries evaluated one
  margin of `4*sqrt(eps)` inside `min(p/q', 1)` -- `eps` for the working JAX
  float type. At float32 that margin is `~1.4e-3`, so a declared `u = 1` is
  honoured only to about that and `triaxial_intrinsic_shape` reports the
  recovered value, not an exact `1`.
- `T_maj_min` (the same two triaxial MGE types): a second, bijective
  reparameterization of `pqu`'s own `(p, q, u)` as `(T, T_maj, T_min) in
  [0,1]^3` (Quenneville, Liepold & Ma 2022, ApJ 926:30, sec. 3 eqs. 3-4, 7),
  chosen for more uniform shape/viewing-geometry sampling, not a different
  deprojection. The `(T,T_maj,T_min) <-> (p,q,u)` algebra (given the anchor's
  `q'`) is `tnt.mge._p_q_u_from_T_Tmaj_Tmin` / `_T_Tmaj_Tmin_from_p_q_u`,
  each guarding its own denominator (`eps`-scaled) before dividing;
  `AbstractMGE.viewing_angles_from_T_Tmaj_Tmin` /
  `T_Tmaj_Tmin_from_viewing_angles` compose that with `triaxial_viewing_angles`
  / `triaxial_intrinsic_shape`, inheriting all of `pqu`'s domain/margin/
  singularity handling for the `(theta, phi, psi)` side. `_tmajmin_to_tpp` /
  `_tpp_to_tmajmin` in `tnt.potential.triaxial_mge` are the registry adapters,
  mirroring `_pqu_to_tpp` / `_tpp_to_pqu`. `T`, `T_maj`, `T_min` are each a
  closed `[0,1]` `ParameterConstraint`; no pairwise relation is needed at
  schema level (unlike `pqu`'s `q <= p`).
  `viewing_angles_from_T_Tmaj_Tmin` additionally checks that its result
  round-trips: `pqu`'s own `u`-margin clamp (previous bullet) is negligible
  in `(p,q,u)` space, but `(T,T_maj,T_min)` divide by `1 - p**2` and
  `p**2 - q**2`, so the same clamp can move the *requested* shape
  coordinates far more than it moved `u`. Each coordinate is accepted only
  if it round-trips (forward then `T_Tmaj_Tmin_from_viewing_angles`) within
  `_TMAJMIN_ROUNDTRIP_ABS_TOL_FACTOR * eps + _TMAJMIN_ROUNDTRIP_REL_TOL_FACTOR
  * sqrt(eps) * |coordinate|` (a combined bound, not relative alone, since a
  requested coordinate can legitimately be exactly `0`); otherwise
  the standalone method raises `MGEDeprojectionError`, while the potential
  builder returns `valid=False`. At float64 this is essentially never triggered by
  an ordinary point; at float32 it can reject points with a small
  `T`/`T_maj`/`T_min` whose `(p,q,u)` sits close enough to `pqu`'s own
  singular boundary -- calibrated against measured round-trip drift
  (ordinary points stay under `~6e-6` relative at float32; degenerate ones
  measured `32%-168%`), not guessed.
- `q_min` (the two oblate MGE types): the oblate counterpart of `pqu` --
  the anchor Gaussian's intrinsic axial ratio <-> the single global
  `inclination`, via `deproject_oblate`'s own relation `q_obs'^2 = q_min^2
  sin(i)^2 + cos(i)^2` at the anchor `q_obs' = min(component q)`
  (`_triaxial_anchor`; its `PA_twist` element is unused here since
  `deproject_oblate` requires every component's twist to be zero). Both
  directions are `AbstractMGE` methods: `inclination_from_q_min(q_min) ->
  inclination` and its inverse `q_min_from_inclination(inclination) ->
  q_min`, which reads the anchor's own intrinsic `q` using shared oblate
  geometry checks rather than integrating the unscaled template's luminosity
  or mass. Forward shape conversion is also geometry-only; construction checks
  density and mass after the proposal's normalization. Unit/twist/domain rules
  remain shared with deprojection. `_qmin_to_inclination` / `_inclination_to_qmin` in
  `tnt.potential.oblate_mge` are the registry adapters, mirroring
  `_pqu_to_tpp` / `_tpp_to_pqu`. Data-independent bound: `0 < q_min <= 1`
  (`ParameterConstraint`); `inclination_from_q_min` additionally rejects a
  circular anchor (`q_obs' == 1`), `q_min` outside `0 < q_min <= q_obs'`,
  and `q_min` too close to 1 to divide by reliably at the working precision
  (`eps`-scaled, same style as `pqu`'s own guards). Unlike `pqu`'s `u`
  boundary, `q_min == q_obs'` (edge-on, `i = 90 deg`) needs no value-level
  precision margin. Its inclination derivative is unbounded; acceptance
  certifies numerical values only.
  The forward conversion also deprojects at the computed inclination and
  checks `abs(q_recovered - q_min) / q_min <= 50 * sqrt(eps)`, where `eps`
  is for the active JAX float type. This is a relative shape-error ceiling:
  approximately `7.45e-7` at float64 and `0.0173` (1.73%) at float32.
  Exceeding it raises `MGEDeprojectionError` in the standalone method and
  returns `valid=False` from the potential builder. Thin or nearly circular configurations
  can fail this check even inside the mathematical domain. Reporting uses
  the recovered shape, so accepted values need not equal inputs exactly.
- `parameterization` is a separate, optional field controlling how config
  `parameters` map onto a component's canonical fields. Omitted, raw
  parameter names must match the resolved `type`'s own native `galax`
  constructor kwargs exactly; production reads their physical dimensions
  directly from `_SUPPORTED_GALAX_TYPES` (each entry a
  `NativeParameter(dimension, exponent, constraint)`). Dynamic derivation from
  `galax`'s
  own `ParameterField(dimensions=...)` metadata is a test-local helper in
  `tests/unit_tests/test_potential.py`
  (`test_supported_galax_types_covers_every_curated_class_parameter` cross-checks
  the curated table against it).
  `GalaxPotentialComponent.rescale()` scales every native parameter by
  `mass_scale ** exponent`, where `exponent` is curated per (class,
  parameter) directly in `_SUPPORTED_GALAX_TYPES` -- not derived from
  dimension, since a parameter's role determines its exponent as much as
  its dimension does: `MonariEtAl2016BarPotential`'s `Omega` (bar pattern
  speed, dimension `"frequency"`) and `v0` (sets the potential's amplitude,
  dimension `"speed"`) share the same time-power but need opposite
  exponents (0.0 vs 0.5) *within the same class*, and
  `HarmonicOscillatorPotential`'s `omega` (also `"frequency"`) needs 0.5,
  the same as `v0`, not `Omega`'s 0.0 -- confirming dimension alone can
  never safely determine role, even restricted to one dimension name.
  Every entry is individually verified against `galax`'s own potential
  formula (source inspection plus, for the ambiguous cases, direct
  numerical confirmation that scaling the parameter by
  `sqrt(mass_scale)` scales the potential by exactly `mass_scale`) before
  being added. `PhysicalType.__str__` joins every
  alias with `/` (e.g. `"speed/velocity"`), which `u.dimension()` silently
  treats as dimensionless rather than raising; dimension derivation takes
  the first name from iterating the `PhysicalType` instead. Given
  explicitly, `parameterization` names a registered non-native conversion.
  NFW registers `concentration_m200`, implemented and verified against
  `galax`'s own NFW enclosed-mass function. It converts a concentration `c`
  and $M_{200c}$ (mass enclosed within the radius where mean density is
  200x the critical density) into native `(m, r_s)` via
  `rho_crit = 3*H**2 / (8*pi*G)`, `r200 = (3*M200 / (4*pi*200*rho_crit))**(1/3)`,
  `r_s = r200 / c`, `m = M200 / (ln(1+c) - c/(1+c))`. A registered
  `ForwardConverter` receives the component's raw parameters, the resolved
  configuration's `cosmological_parameters`, and the component's `mge` (or
  `None`); an `InverseConverter` additionally
  receives the raw parameters' declared units so reported values can be
  restored to the configured representation. Parameterizations like this one
  can therefore use `H` without depending on `units.internal`.
  `cosmological_parameters` is
  threaded from `Configuration` through `ModelIterator` (a stored field, set
  in `from_configuration`) into `Potential.build_with_validity`.
  The internal unit system follows a separate path and is retained for `Potential.to_galax()`. Since
  configuration preparation
  preserves declared quantities as `{value, unit}` rather than stripping
  them (see the units-handling entries above), `ModelIterator.from_configuration`
  converts `cosmological_parameters` into `Quantity`s once via
  `tnt.units.resolve_cosmological_parameters` -- in `tnt.units`, not
  `tnt.potential`, since it's generic declared-quantity conversion with no
  potential-specific knowledge, matching the other declared-quantity helpers'
  home in `tnt.units` (`declared_quantity`, `validate_dimension`). Whole-config
  quantity validation lives in `tnt.configuration.validation`, which already
  consumes the potential registry's authoritative parameter dimensions.
  `tnt.units` therefore remains a low-level unit primitive with no imports from
  runtime-family packages. Configuration validation imports runtime-family
  registries to obtain their authoritative schemas, so importing the
  configuration package can load JAX, Equinox, and galax.
  `_nfw_concentration_m200`/its inverse use `Quantity` arithmetic with local
  `Msun`, `kpc`, and `Myr` units for critical density and radius calculations.
  This keeps float32 reverse-mode intermediates representable even for `H`
  declared in `1 / s`; declared inputs remain unchanged. The native mass
  retains its input mass unit and the forward scale radius is in `kpc`.
  `_nfw_g` uses a Taylor series through c**10 below c=0.01, avoiding small-c
  cancellation in both conversions. The characteristic-mass quotient has a
  custom JAX derivative that avoids g(c)**2 in the denominator. Unrepresentable
  native values or mass/concentration derivative coefficients invalidate the
  conversion (a false validity flag); derivative representability
  is checked in `Msun` and `kpc` so equivalent declared mass units agree.
  A JAX optimization barrier preserves the local H conversion during JIT
  compilation, preventing arithmetic reassociation from recreating underflow.
  Independent decimal
  references test small-c values and gradients, including both sides of the
  series switch in x32/x64. The forward converter cube-roots the volume's
  numeric value
  and attaches the cube-root unit: directly raising a volume `Quantity` to
  `1/3` fails under a batched JAX trace. The forward conversion leaves the
  native quantities in the units produced by that arithmetic;
  `Potential.to_galax()` later supplies the shared unit system to `galax`.
  The inverse returns dimensioned raw parameters in their configured units.
  Bare-number stripping also occurs where a library function isn't
  `Quantity`-aware (`_nfw_g`'s `jnp.log1p`) or where `_solve_nfw_concentration`'s
  bisection needs a plain number to compare against. The concentration solver
  supplies a custom JAX derivative from its implicit root equation, because
  differentiating the bisection comparisons would yield zero gradients. A
  registered non-native parameterization carries its own `raw_dimensions` in its
  `ParameterizationSpec` (see `register_parameterization`); each registered TNT
  composite type declares its own `_raw_dimensions`, while curated native
  `galax` types use `_SUPPORTED_GALAX_TYPES`. A parameterization is deliberately
  scoped to one component's own raw parameters and, where needed,
  `cosmological_parameters` or its own `mge` -- it cannot depend on another
  component's resolved state. TNT does not support an NFW
  `(c, f) -> (m, r_s)` "concentration + mass fraction" parameterization
  (`f = M_200 / M*_TOT`, `M*_TOT` derived from the stellar MGE component)
  was removed for exactly this reason: `Potential.resolve` resolves each
  component independently and `Potential.build_with_validity` converts it
  using only its own inputs, so no component-local converter can see
  another component's resolved mass. That kind of cross-component
  relationship is now `tnt.priors`: consumed by the parameter generator
  (`PriorSampler`) rather than potential construction, never by
  `parameterization`. TNT ships no built-in priors, including a
  mass-fraction one -- only the mechanism (`tnt.priors.Prior`, the
  `sample`/`factor` plugin contract) and a documented worked example (see
  `docs/source/priors.md`). A plugin is a plain
  Python function loaded from its own `.py` file (file-path-only, resolved
  relative to `io_settings.input_directory`, not an installed package) with a
  fixed signature: `def fn(context: tnt.priors.PriorContext) -> None`,
  callable with one positional argument (a trailing default parameter or
  `*args` is tolerated but never populated; `load_prior_plugin` rejects any
  other arity).
  `PriorContext` is an `eqx.Module` -- the single, extensible object holding
  the run state a plugin may read (`context.candidate`, `context.mges`,
  `context.unit_system` today); adding a field there stays backward-compatible
  because a plugin only ever names that one argument. It is built once per
  draw inside the numpyro model, after every `sample` site, and consumed
  immediately -- never a traced/`jit` argument, so its size costs nothing.
  `context.build_potential()` assembles this draw's `tnt.potential.Potential`
  from `candidate` for a factor over a derived quantity (an enclosed mass, a
  circular velocity): it calls `Potential.build_with_validity`, itself
  JAX-traceable. The first call in a draw registers that draw's validity as
  a `numpyro.factor` site giving an invalid geometry exactly `-inf`
  log-probability (the formal statement that point has no support, not a
  `nan` relying on gradient propagation to get rejected) and caches the
  result; a later call in the same draw -- the same plugin or another --
  reuses it rather than rebuilding and registering a second site. `Prior`
  captures the run's `resolved` potential + cosmology + unit system to make
  this work; a `Prior` built without them (a unit test exercising only
  sample sites) leaves `build_potential()` raising. A plugin may only call
  `numpyro.factor` -- never `sample`/`deterministic` -- so it can add a soft
  preference over already-established values but can never independently
  assign or overwrite a parameter, ruling out any collision with that
  parameter's own ordinary `prior` by construction, not validation. `Prior.sample` auto-selects `numpyro.infer.Predictive` (no
  factor sites) or `numpyro.infer.MCMC`/`NUTS` (any factor sites present) --
  a hard `Uniform.log_prob` factor does not work well with NUTS (flat
  interior gradient, discontinuous boundary; verified empirically, not just
  reasoned about) -- use a smooth distribution (`Normal`, `TruncatedNormal`,
  ...) for factor terms instead. Genuine posterior sampling (conditioning on
  a `Model`'s real chi2) needs a further bridge -- turning chi2 into a
  `numpyro.factor` -- that doesn't exist yet; deliberately out of scope,
  real future work reusing the same composed-model machinery.
- Every registered parameterization converts both ways: a
  `register_parameterization` call takes `convert` *and* `invert` (bundled in
  its `ParameterizationSpec`), so one direction can never be registered without
  the other. `AbstractPotentialComponent.raw_parameters`/
  `tnt.potential.raw_potential_parameters` use `invert` to report a
  `Potential`'s components back in their configuration's own
  parameterization (`Model.raw_parameters`, read by
  `AllModels._model_row` for its table columns) -- necessary because
  `Potential.rescale` only knows how to scale native `galax` parameters, so
  the raw values must be recomputed from the rescaled native ones, not
  carried through unchanged. `concentration_m200`'s inverse has no closed
  form: `rescale` holds `r_s` fixed and scales only `m`, which is not the
  same as holding `c` fixed and scaling `M_200`, so recovering `c` means
  solving `c**3 / (ln(1+c) - c/(1+c)) = target` for `c` --
  `tnt.potential._solve_nfw_concentration` does this via fixed-iteration
  bisection, relying on that function being verified (numerically) strictly
  monotonically increasing in `c`. Its gradients use the derivative of the
  solved equation, including when the inverse is batched. The fixed
  `[1e-6, 1e6]` concentration bracket limits this derivative to roots inside
  that range; outside it the solver clamps to an endpoint. The values are
  verified by round-trip self-consistency
  (`forward(inverse(native)) == native`, including after a rescale); gradients
  are checked against finite differences.
- Every component declared under `potential` is active. Excluding a component
  means removing or commenting out its complete configuration entry. Each
  declared component must contain a nonempty `parameters` mapping.
- Explicit kinematics histogram metadata is grouped under `histogram` as
  `width`, `center`, and `bins`.
- Defaults for dynamically named potential parameters are declared under
  `dynamic_object_defaults.parameter`. The merge layer applies them to every
  parameter unless the user overrides the property. The generic resolver can
  also apply component defaults supplied by schema metadata, but the packaged
  profile currently declares none; each potential component therefore states
  its own component-level fields.
- Policies that depend on a kinematics data-set type are declared under
  `kinematics_type_defaults`. The configuration resolver must select the
  matching type policy and then allow settings on the named data set to
  override it.
- The Gaussian-Hermite histogram defaults use a three-sigma velocity extent,
  an approximate bin width of one tenth of the minimum observed dispersion,
  and a zero-centered histogram.
- The Bayesian LOSVD histogram defaults use the symmetric observed velocity
  width without additional scaling or oversampling, center the histogram on
  zero, and derive the systemic velocity from the flux-weighted centroid.
- Proper-motion validation warns when velocity-bin width exceeds 0.25 times
  the global dispersion or histogram width is less than five times the global
  dispersion.
- Values describing the background cosmology belong under
  `cosmological_parameters`; they are not attributes of the modelled system.
- The Hubble parameter used for the modelled halo's epoch is named `H` under
  `cosmological_parameters`; it is not restricted to the present-day value
  `H0`.
- `mge_settings.intrinsic_mass_quad_order` and
  `mge_settings.projected_mass_quad_order` are positive fixed Gauss-Legendre
  quadrature orders for intrinsic spherical-grid and projected pixel
  integration, respectively. The packaged defaults are both 10.
- `SphericalGrid` is defined in `tnt.spatial_binnings`. Runtime coordinate
  conversion uses the angular-to-physical direction.
- Process-wide JAX precision, shared comparison tolerances, and
  constraint-error floors belong under `numerics_settings`.
  `jax_enable_x64` defaults to `true`. Importing `tnt` establishes that
  default before other TNT modules create JAX-backed values; a successfully
  validated configuration applies its resolved value before runtime-object
  construction. The first resolved configuration fixes the policy for the
  process. Further configuration reads and `ModelIterator.run()` calls are
  valid with the same value, while a conflicting configuration requires a new
  Python process. Existing arrays are not converted when the policy changes,
  so callers must prepare configuration before constructing TNT runtime
  objects. The entire `numerics_settings` mapping is resume-critical. Model
  comparison uses a relative tolerance of `1e-10`, while parameter-grid
  comparisons use `1e-6`. Total-mass and intrinsic-mass constraint errors have
  floors of `1e-8` and `1e-16`, respectively.
- Orbit-library radial limits are galaxy-specific and therefore have no
  package-wide defaults; the user configuration must provide them.
- A negative `orbit_library_settings.random_seed` requests a generated seed.
  Zero or a positive integer is an explicit seed for a reproducible run.
- Mutually exclusive chi-squared threshold representations use tagged
  `{mode, value}` objects rather than competing keys. The generator's
  `delta_chi2_threshold` accepts `absolute` or
  `fraction_of_sqrt_2n_observations`; the stopping criterion's
  `minimum_delta_chi2` accepts `absolute` or `relative`. This schema makes it
  impossible to specify both representations simultaneously. Search
  improvement is the cumulative previous best chi2 minus the cumulative new
  best. Absolute mode compares that difference directly; relative mode
  divides it by the previous best. `minimum_delta_chi2.enabled: false`
  disables chi2-improvement stopping, leaving the model/iteration limits or
  the parameter generator to stop the search. Mode and value remain present,
  validated, and nonnegative while disabled. The generator's separate
  `delta_chi2_threshold` also remains nonnegative. Independently of that
  setting, a fresh run records and then stops after a first iteration with no
  successful model, and a resumed all-failed `AllModels` stops before another
  proposal because neither has a valid chi2 base. Once a successful model
  exists, later failed-only iterations retain the previous best, skip the
  delta-chi2 check, and allow the generator to continue subject to its normal
  limits.
- Concrete parameter generators explicitly register their own `_type` through
  `register_parameter_generator`; `build_parameter_generator` dispatches from
  that owned registry. Undecorated subclasses are not selectable through
  configuration, and duplicate or inherited `_type` declarations fail at
  registration. Configuration validation obtains each generator type's
  required settings from the registered class, so dispatch and validation
  share the same authoritative declarations.
- `parameter_space_settings.stopping_criteria.target_model_count` is a soft
  cumulative target, not a strict maximum. TNT starts a new iteration only
  while the existing model count is below it, then completes every proposed
  model and potential rescaling in that iteration. The final count may exceed
  the target; other stopping conditions may end the search below it.
- `parameter_space_settings.stopping_criteria.n_new_iter` is the maximum number
  of additional iterations for the current `ModelIterator.run()` call,
  not a cumulative limit across resumed runs. Model and `RunConfigLog`
  iteration numbers remain cumulative; a resumed call measures its new
  allowance from the persisted `AllModels.n_iterations()` starting point.
- `parameter_space_settings.potential_rescalings` controls optional scaling of
  the complete assembled potential. It contains `enabled`, `range_count`, a
  positive inclusive `mass_scale_range`, `spacing` (`linear` or
  `logarithmic`), and `include_unscaled`. Scaling is independent of the
  ordinary stellar `ml` parameter. Each scale is a separate model-table entry
  with `potential_mass_scale_factor`; `include_unscaled` adds factor `1.0`
  exactly once when needed. Disabled rescaling retains and validates its
  settings but execution produces only the unscaled model.
- Gauss-Hermite `maximum_gh_order` and observational-error policies belong to
  each dynamically named kinematics data set, not to global weight-solver
  settings. Type defaults use order 4 with neutral named systematic
  uncertainties for `v`, `sigma`, `h3`, and `h4`. An explicit systematic map
  replaces the default map and must cover every quantity through the selected
  order.
- `proper_motions.observational_errors.variance_scale` is also per data set.
  It multiplies proper-motion error variances, so uncertainties are scaled by
  its square root; it must be positive and `1.0` is neutral.
- `execution_settings.model_processing_order` accepts `model_by_model` or
  `stage_by_stage`. `model_by_model` completes orbit integration and weight
  solving for each model in turn and is the only implemented order; runtime
  construction and `ModelIterator.run()` raise `NotImplementedError` for
  `stage_by_stage` before model-search work begins.
- `execution_settings.orbit_workers` and `weight_workers` are validated and
  retained but currently have no execution effect because no scheduler
  consumes them yet. TNT calculates `chi2`, `kinchi2`, and `kinmapchi2` as
  part of its normal model evaluation.
- `weight_solver_settings.reattempt_failures` remains in the schema for future
  retry behavior, but configuration currently requires it to be `false`.
  `true` is rejected until retry semantics and execution are implemented;
  `ModelIterator._solve()` makes exactly one attempt.
- Analysis defaults belong under `analysis_settings`. Orbit decomposition uses
  explicit circularity thresholds for cold, warm, and counter-rotating orbit
  classes, with the hot interval implied between `-0.25` and `0.25`. The
  default component nomenclature is `bulge_disk`, decomposition caching is
  enabled, component-weight output is disabled, and Gaussian fitting is used
  to derive mean velocity and dispersion from LOSVD histograms.
- The fully resolved configuration must be preserved with model output for
  reproducibility; preserving only the user delta is insufficient.

## Logging

- Importing TNT and reading a configuration must not configure logging or
  alter the root logger. Modules emit records through `logging.getLogger(__name__)`.
- `Configuration.read()` is the low-level preparation API: it resolves,
  validates, and preserves the same configuration artifacts as
  `configuration_session()`, but it does not install TNT logging handlers.
  Use it when the embedding application owns logging or no TNT preparation
  logfile is required.
- `configuration_session()` is the recommended standalone lifecycle. It wraps
  the same preparation logic and the caller's model-execution block in one TNT
  logging session, records exceptions, and removes only TNT-created handlers
  when the `with` block ends.
- Standalone execution explicitly calls `tnt.configure_logging()` with the
  resolved configuration, or preferably uses
  `tnt.configuration_session(filename)` to include configuration preparation
  in the logfile. The session loads the YAML/defaults only once, bootstraps the
  output and logging settings, and continues full resolution with the same
  mapping. TNT configures only the `tnt` package logger, writes `DEBUG` and
  higher records to a timestamped file below the output directory, and sends
  `INFO` and higher records to the terminal.
- A logging session owns and removes only TNT-created handlers, is idempotent,
  restores the prior `tnt` logger state when closed, and never shuts down or
  reloads Python logging.
- Worker processes call `tnt.configure_worker_logging()` with the parent
  session's queue. Only the parent listener writes to the logfile and terminal,
  avoiding concurrent writes from multiple processes. The session creates its
  queue through an explicit `spawn` multiprocessing context and exposes that
  same context as `LoggingSession.worker_context`; all worker processes and
  future process pools must use it. TNT must not call
  `multiprocessing.set_start_method()`, because the embedding application owns
  that process-global policy. Spawned worker targets must be importable, and
  executable entry points must use an `if __name__ == "__main__"` guard.
- Because `spawn` gives each worker a fresh interpreter with a cold JAX JIT
  cache, a future worker pool should also set `jax_compilation_cache_dir` (via
  `jax.config`) so workers load compiled XLA executables from disk instead of
  each recompiling every jitted function. That is a separate concern from the
  logging context and belongs with the production-execution work, not here.
- `tests/integration_tests/test_configuration_session.py` exercises the
  bootstrap lifecycle with `tnt.configuration_session()` against a complete
  example profile (`configuration.yaml`, alongside it), covering every
  top-level configuration section at once, unlike the synthetic per-feature
  configurations in `tests/unit_tests/test_configuration.py`.

## Human Workflow

Thomas and Prash review each other's pull requests before merging to `main`:

- A PR author requests review from the other.
- A reviewer whose feedback is limited to tests or documentation makes those
  changes directly and completes the merge.
- A reviewer whose feedback touches code records it in a PR-specific audit
  doc (`aidocs/pr-<N>-<topic>-audit.md`) and pings the author (`@<username>`
  in the PR) to respond. They iterate until the PR is ready to merge. The
  audit doc is removed from the branch once its findings are addressed,
  before merging.
- Follow-up work identified during review but out of scope for the current
  PR is filed as a new GitHub issue rather than folded into the PR.
- Claim an issue by assigning yourself to it, either up front or as soon as
  work on it starts. An unassigned issue is open to either of them.
- GitHub's merge strategy (squash vs. a real merge commit) is chosen per PR
  at merge time, not fixed for the repo -- don't assume a branch's
  individual commits will, or won't, survive into `main`'s history without
  checking.
- Always prefer merging `main` into a PR branch rather than rebasing on
  `main` -- rebasing can silently break the other person's copy of a
  shared branch.

Above all: communicate whenever something is unclear.
