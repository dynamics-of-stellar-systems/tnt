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
- `tnt/config_parsing.py` holds shared helpers for parsing/validating
  resolved configuration data (mapping/required-field/reject-unknown/
  string/number checks, named cross-reference resolution, bin-ID reading).
  If a helper like this ends up reimplemented in more than one module,
  move the shared logic there instead of leaving the copies to drift --
  see its module docstring for the full rationale and existing contents
  before adding a near-duplicate.

## Configuration defaults

- The packaged base profile is `tnt/defaults/default_config.yaml`.
- TNT will recursively merge the base profile with a user profile before
  constructing a model.
- Configuration preparation is implemented by `tnt.Configuration`. Its
  `read()` method loads the user YAML, recursively merges package defaults,
  resolves dynamic and kinematics-type defaults, validates the resulting
  data, and atomically preserves `user_config.yaml`, `resolved_config.yaml`,
  and `run_manifest.yaml` below `<output_directory>/config_repository/`. It
  does not instantiate scientific runtime objects.
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
  Legacy
  deprecation-and-ignore warnings are intentionally not reproduced.
- TNT uses `unxt` for configuration units. The required
  `units.internal` block defines length, time, mass, angle, and power;
  `units.display` selectively overrides presentation units and otherwise
  inherits from the internal system. `Configuration.unit_systems` exposes the
  two constructed systems.
- Every known unitful configuration value must state its unit explicitly;
  internal units and zero values are not implicit exceptions. Standalone
  quantities use `{value: ..., unit: ...}`. Unitful parameter definitions use
  a required sibling `unit` applying to the parameter value and generator
  range. Dimensionless values remain plain numbers and reject a `unit`.
  Configuration preparation converts supported quantities to plain
  internal-unit numbers before validation and resolved-YAML generation; the
  byte-identical user copy retains the submitted notation.
- The first unit-aware schema covers `cosmological_parameters.H0`,
  `system_attributes.distance`, explicit kinematics histogram width and center,
  Gauss-Hermite `v` and `sigma` systematic uncertainties, Plummer `m` and `a`,
  and light-MGE potential `ml`. Linear parameter steps are converted;
  logarithmic parameter values and bounds are shifted between reference units
  while log step sizes remain unchanged.
- MGE contents and quantities inside observational files are deliberately
  deferred to the object-construction/data-loading phase. Configuration
  preparation does not open those files. `tnt.kinematics.build_kinematics`
  constructs named `GaussHermite`, `BayesLOSVD`, and `ProperMotions` objects;
  it converts unitful observations into the internal unit system and retains
  JAX arrays in immutable Equinox modules. Population observations are loaded
  separately by `tnt.populations.build_populations()`.
- `tnt.mge.build_mges()` is the explicit runtime boundary that loads the
  resolved `MGEs` registry into named `LightMGE` and `MassMGE` objects.
  `Configuration` continues to contain no instantiated scientific objects.
- Intel macOS needs the compatibility constraints recorded in
  `pyproject.toml`: the latest available JAX wheel is in the 0.4 series, and
  `unxt` 1.1.1 requires matching older Quax, Quaxed, Plum, Astropy, and NumPy
  APIs on that platform.
- Retrieve units from `unxt` unit systems by physical dimension rather than
  generated attribute names. In particular, the luminosity dimension is
  exposed as `radiant_flux` by the Intel-macOS `unxt` stack and as `power` by
  newer versions; dimension-based lookup supports both without platform
  branching.
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
  `portable_data` and `as_portable_dict()` expose the archived form.
  Configuration preparation requires both path strings but creates only the
  output directory and configuration repository.
- `config_repository/user_config.yaml` is a byte-identical copy of the user
  file. `resolved_config.yaml` has all defaults applied with portable paths.
  `run_manifest.yaml` records materialized paths, configuration checksums,
  software versions, Git commit and dirty state when available,
  Python/platform/host context, scheduler identifiers, logfile location, and
  orbit random-seed state.
- Configuration preparation cannot record a generated seed or observational
  input checksums because neither exists yet. A negative seed is recorded as
  `pending_generation`; the execution phase must update the effective seed.
- The user profile must define the physical system, dynamically named
  potential components and parameters, input directory, and output directory.
- TNT user profiles generally use snake-case type identifiers and field names.
  The established `MGEs` registry name and projected-binning `PA` field are
  current schema exceptions. Parameter search bounds belong under
  `generator_settings` as `lower_bound`, `upper_bound`, `step`, and
  `minimum_step`; display labels use `latex_label`.
- Scientific inputs use independent named registries: `MGEs` maps MGE names to
  files; `spatial_binnings` maps names to inline rectangular aperture geometry
  (`min_x`, `min_y`, `x_extent`, `y_extent`, and `PA`) plus a `bins_file`
  containing a 2D NumPy pixel-to-bin map; `potential` defines potential
  components; `kinematic_data` references a binning and optionally an MGE; and
  `population_data` references a binning. Preparation validates all
  cross-references without opening the files.
- Population observations must always use a separate
  `population_data.<name>.data_file`. TNT does not support DYNAMITE's
  `with_pops` pattern or population columns embedded in a kinematics file,
  even when the population and kinematics data share a spatial binning.
- `tnt.populations.build_populations()` loads each configured population ECSV
  into an immutable JAX/Equinox `Populations` object. It resolves a strictly
  typed `ProjectedBinning` but no MGE. Files require a positive unique
  `vbin_id` and one or more `property`/`dproperty` column pairs. Paired units
  must be equivalent; declared quantities are converted to the internal unit
  system, unitless columns remain dimensionless, and uncertainties must be
  positive. Observed bin IDs may be a subset of, but cannot fall outside, the
  referenced binning.
- `tnt.spatial_binnings.build_spatial_binnings()` is the explicit runtime
  boundary that loads the resolved `spatial_binnings` registry into named
  `ProjectedBinning` objects. It validates the complete entry before file
  access, validates the loaded non-empty bin array, converts coordinates to
  the internal angle unit, and precomputes pixel quadrature.
- `AbstractMGE.get_projected_mass()` integrates projected MGE totals into the
  positive bin IDs of a `ProjectedBinning`; bin ID 0 is excluded. The MGE and
  binning coordinate units must be dimensionally consistent.
- `build_kinematics` requires already-built `ProjectedBinning` objects and
  optional `LightMGE`/`MassMGE` objects, resolves each data set's named
  references to those shared runtime objects, and returns a name-to-object
  mapping. Its runtime boundary rejects incorrectly typed registry values.
- `AbstractKinematics.binning` is strictly a `ProjectedBinning`; its optional
  `mge` is strictly a `LightMGE` or `MassMGE`. Each concrete kinematics class
  owns its configuration identifier in `_type`, and the builder's dispatch
  registry is derived from those subclasses with duplicate detection rather
  than maintained independently.
- `AbstractKinematics.design_matrix()` defines the weight-solver projection
  boundary introduced by the model-architecture scaffold. It deliberately
  raises `NotImplementedError` until orbit integration and the concrete
  kinematics projections are implemented; observational values and
  uncertainties already have a shared base-class interface.
- Gauss-Hermite ECSV files require `vbin_id`, unitful `v`, `dv`, `sigma`, and
  `dsigma` columns plus dimensionless `hN`/`dhN` pairs. Configured systematic
  uncertainties are added in quadrature. Missing higher-order pairs are
  represented by zero coefficients only when the corresponding configured
  systematic uncertainty is positive.
- Bayesian LOSVD ECSV files use DYNAMITE-compatible `binID_dynamite`,
  `bin_flux`, `losvd_N`, and `dlosvd_N` columns. Metadata must contain `vcent`,
  `dv`, and an explicit `velocity_unit`; TNT converts the velocity grid and
  applies the configured flux-weighted systemic centering.
- Proper-motion input retains DYNAMITE's NPZ array layout (`PM_2dhist`,
  `PM_2dhist_sigma`, `binID_dynamite`, `nstarbin`, `vxrange`, and `vyrange`)
  and adds a required scalar `velocity_unit`. Construction validates and
  normalizes each 2D distribution, scales uncertainties by the square root of
  `variance_scale`, and emits configured sampling warnings.
- Potential types are `triaxial_light_mge`, `triaxial_mass_mge`, `nfw`, and
  `plummer`. A light-MGE potential requires an `ml` parameter. A mass-MGE
  potential rejects `ml` because its MGE already contains mass.
- Explicit kinematics histogram metadata is grouped under `histogram` as
  `width`, `center`, and `bins`.
- Defaults for properties of dynamically named potential components and
  parameters are declared under `dynamic_object_defaults`. The merge layer
  applies them to each corresponding object unless the user overrides the
  property on that object.
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
- The present-day Hubble parameter is named `H0`, distinguishing it from the
  Hubble parameter at other cosmological times.
- `mge_settings.intrinsic_mass_quad_order` and
  `mge_settings.projected_mass_quad_order` are positive fixed Gauss-Legendre
  quadrature orders for intrinsic spherical-grid and projected pixel
  integration, respectively. The packaged defaults are both 10.
- `SphericalGrid` is defined in `tnt.spatial_binnings`; code importing it from
  `tnt.mge` must update its import. The unused inverse conversion
  `AbstractMGE.physical_to_angular()` and
  `quantity_conversions.physical_to_angular()` have been removed; runtime code
  retains the angular-to-physical direction used by current workflows.
- Shared comparison tolerances and constraint-error floors belong under
  `numerics_settings`. Model comparison uses a relative tolerance of `1e-10`,
  while parameter-grid comparisons use `1e-6`. Total-mass and intrinsic-mass
  constraint errors have floors of `1e-8` and `1e-16`, respectively.
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
- `parameter_space_settings.stopping_criteria.target_model_count` is a soft
  cumulative target, not a strict maximum. TNT starts a new iteration only
  while the existing model count is below it, then completes every proposed
  model and potential rescaling in that iteration. The final count may exceed
  the target; other stopping conditions may end the search below it.
- `parameter_space_settings.stopping_criteria.n_new_iter` is the maximum number
  of additional iterations for the current `ModelIterator.run()` invocation,
  not a cumulative limit across resumed runs. Model and `IterationConfigLog`
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
- The former weight-solver keys `number_GH`, `GH_sys_err`, and
  `PM_sys_err_factor` are not part of the TNT schema.
- `execution_settings.model_processing_order` accepts `model_by_model` or
  `stage_by_stage`. The former completes orbit integration and weight solving
  for each model in turn and is the only implemented order; runtime
  construction and `ModelIterator.run()` raise `NotImplementedError` for
  `stage_by_stage` before model-search work begins.
- `execution_settings.orbit_workers`, `weight_workers`, and
  `external_chi2_workers` are validated and retained but currently have no
  execution effect because no scheduler consumes them yet.
- The former `execution_settings.orbit_family_integration_in_parallel` option
  has been removed from TNT. Configuration rejects it as an unknown field, and
  `Potential.generate_orbit_library()` has no corresponding argument.
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
  avoiding concurrent writes from multiple processes.
- `tests/integration_tests/test_configuration_session.py` exercises the
  bootstrap lifecycle with `tnt.configuration_session()` against a complete
  example profile (`configuration.yaml`, alongside it), covering every
  top-level configuration section at once, unlike the synthetic per-feature
  configurations in `tests/unit_tests/test_configuration.py`.

## Relationship to DYNAMITE

- TNT is a standalone reimplementation and refactoring, not a compatibility
  layer for DYNAMITE.
- TNT's configuration schema may improve or replace DYNAMITE concepts and
  names; compatibility with DYNAMITE configuration files is not a requirement.
- Legacy Fortran functionality will be replaced by Python implementations
  based on JAX. Do not introduce Fortran-specific configuration or execution
  paths into TNT.
