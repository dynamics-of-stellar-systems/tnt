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
- `tnt/kinematics/` keeps shared base objects in `base.py`, one concrete data
  family per module, and registry/orchestration logic in `__init__.py`.
- `tnt/potential/` separates curated type metadata (`registry.py`), NFW
  parameterization mathematics (`nfw.py`), the component hierarchy
  (`components.py`), and whole-potential orchestration (`core.py`). Its
  `__init__.py` defines the intended package-level API; implementation-specific
  names remain in their owning submodules.

## Linux development container

- `Dockerfile` and `compose.yaml` provide the reproducible Linux `x86_64`
  development environment used from Intel macOS. The host checkout is mounted
  at `/workspace`; its macOS `.venv` is never used in the container because
  `UV_PROJECT_ENVIRONMENT` points to `/opt/tnt-venv` inside the image.
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
- Runtime kinematics construction converts configured histogram quantities and
  Gauss-Hermite velocity systematics. Potential parameters instead keep their
  own declared unit all the way through `AbstractParameterGenerator` and
  `Potential` construction -- `ModelIterator.from_configuration()` does not
  pre-convert them into a shared internal-unit copy; `galax`'s own
  potential classes already convert generically at evaluation time, so
  nothing needs them pre-normalized (see `tnt.potential`'s module
  docstring). `ModelIterator.from_configuration()` also converts
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
- Prep-time validation and construction-time conversion don't cover the same
  fields consistently, and this isn't one shared policy: `spatial_binnings`
  has no prep-time check at all (construction owns it exclusively);
  kinematics histogram and systematic-uncertainty fields are checked at both
  prep time and construction time, independently. Potential parameters now
  follow the same pattern as `spatial_binnings`: checked at prep time
  (`validate_configuration_quantities`) and converted -- or, for potential
  parameters specifically, just kept in their declared unit -- at
  construction time (`AbstractParameterGenerator`/`Potential.resolve`/
  `Potential.build`), not eagerly at prep time.
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
- MGE deprojection enforces TNT's intrinsic-axis convention
  `0 < q <= p <= 1` eagerly. `_check_axial_ratios()` converts JAX results to
  Python control flow (`bool(...)` and `.nonzero()`) and raises
  `MGEDeprojectionError`, so `deproject_triaxial()` and
  `deproject_axisymmetric()` are deliberately not `jax.jit`/`jax.vmap`
  traceable. This is acceptable while model evaluation itself remains eager:
  `ModelIterator._evaluate()` catches Python exceptions and returns a
  variable-length `list[Model]`, while orbit integration and weight solving
  are still scaffolding. Revisit deprojection validity and `_evaluate()`
  failure handling together when orbit integration is implemented and TNT
  chooses whether models are individually jitted or evaluated as a masked,
  vectorized batch. Do not design a separate JAX validity mechanism before
  that execution strategy is known.
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
  The established `MGEs` registry name and projected-binning `PA` field are
  current schema exceptions. A parameter's search-space declaration belongs
  under `prior` as `{distribution: "<numpyro.distributions class>", args:
  [...]}` (replaced `generator_settings`'s `lower_bound`/`upper_bound`/
  `step`/`minimum_step` -- `step`/`minimum_step` had no consumer and were
  dropped rather than carried forward, matching how `logarithmic` was
  removed for the same reason, see below); display labels use `latex_label`.
- Scientific inputs use independent named registries: `MGEs` maps MGE names to
  files; `spatial_binnings` maps names to inline rectangular aperture geometry
  (`min_x`, `min_y`, `x_extent`, `y_extent`, and `PA`) plus a `bins_file`
  containing a 2D NumPy pixel-to-bin map; `potential` defines potential
  components; `kinematic_data` references a binning and optionally an MGE; and
  `population_data` references a binning. Preparation validates all
  cross-references without opening the files.
- Population observations always use a separate
  `population_data.<name>.data_file`, even when the population and kinematics
  data share a spatial binning.
- `tnt.populations.build_populations()` loads each configured population ECSV
  into an immutable JAX/Equinox `Populations` object. It resolves a strictly
  typed `ProjectedBinning` but no MGE. Files require a `bin_id` column and one
  or more `property`/`dproperty` column pairs. Paired units must be equivalent;
  declared quantities are converted to the internal unit system, unitless
  columns remain dimensionless, and uncertainties must be positive. The
  shared observational bin-ID rule below applies.
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
  `PlummerPotential` -- 25 classes total), or one of two TNT-specific MGE
  composite types, `TriaxialLightMGEPotential`/`TriaxialMassMGEPotential`, provided
  directly by TNT since `galax` has no native class for a
  sum-of-triaxial-Gaussians potential. A light-MGE potential requires an
  `ml` parameter; a mass-MGE potential requires `mge_mass_scale` instead
  and validation rejects `ml` on it, since its MGE already contains mass.
  Deliberately curated rather than "any `AbstractPotential` subclass":
  `galax` also exports abstract/base classes (which passed the old
  `issubclass` check and only failed later, confusingly, at `to_galax()`),
  pre-packaged multi-component bundles with no free parameters of their own
  like `MilkyWayPotential`/`LM10Potential` (their `disk`/`bulge`/`halo`/
  `nucleus` fields are themselves sub-potentials, not `ParameterField`s --
  redundant with TNT's own multi-component `potential:` section anyway),
  wrapper/transform decorators needing a required nested potential object
  (e.g. `TranslatedPotential`, `FlattenedInThePotential` -- these do carry
  their own `ParameterField`s, but the required nested potential still
  isn't representable), and classes needing a required non-`Quantity`
  hyperparameter (`MultipolePotential`'s `l_max: int`, which the old
  dispatch silently mis-wrapped as a dimensionless `Quantity`) -- none
  representable by the scalar `parameters.<name>.value` schema. Checked
  directly against every one of galax's 45
  `AbstractPotential` subclasses: 28 have every field either a scalar
  `ParameterField` or a galax-provided default (safe under the current
  schema); the curated 25 drops `HenonHeilesPotential`/`NullPotential`
  (not astrophysically relevant to TNT) and `AbstractCompositePotential`
  (an empty-parameter base class) from that 28.
- `parameterization` is a separate, optional field controlling how config
  `parameters` map onto a component's canonical fields. Omitted, raw
  parameter names must match the resolved `type`'s own native `galax`
  constructor kwargs exactly; their physical dimensions are read directly
  from `_SUPPORTED_GALAX_TYPES` (each entry a `NativeParameter(dimension,
  exponent)`). Dynamic derivation from `galax`'s own
  `ParameterField(dimensions=...)` metadata isn't production code at all any
  more -- since curating dimension by hand costs nothing extra once every
  parameter is individually verified for its exponent anyway, that
  derivation now lives only as a test-local helper in
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
  `r_s = r200 / c`, `m = M200 / (ln(1+c) - c/(1+c))`. Converters receive the
  resolved configuration's `cosmological_parameters` as a third argument
  (`tnt.potential.registry.ParameterizationConverter`'s signature) so
  parameterizations like this one that need `H` can use it.
  `cosmological_parameters` is
  threaded from `Configuration` through `ModelIterator` (a stored field, set
  in `from_configuration`) into `build_potential`, mirroring how
  `unit_system` is already threaded. Since configuration preparation now
  preserves declared quantities as `{value, unit}` rather than stripping
  them (see the units-handling entries above), `ModelIterator.from_configuration`
  converts `cosmological_parameters` into `Quantity`s once via
  `tnt.units.resolve_cosmological_parameters` -- in `tnt.units`, not
  `tnt.potential`, since it's generic declared-quantity conversion with no
  potential-specific knowledge, matching `normalize_unitful_value`'s existing
  home rather than the opposite direction (`tnt.units` importing
  `raw_parameter_dimensions` from
  `tnt.potential`, which *does* need `tnt.potential`'s own domain
  knowledge -- galax `ParameterField` metadata, the parameterization
  registry -- and couldn't move the other way).
  `_nfw_concentration_m200`/its inverse do their entire calculation in
  `Quantity` arithmetic rather than eagerly stripping every input to a bare
  float in one specific unit -- `unxt` composes/converts units automatically
  through the whole chain (verified: mixing `H` in `km / (s Mpc)` with
  `_newtonian_gravitational_constant()` in `m3 / (kg s2)` and `M_200` in `Msun`
  still gives the correct `r_s`/`m`
  once converted to `unit_system`'s units at the very end), so `H` works in
  whatever unit it's declared in, not just the internal unit system's.
  Bare-number stripping only remains where a library function isn't
  `Quantity`-aware (`_nfw_g`'s `jnp.log`) or where `_solve_nfw_concentration`'s
  bisection needs a plain number to compare against. Hand-maintained
  dimension tables now
  cover only non-native parameterizations and the two MGE composite types'
  own parameters
  (`tnt.potential.registry.PARAMETERIZATION_RAW_DIMENSIONS` and
  `tnt.potential.registry._MGE_RAW_DIMENSIONS`),
  not native-galax types. A parameterization is deliberately scoped to one
  component's own raw parameters (plus `unit_system`/`cosmological_parameters`)
  -- it can't depend on another component's resolved state. NFW's
  `(c, f) -> (m, r_s)` "concentration + mass fraction" parameterization
  (`f = M_200 / M*_TOT`, `M*_TOT` derived from the stellar MGE component)
  was removed for exactly this reason: `Potential.from_settings` resolves
  each component independently in one pass, so no component-local converter
  can see another component's resolved mass. That kind of cross-component
  relationship is now `tnt.priors`: consumed by the parameter generator
  (`PriorSampler`) rather than potential construction, never by
  `parameterization`. TNT ships no built-in priors, including a
  mass-fraction one -- only the mechanism (`tnt.priors.Prior`, the
  `sample`/`factor` plugin contract) and a documented worked example (see
  `docs/source/model_search.md`'s "Priors" section). A plugin is a plain
  Python function loaded from its own `.py` file (file-path-only, resolved
  relative to `io_settings.input_directory`, not an installed package) that
  may only call `numpyro.factor` -- never `sample`/`deterministic` -- so it
  can add a soft preference over already-established values but can never
  independently assign or overwrite a parameter, ruling out any collision
  with that parameter's own ordinary `prior` by construction, not
  validation. `Prior.sample` auto-selects `numpyro.infer.Predictive` (no
  factor sites) or `numpyro.infer.MCMC`/`NUTS` (any factor sites present) --
  a hard `Uniform.log_prob` factor does not work well with NUTS (flat
  interior gradient, discontinuous boundary; verified empirically, not just
  reasoned about) -- use a smooth distribution (`Normal`, `TruncatedNormal`,
  ...) for factor terms instead. Genuine posterior sampling (conditioning on
  a `Model`'s real chi2) needs a further bridge -- turning chi2 into a
  `numpyro.factor` -- that doesn't exist yet; deliberately out of scope,
  real future work reusing the same composed-model machinery.
- Every registered parameterization converts both ways:
  `tnt.potential.components._PARAMETERIZATIONS`
  maps to a `tnt.potential.Parameterization(convert, invert)` pair, not a
  bare converter, so one direction can never be registered without the
  other. `AbstractPotentialComponent.raw_parameters`/
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
  monotonically increasing in `c`. Verified by round-trip self-consistency
  (`forward(inverse(native)) == native`, including after a rescale) rather
  than against any independently derivable expected value, since none
  exists.
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
  avoiding concurrent writes from multiple processes.
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
