# `read_kinematics` x `model-architecture-scaffold` compatibility

Fresh analysis (diffed the branches directly, not a recap of a prior discussion) as of 2026-08-03.

## Bottom line

The Python implementation merges without textual conflicts, but the branches
overlap in `aidocs/KNOWLEDGE.md`, `docs/source/configuration.md`, and
`docs/source/units.md`; those documentation conflicts need a deliberate
semantic resolution. There is also a real architectural collision: both
independently built a kinematics abstraction, under different names/shapes.
It must be reconciled deliberately even though Git does not flag the duplicate
Python APIs.

**Decision: `tnt/kinematics.py` (read_kinematics) is canonical.** It's the
real, tested implementation; `tnt/kinematic_data.py`'s scaffold should be
retired and callers repointed at it. Do this reconciliation *while* landing
`read_kinematics` into `model-architecture-scaffold`, before that combined
branch reaches `main`.

## The collision

| | `model-architecture-scaffold` | `read_kinematics` |
|---|---|---|
| File | `tnt/kinematic_data.py` (105 lines) | `tnt/kinematics.py` (924 lines) |
| Base class | `AbstractKinematicData` -- every method `raise NotImplementedError` | `AbstractKinematics` -- fully implemented + tested |
| Subclasses | `GaussHermiteKinematicData`, `BayesLOSVDKinematicData`, `ProperMotionsKinematicData` (empty stubs, just `_type` set) | `GaussHermite`, `BayesLOSVD`, `ProperMotions` (real histogram/GH-coefficient/proper-motion parsing) |
| Factory | `build_kinematic_data(kinematic_data, binnings, mges, unit_system)` -- stub, no file I/O | `build_kinematics(kinematic_data, input_directory, unit_system, spatial_binnings, mges=None)` -- reads data files itself, same convention as `build_mges` |
| Construction | abstract `from_settings(cls, settings, binnings, mges, unit_system)` | dispatches via `_KINEMATICS_CLASSES[kind]`, calls `cls.from_config(name=..., settings=..., data_file=..., binning=..., mge=..., unit_system=...)` |
| Fields on base | `binning: ProjectedBinning`, `mge: LightMGE \| MassMGE \| None` | `name`, `data_file`, `binning: Any`, `mge: AbstractMGE \| None`, `histogram: Histogram \| Histogram2D`, `bin_ids` |
| Extra method | abstract `design_matrix(orbit_library: OrbitLibrary) -> jnp.ndarray` (consumed by `weight_solver.py`) | none yet |
| `_type` ClassVar | Every subclass sets `_type: ClassVar[str]` (e.g. `_type = "gauss_hermite"`) -- same self-describing pattern as `AbstractPotentialComponent`/`AbstractWeightSolver`/`AbstractMGE`'s `_intensity_attr` elsewhere in the codebase | No `_type` anywhere; the type->class mapping instead lives in a module-level `_KINEMATICS_CLASSES` dict (kinematics.py:381-385), decoupled from the classes themselves |

Good news: the **config schema** -- the set of valid `type` strings and which
class each one names -- is already consistent everywhere: `type` &isin;
`{gauss_hermite, bayes_losvd, proper_motions}` matches in
`configuration_validation.py` (shared/in `main`), in the scaffold's `_type`
ClassVars, and in `read_kinematics`'s `_KINEMATICS_CLASSES` dict keys. This is
a Python-API naming/shape mismatch only, not a data-format mismatch.

That's a different axis from the `_type` ClassVar row above, though: the
*strings* agree, but *how the mapping is expressed in code* doesn't --
scaffold puts it on each class as a self-describing attribute, `kinematics.py`
puts it in an external dict instead (point 6 below is about closing that
code-convention gap, not a schema fix).

Also worth knowing: everything kinematics-related in
`model-architecture-scaffold` (`kinematic_data.py`, `weight_solver.py`,
`all_models.py`) is explicitly "signature-only scaffold" -- nothing is called
yet, so nothing is functionally broken today. This is a reconciliation of
intended shape, not a runtime bug.

## `configuration_validation.py`: no line-level overlap, but check the split

`read_kinematics` moved several validators out of `configuration_validation.py`
into `kinematics.py` itself (`_validate_kinematics_observational_settings`,
`_validate_gauss_hermite_errors`, `_validate_proper_motion_errors`,
`_validate_histogram`, `_validate_proper_motion_warning_thresholds` -- all
deleted from the central file, replaced by a comment noting type-specific
validation now happens at `build_kinematics` time).

`model-architecture-scaffold`'s diff to the same file only touches
`_validate_potential` (~line 294), `_validate_orbit_library_settings` and its
new `_validate_orbit_sampler`/`_validate_dithering` helpers (~line 596+), and
removes `_validate_counter_rotating_cut` (~line 656) -- none of which overlap
with `read_kinematics`'s changes. Git will merge this file cleanly.

## Action list (before this reaches `main`)

1. Delete `tnt/kinematic_data.py`; keep `tnt/kinematics.py` as-is.
2. Repoint `weight_solver.py` and `all_models.py`'s
   `from tnt.kinematic_data import AbstractKinematicData` at
   `tnt.kinematics.AbstractKinematics` (and update the `Sequence[...]`/
   `Mapping[str, ...]` type hints accordingly).
3. Add `design_matrix(orbit_library) -> jnp.ndarray` to `AbstractKinematics`
   (or its concrete subclasses) -- `read_kinematics` doesn't have this yet
   since it predates the weight-solver scaffold's needs.
4. Standardize on `build_kinematics`'s signature/convention (takes
   `input_directory`, reads files itself -- matching `build_mges`'s existing
   precedent) rather than `build_kinematic_data`'s stub signature. Update
   whatever config-loading glue in `model-architecture-scaffold` currently
   expects `build_kinematic_data`'s call shape.
5. Decide whether `binning`/`mge` on `AbstractKinematics` should be typed
   strictly (`ProjectedBinning`, `LightMGE | MassMGE`) rather than `Any` /
   `AbstractMGE | None` -- scaffold's stricter typing is probably worth
   keeping if there's no reason for the looser types in `kinematics.py`.
6. Add `_type: ClassVar[str]` to `AbstractKinematics` and set it on
   `GaussHermite` ("gauss_hermite"), `BayesLOSVD` ("bayes_losvd"), and
   `ProperMotions` ("proper_motions"), matching the self-describing
   `_type` pattern used everywhere else (`AbstractPotentialComponent`,
   `AbstractWeightSolver`, the `kinematic_data.py` scaffold itself). Once
   present, `_KINEMATICS_CLASSES` could even be derived from subclasses'
   `_type` rather than hand-maintained, though that's optional polish.

## Resolution (2026-08-04)

The action list was implemented while merging `model-architecture-scaffold`
into `read_kinematics`, with two clarified decisions:

- Item 2 applies to `weight_solver.py` and `model_iterator.py`; the original
  reference to `all_models.py` was incorrect. The related `orbit_library.py`
  documentation was updated as well.
- Item 3 adds the explicit `AbstractKinematics.design_matrix()` contract, but
  concrete numerical projections remain intentionally unimplemented until the
  orbit-integration and weight-solving scaffolds are implemented.
- Item 5 uses strict `ProjectedBinning` and `LightMGE | MassMGE | None` fields,
  plus runtime checks when named references are resolved.
- Item 6 is implemented fully: concrete subclasses own `_type`, and the
  dispatch registry is derived from them with missing/duplicate checks.

`tnt/kinematic_data.py` was retired, architecture consumers now use
`tnt.kinematics.AbstractKinematics`, and runtime construction standardizes on
`build_kinematics` after the MGE and spatial-binning registries are built.
