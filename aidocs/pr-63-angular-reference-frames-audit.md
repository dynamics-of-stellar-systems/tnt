# PR #63 audit: Split MGE/binning position angles into independent config fields

Date: 2026-09-04

Pull request: #63, `angular-reference-frames` -> `main`

Audited head: `6df857c08a6efa28c3b326baa303dfbd7df43cea`

Related issue: #62

## Overall judgment

**Needs focused corrections before merge.**

The architectural direction is right and the projection formula is consistent
with the declared astronomical and grid conventions. I found no critical or
high-severity defect, and the change does not need architectural revision.
Three medium findings remain, however: the position-angle domains agreed in
issue #62 are not enforced or documented, the public MGE loading boundary does
not validate its new angle, and the instructions for converting an
opposite-handed data set are ambiguous and incomplete. These are small,
localized corrections, but the first two define the validity of the new
runtime state and the third can lead a user to prepare scientifically incorrect
data.

After those corrections and the accompanying focused tests, this should be fit
to merge. PR #63 should land before PR #60; PR #60 should then be rebased and
its twist-zero-point transformation updated to preserve the absolute on-sky
component orientations introduced here.

## Architectural summary

PR #63 replaces one relative angle with two independently measured absolute
sky orientations:

- `MGEs.<name>.major_axis_pa` is the standard astronomical position angle of
  the MGE reference major axis, measured from north through east.
- `spatial_binnings.<name>.y_axis_pa` is the standard astronomical position
  angle of the binning grid's positive y-axis.
- TNT fixes the grid parity: positive x is 90 degrees east of positive y.
- Each Gaussian component has sky PA
  `major_axis_pa + PA_twist`.
- `AbstractMGE.get_projected_mass()` converts that sky PA into the mathematical
  angle used by its rectangular Gaussian integral:

  ```text
  alpha = y_axis_pa + pi/2 - major_axis_pa - PA_twist
  ```

That sign is correct for a grid whose +x direction is 90 degrees east of +y.
Writing a sky direction at PA `p` in that grid gives the mathematical angle
`pi/2 - (p - y_axis_pa)`, which is exactly the implementation above.

The change fits TNT's runtime-object boundary cleanly. Configuration
preparation validates and preserves the declared MGE angle. `build_mges()`
constructs the `Quantity`, loads the ECSV, and stores the value as a dynamic
leaf on the immutable Equinox MGE. `ProjectedBinning.from_settings()` performs
the corresponding runtime construction for `y_axis_pa`. Both angles retain
their declared units and are converted to radians only at the point of use.
The new fields remain compatible with JAX PyTrees and the existing jitted
projected-mass calculation.

The new value is correctly preserved through `AbstractMGE.rescaled()`,
`AbstractMGE.angular_to_physical()`, and `LightMGE.to_mass()`. It does not enter
MGE deprojection, which is appropriate: absolute orientation on the sky is
irrelevant to intrinsic axial ratios until model/orbit results are projected
back into a specific observed frame.

## Findings

### Critical

None.

### High

None.

### Medium

#### M1. The agreed position-angle domains are neither documented nor enforced

Issue #62 records the accepted domains as:

- `major_axis_pa` in `[0, 180)` because a photometric major axis is an
  undirected line; and
- `y_axis_pa` in `[0, 360)` because the positive grid y-axis is directed.

The implementation currently checks only that `major_axis_pa` is a finite
angular declaration. `y_axis_pa` is handled by the generic quantity validator,
which likewise checks dimension, numeric type, and finiteness but not its
domain. Values such as 720 degrees are accepted.

Although the projection formula is periodic, accepting non-canonical values
has practical consequences: physically identical declarations such as 0 and
360 degrees compare as different preserved configurations, and very large
angles unnecessarily lose trigonometric precision. More importantly, the
merged contract would not match the explicit decision in issue #62.

Action:

- Document both half-open domains.
- Reject values outside those domains at their owning validation boundaries.
  Rejection is preferable to silent normalization because TNT preserves the
  user's configuration declarations and uses them for resume compatibility.
- Add boundary tests for 0, the excluded upper endpoint, negative values, and
  unit-equivalent declarations in radians.

References:

- `tnt/configuration/validation.py:253-267`
- `tnt/spatial_binnings.py:130-138`
- `tnt/spatial_binnings.py:235-275`
- `docs/source/data_preparation.md:10-24`
- `docs/source/configuration.md:175-187`

#### M2. The public MGE readers accept an invalid `major_axis_pa`

`AbstractMGE.from_qtable()`, `AbstractMGE.read()`, and `read_mge()` now accept
the new runtime angle directly, but none validates that it is a finite scalar
angle. For example, `LightMGE.read(path, Quantity(3, "km"))` succeeds and
stores a length as `major_axis_pa`; projection fails later when it tries to
strip radians. An array value can also introduce unintended broadcasting in
`get_projected_mass()`.

The regular configuration path is protected by preparation-time declaration
validation, but these are public constructors and `from_qtable()` explicitly
describes itself as validating the MGE. The invariant should be established
when the runtime object is created, so every construction path is safe and
later numerical code does not fail with an opaque unit or shape error.

Action:

- In `from_qtable()`, validate that `major_axis_pa` is a scalar, finite angle;
  keep its declared unit.
- Apply the canonical `[0, 180)` check from M1 at the same boundary, or route
  both configuration and runtime construction through one small domain helper.
- Add tests for a non-angle unit, NaN/infinity, a non-scalar value, and the
  domain endpoints.
- Update the `Raises` documentation accordingly.

References:

- `tnt/mge.py:89-125`
- `tnt/mge.py:127-141`
- `tnt/mge.py:595-619`
- `tnt/mge.py:239-249`

#### M3. The opposite-handed-data conversion is ambiguous and incomplete

The new data-preparation page says an opposite-parity `bins_file` should be
"flipped along its x-axis (with `min_x` adjusted)." In mathematical language,
reflection *across* the x-axis reverses y, while changing an x direction from
west to east requires reversing the x coordinate (a reflection across the
y-axis). It is therefore unclear which NumPy axis to reverse. The instruction
also discusses only the bin map: for proper-motion data, reversing the spatial
x coordinate also requires transforming the associated x-directed velocity
coordinate/distribution. Future orbit projections will require the same frame
transformation.

This is scientifically relevant rather than editorial. Flipping the wrong
axis, or reflecting positions without the corresponding vector components,
changes the handedness or the observed velocity field.

Action:

- State the exact coordinate transformation, including which `bins` array axis
  is reversed for TNT's `(npix_x, npix_y)` convention and how the new `min_x`
  is calculated.
- State that every x-directed vector quantity must be transformed consistently;
  for proper-motion histograms, document the required reversal/sign change of
  the `vx` coordinate/distribution.
- Explicitly state that `(x, y) = (0, 0)` must denote the same sky centre as the
  MGE before projection.
- Prefer wording such as "reverse the x coordinate" over "flip along the
  x-axis."

References:

- `docs/source/data_preparation.md:20-28`
- `docs/source/configuration.md:181-190`
- `tnt/spatial_binnings.py:141-151`
- `tnt/kinematics/proper_motions.py:83-101`

### Low

#### L1. Current-state documentation still describes the removed schema

The new angular-reference-frame section is correct, but later knowledge and
units sections still say that `MGEs` maps names directly to files and that the
projected-binning field is `PA`. These statements contradict the PR's schema.
The new knowledge entry also contains unnecessary development-history wording
about an "earlier design"; current-state knowledge should directly describe
the chosen contract.

Action:

- Update the later knowledge section to describe `{file, major_axis_pa}` and
  `y_axis_pa`.
- Replace `PA` with `y_axis_pa` in the units documentation and mention that
  `major_axis_pa` is also preserved in its declared angular unit.
- Remove the development-history clause from the new knowledge entry.

References:

- `aidocs/KNOWLEDGE.md:66-73`
- `aidocs/KNOWLEDGE.md:309-322`
- `docs/source/units.md:147-150`
- `docs/source/units.md:162-166`

#### L2. The sign-sensitive test derives its expected angle from the production formula

The numerical integration test is valuable for the Gaussian integral, but its
"independent" reference calculates `alpha` with the same expression as the
production code. The separate convention test covers only exact 0/90-degree
alignments, for which reversing the sign is not distinguishable. A future sign
regression could therefore make the production formula and copied test formula
wrong together while leaving the axis-alignment test green.

Action:

- Add one non-right-angle, asymmetric-aperture regression that derives grid
  components from explicit north/east unit vectors rather than copying the
  `alpha` formula.
- Add a global-frame-rotation invariant: increasing both `major_axis_pa` and
  `y_axis_pa` by the same nontrivial angle must leave projected bin masses
  unchanged.
- Add explicit preservation assertions for `major_axis_pa` through
  `rescaled()`, `angular_to_physical()`, and `to_mass()`; the implementation is
  correct, but the current tests assert only the older fields.

References:

- `tests/unit_tests/test_mge.py:207-248`
- `tests/unit_tests/test_mge.py:580-587`
- `tests/unit_tests/test_mge.py:609-645`
- `tests/unit_tests/test_mge.py:691-738`

## Configuration, units, runtime, and numerical assessment

- **Configuration:** Changing `MGEs` from a filename string to a mapping is
  internally consistent across the representative configuration, schema
  validation, compatibility projection, and runtime builder. TNT is a new
  product, so no legacy-string compatibility path is needed.
- **MGE behavior:** Both `LightMGE` and `MassMGE` receive the same absolute
  orientation semantics. Light-to-mass conversion, rescaling, and
  angular-to-physical conversion preserve the orientation.
- **Units:** The PR respects TNT's preservation boundary. Configuration keeps
  `{value, unit}`; runtime objects keep the declared angular unit; projection
  strips to radians only for trigonometry. No conversion to the configured
  internal unit system is introduced.
- **JAX/Equinox:** The angles are dynamic `Quantity` leaves, not static Python
  metadata, so JIT can consume and vary them. The changed projected-mass path
  passes its JIT regression. No Python data-dependent control flow was added to
  the numerical kernel.
- **Precision:** Canonical domains would keep angle magnitudes small. Within
  those domains, the calculation is numerically ordinary. MGE axes are
  naturally pi-periodic through the Gaussian's doubled-angle expressions.
- **Validation ownership:** MGE configuration shape and declaration units are
  preparation-owned; MGE file content and runtime-object invariants are
  loading-owned. Spatial-binning entry validation remains runtime-owned. The
  recommended checks above preserve that split.
- **Future projection:** The two absolute angles are the correct inputs for
  projecting future orbit observables into multiple independent data grids.
  That orbit-projection implementation remains scaffolding and is not expected
  in PR #63.

## Missing or weak tests

The following test additions are recommended, grouped with their findings:

1. Canonical range endpoints and equivalent angular units for both new fields
   (M1).
2. Direct-reader rejection of wrong dimension, non-finite, and non-scalar
   `major_axis_pa` (M2).
3. A non-right-angle sign-sensitive sky-to-grid regression and simultaneous
   rotation invariant (L2).
4. Preservation of `major_axis_pa` through all MGE transformations (L2).
5. Configuration compatibility explicitly flags a change in
   `MGEs.<name>.major_axis_pa`; the current projection should already do so,
   but a direct regression would protect the new critical field.

The existing tests are otherwise strong: they cover both MGE kinds, mixed
declared units, numerical aperture integration, multiple twisted components,
physical conversion, bin aggregation, and JIT execution.

## Documentation assessment

The new data-preparation page is the right place for this contract, and its
distinction between `mgefit`'s `f.pa` and `f.theta` is useful. The configuration
example and DYNAMITE migration page are consistent with the new schema.

Before merge, the page needs the canonical ranges and an exact, complete
handedness-conversion recipe (M1 and M3). The stale knowledge/units statements
in L1 should also be corrected so the repository has one current description.

## Interaction with PR #60

PR #60 and PR #63 both modify `tnt/mge.py`, `aidocs/KNOWLEDGE.md`,
`docs/source/configuration.md`, and several MGE/potential tests. Even though
GitHub currently reports both branches mergeable against their present base,
they overlap logically and will need reconciliation after either one lands.

Recommended order:

1. Correct and merge PR #63 into `main`.
2. Rebase PR #60 onto the resulting `main`.
3. In PR #60's `(p, q, u)` reference-component re-anchoring, if the anchor
   twist is `delta`, preserve every component's absolute sky PA by applying
   both transformations:

   ```text
   major_axis_pa' = major_axis_pa + delta
   PA_twist_j'    = PA_twist_j - delta
   ```

4. Canonicalize the shifted `major_axis_pa'` modulo 180 degrees while
   preserving the component axes, and add a nonzero-twist invariant test.
5. Re-run PR #60's DYNAMITE cross-checks, inverse conversion, configuration,
   and full-suite tests.

Merging PR #60 first would force PR #63 to reason about the twist re-anchoring
inside a larger parameterization diff. PR #63 first establishes the simpler
and more general frame contract.

## Verification performed

All reported verification was run in the Linux `x86_64` development container
through Colima:

```text
docker --context colima compose run --rm dev pytest -q
370 passed, 1 dependency deprecation warning in 166.32s

docker --context colima compose run --rm dev ruff check .
All checks passed!

docker --context colima compose run --rm dev \
  sphinx-build -E -b html -W docs/source /tmp/tnt-pr63-docs
Build succeeded with warnings treated as errors.
```

The single test-suite warning comes from TensorFlow Probability's use of the
deprecated JAX `jax.core.pytype_aval_mappings` API; it is dependency-generated
and unrelated to PR #63.

## Practical review and merge sequence

1. Confirm the two canonical domains from issue #62: `[0, 180)` for the MGE
   axis and `[0, 360)` for the directed grid y-axis.
2. Implement M1 and M2 with one shared angle-domain helper if that reduces
   duplication without moving validation ownership.
3. Rewrite the reflection instructions in M3 and add the common-centre rule.
4. Correct the stale documentation in L1.
5. Add the range/runtime-input tests and at least one independent sign test.
6. Re-run the focused tests, full Linux suite, `ruff check`, and strict Sphinx
   after the corrections are implemented.
7. Merge PR #63, then rebase and finish PR #60 as described above.

## Decisions or questions for Prash and Thomas

1. Confirm that out-of-domain angles should be rejected rather than silently
   normalized. Rejection is recommended because declarations are preserved and
   compatibility is exact after unit conversion.
2. Confirm the exact data-preparation contract for opposite-handed inputs,
   especially proper-motion `vx`: must users transform all relevant arrays
   before TNT ingestion, or should a future explicit parity field perform that
   transformation at runtime? For this PR's fixed-parity design, explicit
   pre-ingestion transformation and precise documentation are recommended.
3. Confirm that all spatial-binning coordinates are relative to the same sky
   centre as the referenced MGE. The current projection mathematics assumes
   this and the documentation should say so.

---

## Response (2026-09-07, addressed at 4447c63's head + fixes)

Thanks -- the audit is accurate; every claim checked out against the code. All
five findings are now addressed on the branch.

### Decisions

1. **Reject out-of-domain angles.** Confirmed. `major_axis_pa` is enforced to
   `[0, 180)` degrees and `y_axis_pa` to `[0, 360)` degrees, by rejection, not
   normalization.
2. **Opposite-handed inputs: explicit pre-ingestion transform.** Confirmed for
   this fixed-parity design. The data-preparation page now gives the exact
   recipe (below); no runtime parity field.
3. **Common sky centre: yes, and now documented.** `data_preparation.md` was
   restructured around the three preparation criteria; its opening list and a
   dedicated "Common origin" section state that every spatial input shares one
   origin at the galaxy centre, while orientation may differ per data set.

### M1 -- position-angle domains

- New shared helper `tnt.units.validate_position_angle(angle, *, minimum_deg,
  maximum_deg, path)` (a unit-aware peer of `validate_dimension`): rejects a
  non-finite, non-scalar, non-angular, or out-of-domain value. The domains
  themselves are semantic facts about the owning fields, so they live with
  those fields: `tnt.mge.MAJOR_AXIS_PA_DOMAIN_DEG` and
  `tnt.spatial_binnings._Y_AXIS_PA_DOMAIN_DEG`.
- `major_axis_pa` `[0, 180)` enforced at the config boundary
  (`_validate_mges`, now via `declared_quantity` + `validate_position_angle`)
  and at runtime construction (`AbstractMGE.from_qtable`).
- `y_axis_pa` `[0, 360)` enforced in `_declared_quantities`, i.e. for both
  `ProjectedBinning.from_settings()` and `build_spatial_binnings()`.
- Domains documented in `configuration.md`, `units.md`, `data_preparation.md`,
  the `AbstractMGE`/`ProjectedBinning` docstrings, and `KNOWLEDGE.md`.
- Tests: endpoint acceptance/rejection (0, excluded upper, negative) and
  radian-equivalent declarations, in `test_mge.py`, `test_spatial_binnings.py`,
  and `test_configuration.py`.

### M2 -- public MGE readers

`from_qtable()` (and therefore `read()` / `read_mge()`) now runs
`validate_position_angle` on `major_axis_pa` before construction: rejects a
non-angle unit, NaN/infinity, a non-scalar array, and out-of-domain values,
with `Raises` docs updated. Tests added in `test_mge.py`. Bare
`LightMGE(...)`/`MassMGE(...)` construction is deliberately left unchecked
(no `__check_init__`) -- it is the low-level "trust the caller" path, and a
Python-level check there would break tracing when an MGE is reconstructed
under `jax.jit`.

### M3 -- opposite-handed conversion

`data_preparation.md` now gives the exact transformation: reverse the bin
array along its first (`npix_x`) axis (`bins = bins[::-1, :]`), set `min_x`
to `-(min_x + x_extent)`, and negate every x-directed vector quantity --
naming proper-motion `vx` explicitly. The common-centre rule is stated in the
new opening section. `KNOWLEDGE.md` carries the same recipe.

### L1 -- stale schema docs

- `KNOWLEDGE.md`: the "current schema exceptions" bullet and the registry
  description now say `{file, major_axis_pa}` and `y_axis_pa` (no `PA`).
- `units.md`: the MGE bullet now covers `major_axis_pa`; the spatial-binning
  bullet says `y_axis_pa`, both with their domains.
- The "earlier design" development-history clause is removed from the new
  `KNOWLEDGE.md` entry.

### L2 -- test independence

- `_brute_force_aperture_mass` no longer copies the production `alpha`
  formula: it builds the geometry from explicit `(east, north)` unit vectors
  (`PA -> (sin, cos)`), so the existing non-right-angle, asymmetric-aperture
  parametrized cases now independently check the sky-to-grid composition.
- New `test_get_projected_mass_invariant_under_global_frame_rotation`: adding
  the same angle to `major_axis_pa` and `y_axis_pa` (canonicalized mod 180 /
  mod 360) leaves every bin mass unchanged.
- `major_axis_pa` preservation asserts added to the `to_mass`, `rescaled`,
  and `angular_to_physical` tests.
- `test_configuration_compatibility.py`: a `MGEs.<name>.major_axis_pa` change
  is now an explicit case in `test_critical_configuration_changes_are_rejected`.

### PR #60 interaction

Agreed: land #63 first, then rebase #60 and apply the paired
`major_axis_pa += delta` / `PA_twist_j -= delta` transform in the twist
re-anchoring, canonicalizing the shifted `major_axis_pa` mod 180, with a
nonzero-twist invariant test.

### Verification (local, macOS)

```text
uv run pytest -q            -> 395 passed, 1 dependency warning
uv run ruff check .         -> All checks passed!
uv run sphinx-build -E -b html -W docs/source ...  -> build succeeded
```

(395 = 370 previously + 25 new tests for the findings above.)

---

## Second-pass review (2026-09-07)

A fresh full-diff review turned up no code defects; the fixes above hold. A
few follow-ups were applied on top:

### Semantic redefinition of the old `PA` (no releases affected)

`PA` -> `(major_axis_pa, y_axis_pa)` is a redefinition, not a rename. The old
single `ProjectedBinning.PA` was documented as measured *counterclockwise
from the grid y-axis* with no declared on-sky parity, and consumed as
`alpha = PA - pi/2 + PA_twist`. The new fields are absolute astronomical PAs
(north through east) with the grid parity fixed, giving
`alpha = y_axis_pa + pi/2 - major_axis_pa - PA_twist`. Copying an old `PA`
value into either new field does **not** reproduce the previous projected
geometry (with `y_axis_pa = 0`, `major_axis_pa = PA` the new `alpha` is the
negation of the old, i.e. a mirror image). TNT has no tagged releases, so no
existing configuration or run is affected; noting it so the change isn't
mistaken for a field rename.

### `data_preparation.md` MGE section

Rewritten against `mgefit` 6.2.6 (docstrings + shipped `mge_fit_example.py`):
- corrected the fitting snippet to the real flat API (`import mgefit as mge`;
  `sectors_photometry(image, f.eps, f.theta, ...)` returning `.radius/.angle/
  .counts`; `mge.fit_sectors`);
- `f.pa` is mgefit's documented astronomical PA (north through east, y-axis =
  north) -> use as `major_axis_pa`; `f.theta` (`f.pa = 270 - f.theta`) is the
  image-frame angle `sectors_photometry` consumes;
- plain `fit_sectors` fits one PA (no twist); `fit_sectors_twist`'s per-
  Gaussian `sol[3]` is in the image-frame sense, opposite to astronomical PA,
  so those angles must be negated to become `PA_twist`. Replaces the earlier
  vague "check against a component whose orientation you know";
- added: do not substitute the kinematic PA for a missing photometric one;
- removed the "rejected, not wrapped" paragraph (already in
  `configuration.md`).

### Minor hardening

- `validate_position_angle` now rejects a non-`Quantity` argument with a
  `ValueError` (was an opaque `AttributeError` on `.unit`); test added.
- `ProjectedBinning` docstring: "must be flipped before use" -> "must be
  converted before use -- see the data-preparation guide".

### Open (not blocking)

- The example configs (`configuration.md`, `tests/integration_tests/
  configuration.yaml`) pair `y_axis_pa: 0` with `major_axis_pa: 126`.
  Investigated: DYNAMITE stores the bin map in the input-data frame (grid +y
  ~ north, consistent with the legacy `90 - PA` aperture convention), so the
  `0` is plausible for the angle -- but CALIFA DR3 cubes are north-up/
  east-left (opposite parity to TNT), so `bins.npy` may need the
  `bins[::-1, :]` + `min_x` conversion and doesn't have it. Undeterminable
  from the repo (depends on the map-building sign convention). Inert today (no
  projected-mass assertion); recorded in `aidocs/KNOWLEDGE.md` as a fixture to
  check when first cross-checking TNT projections against DYNAMITE.

### Re-verification (local, macOS)

```text
uv run pytest -q            -> 396 passed, 1 dependency warning
uv run ruff check .         -> All checks passed!
uv run sphinx-build -E -b html -W docs/source ...  -> build succeeded
```
