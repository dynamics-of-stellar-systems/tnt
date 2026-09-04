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
