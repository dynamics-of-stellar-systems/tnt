# PR #76: MGE physical surface density audit

- Date: 2026-09-30
- Reviewer: Codex, for Thomas Maindl
- PR: [Bug fix: MGE surface density to the standard physical convention](https://github.com/dynamics-of-stellar-systems/tnt/pull/76)
- Author: `prashjet`
- Reviewed head: `4b6303c264ba0714d8e01f7b99480d9fd1137a02`
- PR base commit: `f34f21e6dc6ecbe46585c478653a8713461479b0`

## Recommendation

The scientific correction makes sense. No high-priority numerical or production
correctness defect was found. Make the small documentation, test-fixture, and
lint corrections below before merging. A redesign of the implementation is
not needed based on this audit.

This audit covers the actual PR head in an isolated checkout. Main has since
advanced through PR #74; the combined result with that newer main was not
tested here. GitHub reports PR #76 as mergeable.

## Findings

### F1 — P2: The documented input convention still contradicts the implementation

Locations at the reviewed head:

- `docs/source/units.md:149–151` says `angular_to_physical()` works for angular
  units of both `sigma` and `I`.
- `tnt/mge.py:1060–1062`, the `read_mge()` docstring, still identifies light and
  mass intensity dimensions as `power/angle**2` and `mass/angle**2`.

Those instructions now describe inputs that the ECSV loader rejects. The PR
changes the required intensity dimensions to power or mass per physical area;
only `sigma` is converted using distance. This is material input documentation
for a correction that changes scientific normalization.

Requested fix: state the ECSV contract explicitly: `I` in physical surface
brightness/density units, such as `Lsun/pc2` or `Msun/pc2`; `sigma` in an
angular unit; `q` dimensionless; `PA_twist` angular. Correct the loader
docstring to `power/length**2` and `mass/length**2`, and say that only `sigma`
changes with distance. Add this convention to the relevant current project
knowledge or data-preparation documentation. Clarify the new class docstring
at `tnt/mge.py:265–267`: file-loaded widths start angular, while returned
physical MGEs and direct constructors can already carry physical widths.

### F2 — P3: Two iterator tests still construct obsolete angular intensities

Locations: `tests/unit_tests/test_model_iterator.py:246` and `:292`.

Both construct `LightMGE` with `I` in `Lsun/rad2`, then call
`angular_to_physical()`. Under the new implementation that intensity remains
angular. The tests can still pass because their invalid geometry is rejected
before a physical potential is built, and direct construction bypasses the
ECSV dimension checks. They therefore retain fixtures that no longer represent
valid runtime input.

Requested fix: change those fixture intensities to `Lsun/pc2`, keeping the
geometry and intended rejection assertions. No new production validation is
required to resolve this finding.

### F3 — P3: The project lint check fails on the new helper docstring

Location: `tnt/mge.py:39`.

`ruff check --no-cache .` reports `E501 Line too long (100 > 88)` on the first
line of `_rebased_unit()`'s docstring. `pyproject.toml` explicitly enables
this rule.

Requested fix: wrap or shorten that line and rerun the project lint check.

## Scientific and implementation review

- The physical-intensity/angular-width convention agrees with Cappellari's
  [JamPy documentation](https://pypi.org/project/jampy/), which specifies
  peak light intensity in `Lsun/pc2`, mass intensity in `Msun/pc2`, and
  projected Gaussian width in arcseconds.
- Leaving physical `I` unchanged and converting only `sigma` is consistent
  with the Gaussian total `2*pi*I*sigma_physical**2*q`. For fixed observed
  angular width, total luminosity or mass scales as distance squared.
- The new projected-integration unit conversion correctly accounts for the
  difference between intensity's area unit and the binning's coordinate unit.
  The returned quantity retains the units needed to recover physical totals.
- Existing oblate and triaxial deprojection formulas are dimensionally
  compatible with the corrected convention: physical surface intensity is
  divided by physical width to obtain intrinsic volume density.
- The changed integration fixtures retain their numerical intensities and
  change their unit declarations to physical area units. The PR's exact
  DYNAMITE `orblib.log` comparison and its reported historical error factor
  were not independently verified: that comparison log is not present in
  the reviewed project files. Independent analytic checks are recorded below.

## Verification

Commands ran against the reviewed PR head using the project's Docker runtime:

```sh
docker compose run --rm -e PYTHONDONTWRITEBYTECODE=1 dev pytest -q tests/unit_tests tests/integration_tests -p no:cacheprovider
docker compose run --rm -e PYTHONDONTWRITEBYTECODE=1 dev pytest -q tests/unit_tests -p no:cacheprovider
docker compose run --rm -e PYTHONDONTWRITEBYTECODE=1 dev pytest -q tests/integration_tests -p no:cacheprovider
docker compose run --rm dev ruff check --no-cache .
docker compose run --rm -e PYTHONDONTWRITEBYTECODE=1 dev sphinx-build -E -b html -W docs/source /tmp/pr76-docs
git diff --check f34f21e6dc6ecbe46585c478653a8713461479b0 HEAD
```

- Full unit and integration suite: the combined run was stopped after progress
  stalled without a reported assertion failure. Docker diagnostics showed
  `1.686 GiB / 1.92 GiB` memory usage and `122 GB` block reads, consistent with
  memory pressure in the local runtime. Separate runs reduce accumulated
  compilation memory:
  - Unit tests: **477 passed**, one dependency deprecation warning, in
    `212.73 s`.
  - Integration tests: **12 passed**, the same dependency deprecation
    warning, in `42.52 s`.
  - Together: **all 489 tests passed** across the two separate runs. The
    interrupted combined run is not counted as a completed passing run.
- Ruff: **failed**, exactly one `E501` violation, described in F3.
- Sphinx documentation build with warnings treated as errors: **passed**.
  This confirms the documentation builds; it does not detect the stale meaning
  described in F1.
- Whitespace check: **passed**.

Additional independent calculations in the same runtime, with JAX float64:

- A circular Gaussian with `I = 5 Lsun/pc2`, angular `sigma = 1 arcsec`,
  and distance `30 Mpc` gives `664572.116775973 Lsun`. Numerical projection
  over a grid covering ten Gaussian widths in each direction agrees with
  `2*pi*I*sigma_physical**2` to approximately `1e-15` relative error for
  coordinate units `pc`, `kpc`, and `m`, keeping `I` in `Lsun/pc2` throughout.
- Repeating at `60 Mpc` gives four times the total. The oblate deprojected
  Gaussian total agrees with the projected analytic total at both distances.
- At `30.5 Mpc`, the light integration fixture totals
  `9.466048388393452e9 Lsun`, and the mass fixture totals
  `1.8882646124457733e10 Msun`. For each individual Gaussian, the total from
  an edge-on oblate deprojection agrees with the independently evaluated
  projected formula to within `1e-12` relative tolerance.

Optional improvement: preserve a distance-squared regression check in the
test suite, since it directly protects the physical behavior this PR repairs.
This is supplementary coverage, not an additional correctness finding.

## Scope and follow-up

Only this audit document was added to the primary workspace. Implementation
fixes, publication of review comments, and merging are separate follow-up
actions. After the findings are resolved, rerun the affected tests and lint,
then remove the audit from the PR branch according to the project workflow.

Assisted by Codex (OpenAI).
