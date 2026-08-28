# PR #47 — Unit System Threading Audit

Audited pull request: **#47, “Validate unit dimensions on input; reserve the
internal unit system for galax”** (`unit-system-threading` into `main`).

Audit date: 2026-08-28.

## Overall judgment

**Ready to merge from this audit's perspective.**

The runtime design and scientific unit transformations appear sound. No
critical or high-severity code defects were found. The focused documentation
corrections and selected regression-test improvements identified by this audit
have now been completed. No architectural revision or further audit-driven
change remains necessary before merging.

## Architectural summary

PR #47 establishes a clearer unit boundary:

- Configuration values and observational inputs retain their declared units.
- Runtime readers validate physical dimensions without normalizing everything
  into `units.internal`.
- MGEs and spatial binnings still undergo the scientifically necessary
  angular-to-physical projection using the system distance.
- Population uncertainties are converted only into their paired value
  column's unit.
- Potential parameters remain in declared units through parameter generation
  and potential construction.
- Registered parameterizations use unit-aware arithmetic. The NFW inverse
  restores `M_200` to its configured unit.
- `units.internal` is reserved for constructing `galax` potential objects
  through `Potential.to_galax()`.
- `ModelIterator.unit_system` is retained correctly for future orbit
  integration.

This is cleaner than threading a unit system through readers that do not
require a shared representation.

## Findings

### Resolved medium — `configuration.md` documented removed APIs and old behavior

The examples pass `config.unit_systems.internal` to functions whose
`unit_system` argument has been removed:

- `docs/source/configuration.md:218` — `build_mges`;
- `docs/source/configuration.md:246` — `build_spatial_binnings`;
- `docs/source/configuration.md:320` — combined MGE, binning, and kinematics
  example; and
- `docs/source/configuration.md:373` — `build_populations`.

Those snippets now raise `TypeError`.

The surrounding prose also retains behavior removed by this pull request:

- `docs/source/configuration.md:258` says spatial geometry is first converted
  into the internal angle unit; and
- `docs/source/configuration.md:388` says population properties are converted
  into the internal unit system.

These passages should describe preservation of declared units and the
separate angular-to-physical projection. This is the only issue considered
necessary to correct before merging.

**Resolution:** The obsolete `unit_system` arguments were removed from all
four examples. The spatial-binning and population descriptions now explain
preservation of declared units, local pair conversion, and the separate
angular-to-physical projection. The warning-as-error Sphinx build passes.

### Resolved medium — `aidocs/KNOWLEDGE.md` contradicted the new architecture

Several entries still describe the old conversion model:

- `aidocs/KNOWLEDGE.md:119` says runtime kinematics construction converts
  histogram quantities and systematic uncertainties;
- `aidocs/KNOWLEDGE.md:142` frames construction primarily as a conversion
  boundary;
- `aidocs/KNOWLEDGE.md:413` describes the old converter signature and names
  `ParameterizationConverter`;
- `aidocs/KNOWLEDGE.md:420` says `unit_system` is threaded into potential
  construction;
- `aidocs/KNOWLEDGE.md:439` says NFW results are converted to the unit system
  at the end; and
- `aidocs/KNOWLEDGE.md:450` still lists `unit_system` as parameterization
  context.

These contradictions should be corrected in the same documentation pass.

**Resolution:** Current project knowledge now records the actual runtime unit
boundary, on-demand local conversions, the separate `Potential.to_galax()`
unit-system path, the `ForwardConverter`/`InverseConverter` signatures, NFW's
unit-aware arithmetic and configured-unit inverse, and the authoritative
sources of parameter-dimension metadata.

### Resolved low — current-state documentation contained migration history

Because TNT is a new product, durable repository documentation does not need
to explain that `power` “used to be required” or characterize this as a
breaking schema change:

- `docs/source/units.md:31`;
- `aidocs/KNOWLEDGE.md:94`; and
- `tnt/units.py:18`.

The current-state explanation only needs to say that internal units consist
of length, time, mass, and angle, and that derived dimensions must not be
declared there.

**Resolution:** `docs/source/units.md`, `aidocs/KNOWLEDGE.md`, and
`tnt/units.py` now describe only the current four-base schema and its rationale.
Prior-schema and breaking-change language was removed.

## Test follow-up

The original recommendations were resolved as follows:

1. **Addressed in the existing Gauss-Hermite test:** `v` remains declared in
   `km / s`, while `dv` is now declared in `m / s`. The test asserts that the
   resulting uncertainty retains `m / s` and verifies its physical value in
   `km / s` after quadrature with a `km / s` systematic. This covers the
   on-demand mixed-unit conversion without duplicating the construction test.
2. **Intentionally unchanged:** the NFW raw-reporting test continues to use
   `Msun`. Separate tests already establish NFW unit invariance, and the final
   return to the configured unit is a direct `Quantity.to(declared_unit)`
   operation. Another otherwise-duplicate end-to-end test was not considered
   proportionate.
3. **Addressed:** the direct potential and model-iterator test fixtures now
   construct the current four-base internal unit system without `Lsun`.

Manual probes confirmed that mixed equivalent histogram speed units work in
both eager execution and `jax.jit`, and that the corrected MGE conversion is
invariant for intensity declared per steradian.

## Verification performed

Checks were run through the Linux x86-64 Colima development environment:

- `ruff check .` — passed.
- `sphinx-build -E -b html -W docs/source docs/build/html` — passed.
- Full `pytest` suite — 317 passed and one isolated-process JAX precision
  test exceeded its 30-second subprocess timeout.
- The timed-out test was rerun alone and passed. This appears to be timing or
  resource sensitivity rather than a PR #47 functional failure.
- `ruff format --check .` — reported 14 files requiring formatting. This
  includes existing project-wide formatting drift as well as PR-touched
  files; it is not a functional failure.
- After the follow-up test changes, the three affected unit-test modules ran
  successfully: 86 passed.
- After the documentation follow-up, `ruff check tnt/units.py`,
  `git diff --check`, and the warning-as-error Sphinx build passed.

No native Apple Silicon run was performed. GitHub reported no CI check
results or submitted reviews at audit time. The pull request was Git-mergeable,
although its merge-state status was `BLOCKED`, presumably pending required
review or checks.

## Interaction with other work

PR #47 is based on the current `main`, including PR #45. Recommended order:

1. Merge PR #47.
2. Rebase or update PR #48 against it, then audit PR #48.
3. Implement issue #44 against the resulting settled parameterization and
   unit interfaces.

Issue #43 does not materially affect this unit-boundary change.

## Recommended path to merge

The documentation corrections, focused regression strengthening, and affected
Linux checks are complete. Subject to the repository's ordinary required
review or CI policy, PR #47 is suitable for approval and merge.
