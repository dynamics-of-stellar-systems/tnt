# PR #47 — Unit System Threading Audit

Audited pull request: **#47, “Validate unit dimensions on input; reserve the
internal unit system for galax”** (`unit-system-threading` into `main`).

Audit date: 2026-08-28.

## Overall judgment

**Needs focused corrections, not architectural revision.**

The runtime design and scientific unit transformations appear sound. No
critical or high-severity code defects were found. Before merging, the stale
examples and architectural statements in `docs/source/configuration.md` and
`aidocs/KNOWLEDGE.md` should be corrected. The additional test improvements
listed below are worthwhile but are not considered merge blockers.

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

### Medium — `configuration.md` still documents removed APIs and old behavior

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

### Medium — `aidocs/KNOWLEDGE.md` contradicts the new architecture

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

### Low — current-state documentation contains unnecessary migration history

Because TNT is a new product, durable repository documentation does not need
to explain that `power` “used to be required” or characterize this as a
breaking schema change:

- `docs/source/units.md:31`;
- `aidocs/KNOWLEDGE.md:94`; and
- `tnt/units.py:18`.

The current-state explanation only needs to say that internal units consist
of length, time, mass, and angle, and that derived dimensions must not be
declared there.

## Missing or weak tests

These are recommended improvements, not merge blockers:

1. Add a Gauss-Hermite regression where `v`, `dv`, `sigma`, configured
   systematics, histogram width, and histogram center use different but
   equivalent speed units. The existing test at
   `tests/unit_tests/test_kinematics.py:62` uses `km / s` throughout.
2. Verify that NFW inverse reporting returns `M_200` in a configured unit
   other than `Msun`. The current raw-reporting test at
   `tests/unit_tests/test_potential.py:520` configures `Msun`.
3. Update direct potential-test helpers to represent the new four-base unit
   system. They still construct a five-unit system containing `Lsun` at
   `tests/unit_tests/test_potential.py:71` and
   `tests/unit_tests/test_model_iterator.py:110`. The realistic integration
   test already exercises the actual four-key configuration, so this is
   primarily test clarity.

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

No native Apple Silicon run was performed. GitHub reported no CI check
results or submitted reviews at audit time. The pull request was Git-mergeable,
although its merge-state status was `BLOCKED`, presumably pending required
review or checks.

## Interaction with other work

PR #47 is based on the current `main`, including PR #45. Recommended order:

1. Correct PR #47's documentation.
2. Merge PR #47.
3. Rebase or update PR #48 against it, then audit PR #48.
4. Implement issue #44 against the resulting settled parameterization and
   unit interfaces.

Issue #43 does not materially affect this unit-boundary change.

## Recommended path to merge

1. Correct the stale examples and behavior descriptions in
   `docs/source/configuration.md`.
2. Reconcile `aidocs/KNOWLEDGE.md` with the new runtime boundary and current
   converter interfaces.
3. Remove unnecessary prior-schema history from durable current-state
   documentation.
4. Add the focused mixed-unit and configured-output-unit regression tests.
5. Rerun the Linux test, lint, and warning-as-error documentation checks.

After the documentation corrections, the pull request should be suitable for
approval; the recommended tests provide stronger regression protection but do
not expose a known runtime defect.
