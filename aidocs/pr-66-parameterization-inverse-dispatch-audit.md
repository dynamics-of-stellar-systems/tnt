# PR 66: centralized parameterization inverse-dispatch audit

Audit date: 2026-09-11.

PR: [Centralize non-native parameterization inverse dispatch](https://github.com/dynamics-of-stellar-systems/tnt/pull/66).

**Decision: ready to merge. No blocking findings.**

## Reviewed state and scope

Reviewed commit: `cfeb79cc4e7eec378e28b411936b3a968345bb2b`, the remote
`origin/centralize-parameterization-inverse-dispatch` and GitHub PR 66 head
at review startup. The remote was fetched, and the checked-out local branch
matched that commit with a clean worktree. Comparison base: `origin/main`
at `2feff769b0151c57b58db34a27c25a41c86e7322`.

Read the supplied user instructions, `README.md`, relevant architecture,
units, parameterization and workflow sections of `aidocs/KNOWLEDGE.md`, the
complete PR diff, and [issue 65](https://github.com/dynamics-of-stellar-systems/tnt/issues/65).
No project `AGENTS.md`, `CLAUDE.md` or `aidocs/INDEX.md` was present.
The review covers implementation correctness, registry lookup, converter
context, declared units, rescaling, model-table reporting and test coverage.

## Current behavior and architectural assessment

Inverse dispatch means selecting and calling the converter that reports a
built component in its configured parameter names and units.

- `AbstractPotentialComponent.raw_parameters` is the single implementation.
  With no parameterization, it returns the canonical parameter mapping.
  With a parameterization, it looks up `(type_name, parameterization)` and
  invokes the registered inverse. An unknown name raises
  `NotImplementedError`; it does not silently return canonical parameters.
- `_registry_type_name()` returns the registered TNT class's `_type`.
  `GalaxPotentialComponent` supplies its concrete `galax_type` instead.
  This is an appropriate small polymorphic hook: it expresses the actual
  difference in registry identity while leaving dispatch in one place.
- The inverse receives current canonical parameters, declared units,
  cosmological parameters and the component's `mge` field, or `None` when
  no MGE is present. All four MGE component classes retain that field.
  `ResolvedPotentialComponent.build` supplies the corresponding forward
  context from resolved extra fields.
- `Potential.rescale` produces rescaled canonical parameters; inverse
  reporting therefore calculates the appropriate raw values from those
  parameters. `raw_potential_parameters` passes them through to
  `Model.raw_parameters`, which supplies `AllModels` column names and units.
- Registry declarations remain authoritative. There are no parallel type
  lists or per-type inverse implementations. Converter formulas, physical
  domain checks, potential construction and JAX precision policy are
  unchanged by this PR.

No implementation correctness or architectural blocker was identified by
source review or the completed runtime checks below.

## Non-blocking observations

### Strengthen the configured round-trip regression test

`tests/unit_tests/test_potential.py:1093-1130` directly verifies inverse
dispatch on an oblate light component and checks the MGE context. However,
it registers `_identity_forward` with a `doubled_ml` raw schema, builds a
native component without selecting that parameterization, then calls the
inverse manually. Selecting this particular test parameterization during
construction would fail the canonical parameter-name check because the
identity converter does not map `doubled_ml` to `ml`.

This is a limitation of the test fixture, not a production converter bug.
A stronger permanent regression should provide a matching forward converter,
select the parameterization in component settings, build it, rescale it and
assert `AllModels` column names and values. Including a component without an
MGE would cover the generic `None` context. The independent audit probe
described below exercises these paths without adding production registrations.

### Retain tracking for deferred context design

The primary inverse-dispatch goal of issue 65 is addressed. Its related
shared-converter-context proposal remains outside this PR. The PR also
describes separately capturing cosmological parameters in the resolved
once-per-run structure; this is not the same change as merely combining the
converter arguments into a context object.

Closing issue 65 should not imply either context change is implemented.
Retain a separate follow-up if that work is still intended. Neither change
is required for correct dispatch here: current forward and inverse calls
explicitly receive cosmological parameters, and their signatures agree.

## Validation

The documented Linux/amd64 Compose environment is used for validation.

- Full suite, `docker compose run --rm dev pytest -q`: **443 passed** in
  833.70 seconds, with one dependency-owned TensorFlow Probability/JAX
  deprecation warning. No failed tests.
- `ruff check .`: passed.
- Strict Sphinx (`sphinx-build -E -b html -W docs/source /tmp/pr66-sphinx`):
  passed.
- `ruff format --check` on all four changed Python files: passed.
- `git diff --check origin/main...HEAD`: passed.
- New audit document: no whitespace errors in the diff check.

The 2 GB Colima VM experienced severe memory pressure while the suite and
additional probe overlapped. The probe was stopped, allowing the same suite
run to finish successfully; the independent probe was then restarted alone.
The first, interrupted probe is not counted as a validation pass. No VM
restart or configuration change was made. Native macOS tests were not run.

The independent probe uses matching test-only forward and inverse converters
registered under the same parameterization name for oblate light, oblate
mass and a newly registered non-MGE component. It checks type-specific
lookup, MGE/`None` context, cosmological context forwarding, units declared
as `kg / Lsun` and `arcmin`, raw names in `AllModels`, and normalization
after a factor-three mass rescaling. It also checks identity reporting
without a parameterization and explicit errors for unknown names.

The standalone run **passed all six component/scale combinations** (three
component types, each unscaled and rescaled by three), including the
`AllModels` table assertions. It exercised resolved component settings and
the runtime construction/reporting path, not YAML configuration preparation.
The existing full suite covers the preparation layer and registered NFW and
PQU reporting behavior.

## Merge assessment

**Recommend approval and merge.** The primary issue 65 behavior is implemented
centrally, and both the existing suite and independent end-to-end checks
pass. The observations above are non-blocking; they do not identify a
production failure requiring another implementation cycle.

GitHub was checked again after validation and still reports reviewed head
`cfeb79cc4e7eec378e28b411936b3a968345bb2b`, technically mergeable, with
`REVIEW_REQUIRED` and no status checks. This recommendation is the audit
assessment, not a submitted GitHub approval.

Only this audit document was added. No production code or repository tests
were changed, and no GitHub review, comment, commit, push or merge was
submitted. Retain the audit for review; the project workflow calls for
removing it before merge once the findings have been handled.
