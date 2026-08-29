# PR 51 audit: remove `potential.<name>.include`

Audit date: 2026-08-29

Pull request: #51, `Remove potential.<name>.include`

Base/head reviewed: `main` at `3211481` / `remove-potential-include` at
`3020a5d`.

## Overall judgment

The implementation is coherent and appropriately scoped. It removes the
`include` branch from configuration preparation, potential resolution, resume
column expectations, defaults, fixtures, and the primary potential
documentation. Every declared component now participates in the potential,
which is simpler and matches the decision recorded in issue #43. I found no
functional defect, no critical or high-severity finding, and no architectural
revision is needed.

I recommend **two focused corrections before merge**:

1. add a direct regression for the changed resume/model-table parameter-column
   rule; and
2. remove the remaining stale `inclusion` wording from current-state
   documentation and record the final schema decision in `KNOWLEDGE.md`.

After those small test/documentation changes, PR 51 should be ready to merge.

## Architectural summary

Before this PR, a configured potential component could remain in the resolved
configuration while `Potential.resolve()` filtered it out when
`include: false`. Configuration validation consequently allowed such a
component to omit otherwise-required fields, and resume validation ignored its
model-table parameter columns.

PR 51 removes that alternate state:

- configuration preparation rejects `include` as an unknown component field;
- every component must declare a non-empty `parameters` mapping;
- registered MGE components always require a valid `mge` reference and their
  complete registered parameter schema;
- `Potential.resolve()` resolves every declared component;
- parameter generators already iterate over every declared component and now
  naturally remain aligned with the resolved potential;
- resume validation expects parameter columns for every declared component;
  and
- the packaged `dynamic_object_defaults.potential.include` default disappears.

Removing or commenting out the complete component entry is now the sole way to
exclude it. This eliminates a three-way conditional spanning preparation,
runtime construction, and persisted model compatibility.

## Prioritized findings

### Critical

None.

### High

None.

### Medium 1: the changed resume parameter-column rule has no direct regression

References:

- `tnt/configuration/compatibility.py:137`
- `tnt/configuration/compatibility.py:145`
- `tests/unit_tests/test_configuration_compatibility.py:398`
- `tests/unit_tests/test_configuration_compatibility.py:418`

`_validate_parameter_columns()` correctly removed its `include` filter and now
collects columns from every declared component. This is one of the three
load-bearing behavior changes identified by issue #43, but the compatibility
tests do not directly exercise it. The existing resume tests either use an
empty `AllModels` table or fail earlier on a changed configuration or missing
chi-square column. No test would fail if the expected-column comprehension
were accidentally narrowed or bypassed later.

Add a focused regression with a non-empty `AllModels` table and at least two
declared components where one required component parameter column is absent.
Assert that `ensure_resume_compatible()` raises
`ConfigurationCompatibilityError` naming that missing column. A companion
valid case is useful if it fits naturally, but the missing-column regression
is the important one.

This is not evidence that the current one-line implementation is wrong; it is
coverage for a persisted-search safety contract that this PR deliberately
changes.

### Low 1: current-state documentation still refers to component inclusion

References:

- `docs/source/model_search.md:180`
- `docs/source/model_search.md:203`
- `aidocs/KNOWLEDGE.md:242`
- `aidocs/KNOWLEDGE.md:494`
- `docs/source/configuration.md:68`

`docs/source/model_search.md` still says compatibility rejects changes to
potential-component "inclusion" and that model tables require columns for
"included" components. There is no longer an inclusion property; these should
say that component additions/removals are incompatible and that every declared
component's parameter columns are required.

The durable project knowledge does not yet record the completed schema
decision. Add a concise current-state statement that every declared potential
component is active and that exclusion means removing/commenting its entry.
The general defaults wording in `KNOWLEDGE.md` and `configuration.md` should
also be checked: the packaged defaults now contain shared parameter defaults
but no potential-component default. It can still describe the generic merge
mechanism, but should not imply that the current packaged profile supplies a
component property.

This documentation drift does not affect runtime behavior, and strict Sphinx
does not detect it because the prose remains syntactically valid.

## Implementation assessment

### Configuration and validation

The removed key is rejected clearly rather than silently ignored. Requiring
`parameters`, a non-empty mapping, the MGE reference, and the registered MGE
schema unconditionally is the correct simplification. Native `galax`
parameter completeness remains intentionally deferred to issue #44 and should
not be added to this PR.

No migration compatibility layer is needed: TNT is new, and rejecting an old
`include` declaration is preferable to interpreting it ambiguously. In
particular, a former `include: false` profile must not silently start including
that component during normal configuration preparation.

### Runtime objects and model search

`Potential.resolve()`, `Potential.build()`, `SinglePointParameterGenerator`,
raw-parameter reporting, potential rescaling, and `AllModels` column naming now
operate on the same component set. There is no longer a configuration/runtime
asymmetry. The change does not affect JAX tracing, Equinox PyTree structure,
units, numerical precision, MGE conversion, or `galax` construction.

### Compatibility and existing searches

Component addition/removal remains compatibility-critical through the
projected potential schema. A resolved historical configuration that contains
the removed `include` field will compare differently from the new schema; that
is a reasonable consequence of the intentional schema removal. Scientific
input files and parameter value/unit flexibility are otherwise unchanged.

## Tests and checks performed

On `remove-potential-include` at `3020a5d`:

- `docker --context colima compose run --rm dev pytest`
  - 331 passed; three pre-existing dependency/multiprocessing warnings.
- `docker --context colima compose run --rm dev ruff check .`
  - passed.
- `docker --context colima compose run --rm dev sphinx-build -E -b html -W docs/source docs/build/html`
  - passed with warnings treated as errors.
- `git diff --check origin/main...HEAD`
  - passed.

No native Apple Silicon run was performed. The PR contains no platform-specific
code or dependency change and uses runtime paths already covered on supported
platforms.

GitHub reported no CI checks or reviews at audit time. The PR was conflict-free
and `MERGEABLE`; its `BLOCKED` state reflected the required-review policy.

## Interactions and merge order

- PR 51 resolves issue #43.
- Issue #44 should be implemented after PR 51, as proposed. It can then extend
  exact parameter completeness to recognized native `galax` types without
  carrying an obsolete inclusion branch.
- Issue #30's physical-domain validation remains separate and unaffected.
- Work depending on the settled potential registry/schema, including the
  previously discussed follow-up around PR 33, should consume PR 51 before
  finalizing its own validation behavior.

## Recommended review sequence

1. Confirm the issue #43 product decision: every declared component is active.
2. Review configuration rejection of `include` and unconditional requirements
   in `tnt/configuration/validation.py`.
3. Review the removal of filtering in `tnt/potential/core.py`.
4. Add/review the resume parameter-column regression.
5. Correct the stale model-search and project-knowledge wording.
6. Re-run the full Linux suite, Ruff, and strict Sphinx.
7. Merge, then implement issue #44 on the simplified schema.

## Open design questions

None. Issue #43's discussion already settled the only substantive product
decision. The remaining work is test and documentation hardening.
