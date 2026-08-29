# PR 52 audit: native parameter-completeness validation

Pull request: #52, `Extend parameter-completeness validation to native galax
types; add register_parameterization`

Branch: `native-param-completeness-validation`

Issue: #44, `Extend potential-parameter completeness validation to native
galax types`

Base/head reviewed: `main` at `a1b3a27` / PR head at `599b0a4`

## Overall judgment

PR 52 addresses issue 44 correctly for all currently implemented potential
types and parameterizations. No critical or high-severity defects were found.
One focused API correction is recommended before merging, followed by two
small cleanups.

The pull request therefore **needs focused corrections**, not architectural
revision. Once those corrections are made and checked, it should be
squash-merged.

## Architectural summary

The pull request makes three coherent changes:

- Preparation-time validation now requires every recognized potential
  component to declare exactly the parameter names expected by its type and
  parameterization.
- `parameter_schema_is_known()` distinguishes an authoritative empty schema
  from an unknown type or unsupported parameterization, preventing misleading
  "all parameters are extra" errors.
- `ParameterizationSpec` consolidates forward conversion, inverse conversion,
  and raw parameter dimensions into one registration. This removes the
  previously duplicated parameterization registries.

The unit validator uses the same schema-recognition predicate. Unknown types
and unsupported parameterizations can therefore proceed through preparation
and receive the intended runtime resolution error instead of a misleading unit
error.

The intentional policy that even `galax` constructor parameters with defaults
must be declared explicitly is consistently implemented, tested, and
documented. This is a sound policy for reproducible configurations and stable
`AllModels` columns.

## Findings

### Critical

None.

### High

None.

### Medium: the new registration API accepts combinations the runtime cannot round-trip

References:

- `tnt/potential/registry.py:245-268`
- `tnt/potential/components.py:258-286`
- `tnt/potential/components.py:337-351`

`register_parameterization()` accepts a parameterization for any `type_name`,
and `parameter_schema_is_known()` then considers that pair authoritative.

Forward construction is generic: `ResolvedPotentialComponent` applies its
registered converter regardless of whether the target is a curated native
`galax` type or a TNT MGE composite. Inverse conversion is not generic,
however. Only `GalaxPotentialComponent.raw_parameters()` looks up and invokes
the registered inverse converter. TNT composite components inherit
`AbstractPotentialComponent.raw_parameters()`, whose implementation returns
the canonical parameters unchanged.

Consequently, registering a non-native parameterization for a TNT composite
component appears supported by registration, configuration validation, and
forward construction, but `AllModels` would receive canonical parameter names
and values instead of those declared by the configuration.

The currently registered NFW `concentration_m200` path is unaffected. The
problem is the contract exposed to future parameterizations.

Recommended resolution: make an explicit design choice and enforce it.

1. If parameterizations are intentionally limited to curated native `galax`
   types, have `register_parameterization()` reject a `type_name` outside
   `_SUPPORTED_GALAX_TYPES`. This is the smaller correction and matches the
   present human documentation.
2. If parameterizations should support TNT composite types, move inverse
   dispatch into a type-independent layer and add a composite-component
   forward/inverse round-trip test.

The first option is recommended for this pull request.

### Low: the registration test prevents adding future parameterizations

Reference: `tests/unit_tests/test_potential.py:788-822`

`test_register_parameterization_bundles_converters_and_schema()` ends by
asserting that the complete registry contains exactly the one current NFW
entry. Adding a legitimate second parameterization would therefore break this
unrelated duplicate-registration regression.

The test should snapshot the registry before attempting duplicate registration
and assert that it remains unchanged afterward. Ideally, it should also use
`monkeypatch` to register a temporary unique specification and directly test
the successful registration path.

### Low: the NFW documentation still writes H0

Reference: `docs/source/potential.md:166-171`

The critical-density equation uses $H_0$ immediately after the documentation
correctly explains that TNT's configured `H` describes the Hubble parameter at
the halo's epoch and need not be the present-day value. The equation should use
$H$.

This wording was inherited from `main`, rather than introduced by PR 52, but
the pull request already edits the surrounding section.

## Test coverage assessment

Coverage of issue 44 itself is strong:

- valid, missing, and extra native parameters;
- parameters with `galax` constructor defaults;
- missing and extra `concentration_m200` parameters;
- parameterization-specific rather than native schema selection;
- unknown-type and unsupported-parameterization preparation behavior;
- unit-validation deferral;
- converter/schema registry consolidation; and
- existing NFW forward, inverse, rescaling, and model-search integration
  coverage.

Recommended additions or adjustments:

- Test the chosen registration scope: reject TNT composite/unknown target
  types, or prove their complete forward/inverse round trip.
- Avoid asserting that NFW is permanently the registry's only entry.
- A fully end-to-end assertion that an unknown type and unsupported
  parameterization eventually produce the intended runtime messages would be
  useful, although existing component-resolution tests already cover those
  messages independently.

## Documentation assessment

The new exact-parameter-set policy and the treatment of constructor defaults
are documented consistently in `docs/source/configuration.md`,
`docs/source/potential.md`, and `aidocs/KNOWLEDGE.md`.

The only identified contradiction is the use of $H_0$ rather than configured
$H$ in the NFW critical-density equation described above.

## Related work and merge order

- PR 52 is based on the merged PR 51 and has current `main` as its merge base.
  No content conflict was found.
- Issue #46 remains the appropriate follow-up for generalizing component
  registries across TNT. PR 52 need not absorb that work.
- Issue #30's physical parameter-domain validation remains separate. PR 52
  provides the exact parameter-name schema on which that validation can build.
- PR #33 is closed and is no longer a merge-order dependency.
- GitHub reported PR 52 as technically mergeable but `BLOCKED`, with no review
  approval and no configured status checks shown at audit time.

Because the branch history contains the original PR 51 commit and a merge from
the squash-merged `main`, squash merge is recommended for PR 52.

## Checks run

Read-only checks were run from the PR head using the Linux development
container under Colima:

- full test suite: **342 passed**, with three existing dependency/process
  warnings;
- Ruff: passed;
- strict Sphinx build with `-W`: passed; and
- `git diff --check origin/main...HEAD`: passed.

One focused run initially produced 134 passes and a timeout in the existing
JAX precision subprocess test. The isolated rerun passed in 29.42 seconds
against its 30-second timeout, and the subsequent full suite also passed it.
This indicates an existing timing-margin weakness rather than a PR 52
regression.

No native Apple Silicon test run was performed. No GitHub CI results were
reported for the pull request at audit time.

## Recommended merge path

1. Decide whether registered parameterizations are restricted to curated
   native `galax` types or supported generically for TNT composite types.
2. Implement and test that decision. The smaller recommended change is to
   reject non-`galax` registration targets.
3. Make the registry test independent of the total number of registered
   parameterizations.
4. Replace $H_0$ with configured $H$ in the NFW critical-density equation.
5. Rerun the focused tests, full suite, Ruff, and strict Sphinx build.
6. Obtain the required GitHub review approval and squash-merge PR 52.

## Decision required

The only material design question is whether `register_parameterization()` is
intended exclusively for TNT's curated native `galax` components, as current
documentation and inverse conversion imply, or whether it is intended to
support TNT composite components as well. The implementation should enforce
whichever contract is chosen rather than accepting an only partially supported
combination.
