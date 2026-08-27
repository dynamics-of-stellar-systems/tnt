# PR #45 audit: Consolidate potential-component type registration

Audited PR #45 at head commit `971a2a0` against its actual `main` base and
issue #39. This was a read-only code audit: no implementation changes or
GitHub review comments were made.

## Overall judgment

PR #45 is **not yet ready to merge as the resolution of issue #39**.

The implementation is clean, tested, and improves the current code, but it
does not deliver issue #39's central requirement: one authoritative source of
component-owned metadata. A focused architectural correction within this PR
should be sufficient; this does not require a wholesale rewrite.

No critical findings were identified. There is one high-, one medium-, and
one low-priority finding.

## Findings

### High — Runtime and validation still have separate registries

Runtime dispatch discovers `_type` values from subclasses in
`tnt/potential/components.py:197`, while configuration validation and
parameter dimensions use `_MGE_RAW_DIMENSIONS` and `MGE_POTENTIAL_TYPES` from
`tnt/potential/registry.py:155`.

The new agreement test at `tests/unit_tests/test_potential.py:759` explicitly
describes these as two independently computed sources. That test catches
repository drift, but it does not create a single authority. Adding a TNT
component still requires coordinated changes to:

- the class `_type`;
- `_MGE_RAW_DIMENSIONS`;
- the implementation-module import; and
- potentially validation behavior.

If one is missed, runtime resolution, configuration validation, and unit
handling can disagree. This is exactly the condition issue #39 was intended
to eliminate before PR #33.

Recommendation: place the type identifier, raw parameter dimensions, and
schema metadata in one component specification consumed by runtime dispatch,
configuration validation, and dimension lookup. An explicit authoritative
registry is also acceptable if class-owned metadata would introduce import
cycles.

### Medium — Inherited `_type` values produce false duplicate errors

Recursive discovery uses `getattr(subclass, "_type", None)` at
`tnt/potential/components.py:127`. A subclass that merely inherits its
parent's `_type` is therefore treated as a second registration.

This was reproduced under Linux: a registered class plus an ordinary derived
class caused:

```text
ValueError: Duplicate potential type 'registered' on Registered and Derived.
```

A harmless implementation or testing subclass can consequently prevent all
subsequent component resolution.

If subclass discovery is retained, inspect only identifiers declared directly
on the class, for example through `subclass.__dict__`, and add an inherited-
identifier regression test. An explicit registry would avoid this failure
mode entirely.

### Low — Documentation is incomplete

`tnt/potential/components.py:73` still says exact parameter-schema validation
is not implemented. PR #45 now implements exact schema validation for TNT MGE
components, while issue #44 covers the remaining native `galax` case. The
wording should distinguish those states.

Because this changes an architectural registry boundary,
`aidocs/KNOWLEDGE.md` should also record the final authoritative design and
the invariant that runtime resolution, configuration validation, and
parameter-dimension lookup share.

## What is good

- Recursive traversal is otherwise straightforward.
- Duplicate and malformed identifier errors are clear.
- MGE validation is substantially simpler.
- Required and forbidden MGE fields now produce consistent errors.
- Issues #43 and #44 are reasonable separate follow-ups.
- The separately curated native `galax` and non-native parameterization
  registries remain intact.

## Verification

Run from an isolated PR snapshot under Linux/Colima:

- complete test suite: **306 passed**;
- Ruff over `tnt` and `tests`: **passed**; and
- targeted inherited-identifier probe: **failed as described above**.

At audit time GitHub reported that the PR was mechanically mergeable and
blocked only by required review. No CI checks, reviews, or inline review
comments had been posted.

## Active-work interaction

PR #45 is one `main` commit behind, although GitHub can currently merge it
automatically.

It overlaps PR #42 in configuration validation and two test modules. A
synthetic merge showed small textual conflicts, principally imports and
adjacent additions, but no apparent semantic conflict. Since PR #42 is
complete and independent, merge PR #42 first, then merge current `main` into
the shared PR #45 branch while addressing these findings.

After PR #45 is corrected and merged, PR #33 should be updated on top of that
result and add its axisymmetric classes through the final authoritative
registration mechanism.

## Recommended path

1. Decide on class-owned specifications or one explicit component registry.
2. Make runtime dispatch, configuration validation, and dimension lookup
   consume it directly.
3. Remove the independently maintained MGE type-name set.
4. Fix or eliminate inherited `_type` discovery.
5. Update the component docstring and `aidocs/KNOWLEDGE.md`.
6. Add the inherited-identifier regression test.
7. Rerun the full suite, then merge PR #45 before finalizing PR #33.

If the team intentionally prefers two registries guarded by a consistency
test, PR #45 could be merged as an incremental improvement, but it should not
close issue #39 and would not yet provide the foundation issue #39 requires
before PR #33.
