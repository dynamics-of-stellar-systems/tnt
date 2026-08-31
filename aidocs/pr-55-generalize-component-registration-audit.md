# PR 55 audit: generalize component-type registration across TNT

Pull request: #55, `Generalize component-type registration across TNT`

Branch: `codex/issue-46-component-registration`

Issue: #46, `Generalize component-type registration across TNT`

Base/head reviewed: `main` at `1dfd01b` / PR head at `32c2127`

## Overall judgment

PR 55 does the substantive work issue #46 asks for -- one shared
`register_typed_class` helper, explicit registration for the potential,
kinematics, and parameter-generator families, and the follow-up filed on #46
(configuration validation reads potential type predicates through public
accessors rather than the private `_COMPONENT_REGISTRY` dict). No critical or
high-severity defect.

One **Medium** finding that **blocks merge**: PR 55 registers the
parameter-generator family but stops short of removing the hand-maintained
`_GENERATOR_SETTINGS_KEYS` schema list that registration makes redundant --
the exact "second parallel list" issue #46 exists to eliminate, and #46's
own step 3 ("revisit `tnt.parameter_generator`'s hand-written tuple once the
shared utility exists") applies to it. Plus three Low notes. None is a
correctness bug.

## Architectural summary

- `tnt/registry.py` (new) holds one family-agnostic helper,
  `register_typed_class(registry, cls, *, family)`. It reads
  `cls.__dict__.get("_type")` -- the class's *own* attribute, not an
  inherited one -- requires a non-empty string, rejects a duplicate already
  in the passed-in registry, and returns `cls` so it can back a decorator.
  It bakes in nothing family-specific: each family owns its registry dict,
  its decorator, and its lookup helpers.
- `tnt.potential.registry.register_component` now delegates to that helper.
  New public accessors `get_component_class`, `component_type_names`, and
  `is_registered_component_type` replace direct reads of
  `_COMPONENT_REGISTRY` in `components.py` and `configuration/validation.py`.
- `tnt/kinematics/registry.py` (new) replaces
  `tnt.kinematics._kinematics_class_registry()` (an `AbstractKinematics.
  __subclasses__()` walk with the same inherited-`_type` false-duplicate
  vulnerability potential's dispatch had). The three concrete kinematics
  modules gain `@register_kinematics`; `build_kinematics` and
  `_validate_kinematics` use `get_kinematics_class` / `kinematics_type_names`.
  `configuration/validation.py` drops its hand-maintained
  `_KINEMATICS_TYPES` set.
- `tnt.parameter_generator` replaces the linearly-scanned
  `_GENERATOR_CLASSES` tuple (no duplicate detection at all) with
  `register_parameter_generator` + registry dispatch, and
  `build_parameter_generator` now returns a proper "expected one of" error.
- `weight_solver` / `orbit_library` untouched, as the issue's non-goals
  require.

`cls.__dict__.get("_type")` is stricter than the previous `cls._type`: a
class decorated without declaring its own `_type` now raises `TypeError`
rather than registering under an inherited name. Every real TNT class in the
three families declares its own `_type` (verified), so nothing breaks.

No import cycle is introduced.

## Findings

### Critical / High

None.

### Medium (blocks merge): `_GENERATOR_SETTINGS_KEYS` is a redundant hand-maintained list

References:

- `tnt/configuration/validation.py:635` (`_GENERATOR_SETTINGS_KEYS` and its
  comment)
- `tests/unit_tests/test_parameter_generator.py:20`
  (`test_generator_settings_keys_match_the_real_classes`)

`configuration/validation.py` keeps `_GENERATOR_SETTINGS_KEYS` as a literal
dict, kept in step with each registered generator class's own
`_required_generator_settings` only by a regression test. PR 55 adds the
generator registry (`register_parameter_generator`,
`parameter_generator_type_names`, `get_parameter_generator_class`) but leaves
this parallel list in place, so the PR does the mechanism half of #46's
step 3 and not the removal half. That is the specific outcome #46 is meant to
prevent: one authoritative registration, no second list to keep in sync.

The comment justifies the hand-list as keeping `galax` out of a
"validation-only module". That justification does not hold. `tnt/__init__.py`
eagerly executes `from tnt.configuration import Configuration`, so importing
*any* `tnt.*` submodule -- `tnt.configuration.validation`, `tnt.validation`,
`tnt.units`, bare `tnt` -- already loads `galax`, `jax`, `equinox`, and the
full `tnt.potential`/`tnt.mge` stack. There is no import path by which config
preparation runs without `galax` already imported, so the hand-list buys no
isolation. (The same is true of the kinematics-registry import the PR adds to
`validation.py` -- also not galax-free; it is acceptable because
`tnt.kinematics` sits low in the graph and was already a legitimate
validation dependency, not because of the stated reason.)

**Required change:** delete `_GENERATOR_SETTINGS_KEYS` and its sync test;
derive the per-`generator_type` required-settings mapping from the registered
generator classes' `_required_generator_settings` at the point of use in
`_validate_parameter_space_settings` (via a small accessor on
`tnt.parameter_generator`, mirroring `parameter_generator_type_names`).
Importing `tnt.parameter_generator` into `validation.py` was checked and
introduces no cycle: nothing in its `tnt.all_models` / `tnt.potential`
dependency chain reaches back to `tnt.configuration`, and both import orders
were verified to load cleanly.

A genuinely galax-free config-preparation path (making `tnt/__init__.py`
lazy) is a real, separate improvement worth its own issue, but it is not a
precondition for this change and does not affect it either way.

### Low 1: removing `_KINEMATICS_TYPES` left a formatting regression

Reference: `tnt/configuration/validation.py:53-54`

Deleting `_KINEMATICS_TYPES` removed the blank lines that separated it from
`validate_resolved_configuration`, leaving the dict's closing `}` directly
above the `def`. `ruff check` passes but `ruff format --check` reports the
file as needing reformatting. One blank line pair restores it.

### Low 2: `test_inherited_type_does_not_register_or_raise` is now mis-named

Reference: `tests/unit_tests/test_potential.py:749`

The test still correctly checks that an *undecorated* child inheriting
`_type` does not enter the registry. But a *decorated* child now raises
`TypeError` via `register_typed_class`, so "does not ... raise" no longer
describes the whole behaviour. `tests/unit_tests/test_registry.py::
test_register_typed_class_rejects_an_inherited_type` covers the raise at the
shared-helper level, so coverage is complete; the name is stale, and a
potential-layer assertion that `register_component` on such a child raises
would be a small, worthwhile strengthening.

### Low 3: first use of PEP 695 type-parameter syntax

Reference: `tnt/registry.py:6`

`def register_typed_class[RegisteredType](...)` is (as far as this audit
found) the first PEP 695 generic in the codebase. Valid on TNT's 3.12 floor
and reads well; flagged only so the team is aware the syntax is now in use.

## Sequencing against issue #46

The issue asked for kinematics migration first and the shared helper
extracted *after* two working examples exist. PR 55 does both in one change,
plus the parameter-generator step. Doing all three at once is acceptable: the
issue's stated reason for staging was to avoid over-fitting the helper to
potential's quirks, and `register_typed_class` demonstrably carries none of
them (no `GalaxPotentialComponent` fallback, no `_raw_dimensions` handling --
both stay in `tnt.potential`). The helper is the genuine common denominator.

What is *not* acceptable is doing the parameter-generator step only halfway
-- adding the registry but keeping `_GENERATOR_SETTINGS_KEYS` -- since #46's
step 3 is explicitly about removing that hand-maintained data. See the
Medium finding.

## Test coverage assessment

- `tests/unit_tests/test_registry.py` (new): own-`_type` registration,
  inherited-`_type` rejection, duplicate rejection.
- kinematics registry test rewritten from "derived from `__subclasses__()`"
  to an explicit expected mapping.
- parameter-generator: registry contents, dispatch through the registry,
  unregistered-type error. The settings-keys sync test is repointed at the
  registry but should be deleted along with `_GENERATOR_SETTINGS_KEYS` (see
  the Medium finding), not kept.
- `is_registered_component_type` / `component_type_names` exercised
  indirectly through existing potential and configuration tests.

Adequate. Optional: a direct test that `_validate_kinematics` rejects an
unknown `type` with the registry-derived "expected one of" list, mirroring
the parameter-generator one.

## Documentation assessment

`aidocs/KNOWLEDGE.md` is updated thoroughly and accurately for the new
`tnt/registry.py`, the kinematics and parameter-generator registration, and
the "public predicate/accessor, not private dict" access pattern; several
stale claims about the "old `issubclass` check" are corrected in the same
pass. `docs/source/model_search.md` now correctly describes
`GridSearchParameterGenerator` as a registered scaffold whose proposal
algorithm is not implemented yet.

The `_GENERATOR_SETTINGS_KEYS` comment and the matching `KNOWLEDGE.md`
sentence both state the false "keeps galax out of validation" rationale.
Both go away with the Medium fix; if the hand-list somehow survived, both
would need correcting.

## Checks run

From the PR head on macOS (Apple Silicon):

- full suite: **350 passed**, one warning (the pre-existing dependency-owned
  TensorFlow Probability / JAX `pytype_aval_mappings` deprecation);
- `import tnt.configuration.validation` / `tnt.kinematics` /
  `tnt.parameter_generator`: no cycle in either direction;
- import-graph check: `import tnt.configuration.validation` loads `galax`,
  `jax`, `equinox`, `tnt.potential`, `tnt.mge` -- via `tnt/__init__.py`;
- `ruff check .`: passed;
- `ruff format --check`: flags `configuration/validation.py` (Low 1);
- strict `sphinx-build -E -b html -W`: passed;
- `git diff --check main...HEAD`: passed.

## Recommended merge path

1. **Required:** remove `_GENERATOR_SETTINGS_KEYS` and its sync test; derive
   the required-settings mapping from the registered generator classes (see
   the Medium finding). Update the `KNOWLEDGE.md` line that repeats the
   "keeps galax out" rationale.
2. Fix the `configuration/validation.py` formatting regression (Low 1) and,
   optionally, rename the stale test (Low 2).
3. Re-run Ruff (`check` and `format --check`), the full suite, and strict
   Sphinx.
4. Obtain the required GitHub review approval and squash-merge PR 55.
5. Remove this audit document in its own commit before merge.

## Decision required

None. The Medium fix is the removal described above; a lazy `tnt/__init__.py`
is a separate, optional future issue.
