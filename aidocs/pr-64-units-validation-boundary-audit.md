# PR #64 audit: Move configuration quantity validation to the configuration layer

Date: 2026-09-10

Pull request: #64, `codex/units-validation-boundary` -> `main`

Audited head: `fce480c88b9058c16cb748295145035fc78337ee`

Related issue: #56

## Overall judgment

**Fit to merge; three minor follow-ups, none blocking.**

This is a narrow, correct dependency-boundary cleanup. `validate_configuration_quantities`
and its three private helpers move verbatim from `tnt.units` to
`tnt.configuration.validation`; `_validated_declared_unit` is promoted to the
public `validate_declared_unit` because it is now consumed across a package
boundary; the four affected tests move with the function and one fresh-process
import-order regression is added. Behaviour is unchanged -- every assertion in
the moved tests still holds and the full suite passes (398).

The move is a genuine improvement, not just relocation: `tnt.units` loses its
lazy `tnt.potential.registry` import and no longer imports any runtime-family
module, and the function lands in a module that already imports
`parameter_schema_is_known` / `raw_parameter_dimensions` at module level (used
since #52 in `_validate_potential`), so the same registry dependency now
covers both uses. The ordering dependency between `_validate_potential` and
`validate_configuration_quantities` (specific "ml invalid for a mass MGE"
message must precede the generic dimension check) becomes an intra-module
concern and is easier to keep correct.

The doc commit (`fce480c`) trims resolved-issue and forward-looking
references from `KNOWLEDGE.md`, `docs/source/units.md`, and
`docs/source/potential.md`. Consistent with describing current behaviour
directly.

## Verification performed (local, macOS)

```text
uv run pytest -q            -> 398 passed, 1 dependency warning
uv run ruff check .         -> All checks passed!
uv run sphinx-build -E -b html -W docs/source ...  -> build succeeded
```

Matches the PR's reported Linux-container run. Confirmed separately:

- no remaining references to `_validated_declared_unit` or
  `tnt.units.validate_configuration_quantities` anywhere;
- `tnt/units.py` imports nothing from `tnt.potential` / `tnt.mge` /
  `tnt.kinematics` / `tnt.spatial_binnings`;
- `_optional_mapping` / `_validate_field` fully removed from `tnt.units`;
- both import orders in the new subprocess test pass.

## Findings

### Critical / High

None.

### Medium

#### M1. The "silently skip unknown potential types" intent is now undocumented

`_potential_parameter_dimensions` returning `None` for an unrecognized `type`
or unimplemented `parameterization` is deliberate: it leaves the
"unsupported type" error to `AbstractPotentialComponent.resolve()` instead of
emitting a misleading "unit not supported for this parameter". The pre-move
docstring said so; the new one-line docstring does not, and
`test_parameter_unit_check_defers_for_an_unrecognized_potential_type` also
lost its explanatory comment. Nothing is broken, but the next reader sees a
silent skip with no stated reason.

Action: restore one line of rationale -- either in the docstring or as a
comment at the `if parameters is not None and dimensions is not None:` call
site -- and a one-line comment back on the "defers" test.

References:

- `tnt/configuration/validation.py:200-213` (`_potential_parameter_dimensions`)
- `tnt/configuration/validation.py:165-172` (the call site)
- `tests/unit_tests/test_configuration.py:79`

### Low

#### L1. `validate_declared_unit`'s docstring does not distinguish it from `validate_dimension`

Both are now public peers in `tnt.units`. `validate_dimension` takes a unit
object and returns `None`; `validate_declared_unit` takes a string, parses
it, and returns the parsed unit. One sentence noting the difference would
save a reader from guessing which to call.

Reference: `tnt/units.py:226`

#### L2. The import regression could lock the invariant it targets more directly

`test_configuration_and_potential_import_orders_are_acyclic` checks that the
two packages import in either order. The invariant this PR actually
establishes is stronger: `tnt.units` is a low-level primitive. A case that
runs `import tnt.units` in a fresh process and asserts `tnt.potential` and
`tnt.mge` are absent from `sys.modules` would fail loudly if a future edit
reintroduces the coupling.

Reference: `tests/unit_tests/test_configuration.py:17-32`

### Observation (not a finding)

The doc commit removes forward references to the planned "prior" mechanism
from `units.md` / `potential.md` / `KNOWLEDGE.md` (it exists on the unmerged
`prior-concept` branch). Correct for current-state main docs; it does mean
the docs no longer signal that a concrete cross-component mechanism is
prototyped. Re-add when `prior-concept` lands if desired.

## Response (2026-09-10)

M1 and L1 addressed on the branch.

- **M1** -- `_potential_parameter_dimensions`'s docstring again states that
  `None` covers a malformed / unrecognized `type` / unimplemented
  `parameterization`, and that the caller skips unit validation so the error
  falls to `AbstractPotentialComponent.resolve` instead of a misleading
  "unit not supported". The explanatory comment is back on
  `test_parameter_unit_check_defers_for_an_unrecognized_potential_type`.
- **L1** -- `validate_declared_unit`'s docstring now names it the string-input
  peer of `validate_dimension` and says which to use where.

L2 (a fresh-process `import tnt.units` isolation assertion) and the
prior-mechanism doc observation are left for the PR author's discretion; both
are optional.

Re-verified locally (macOS): `pytest -q` 398 passed, `ruff check .` clean,
strict `sphinx-build` succeeded.
