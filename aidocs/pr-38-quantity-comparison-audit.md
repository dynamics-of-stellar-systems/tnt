# PR #38 audit: compare configuration quantities on demand

Audit date: 2026-08-27

Pull request: #38, `codex/issue-36-quantity-comparison` into `main`

## Overall judgment

PR #38 is a correct, well-scoped, well-tested implementation of issue #36's
proposal: it removes the eager whole-configuration unit normalization
(`normalize_configuration_quantities` and its helpers, 180 lines gone from
`tnt/units.py`) and replaces it with on-demand, unit-aware comparison of just
the `{value, unit}` leaves `_different_paths` actually reaches while diffing
two critical configurations. I did not find a functional bug: no stale
reference anywhere to the removed normalization functions,
`normalize_unitful_value` (a different, still-needed helper used by
`tnt.kinematics`) was correctly left alone, and the new
`_looks_like_quantity_declaration` heuristic doesn't false-positive against
anything else currently in the critical-configuration projection (checked
`orbit_library_settings`, `weight_solver_settings`, `MGEs`,
`kinematic_data.*.histogram`/`systematic_uncertainties`, `units.internal`
by hand).

One real finding: the new pairwise comparison itself doesn't go far enough.
It still manually reimplements unit conversion (`left_unit.to(right_unit,
left_values)`) instead of using `unxt.Quantity`'s own equality operator,
which already does exactly this. That's not just a style nit -- it's the
same category of unnecessary-conversion-machinery the PR's own rationale
(quoting issue #36) argues against, just relocated to a smaller scope rather
than eliminated.

## Architectural summary

- `tnt/units.py` loses `normalize_configuration_quantities`,
  `normalize_potential_settings`, `_normalize_kinematics`,
  `_normalize_parameters`, `_normalize_field`, `_scale_field` -- the eager
  traversal that used to rewrite every quantity in a resolved configuration
  into the internal unit system before `_critical_configuration` compared
  two configs structurally.
- `tnt/configuration/compatibility.py::_critical_configuration` now projects
  the *raw* declared configuration (unconverted, in whatever units it was
  written in) onto the critical fields -- no normalization step at all.
- The actual physical-equivalence check moved into `_different_paths`
  itself: `_looks_like_quantity_declaration` recognizes an atomic
  `{value, unit}`-shaped mapping during recursion, and
  `_quantity_declarations_equal`/`_parse_quantity_declaration` convert and
  compare just that one pair on demand -- scalars and nested arrays both,
  via `numpy`, with explicit shape/dimension/malformed-declaration handling
  (`ConfigurationCompatibilityError` for anything not cleanly `{value, unit}`
  with a finite numeric value and a real unit string).
- `_NON_SCHEMA_PARAMETER_KEYS` (was `_SEARCH_PARAMETER_KEYS`) gained `"unit"`
  alongside the existing `fixed`/`generator_settings`/`latex_label`/`value`
  exclusions. This is load-bearing, not cosmetic: potential-parameter
  declarations are compared via the schema-exclusion path, not the new
  quantity-comparison path, so without excluding `unit` too, two runs
  declaring the same parameter in different-but-equivalent units (`10 kpc`
  vs `10000 pc`) would now be flagged as an incompatible schema change --
  a regression the old eager-normalization design didn't have, since by the
  time `_potential_schema` ran every parameter's unit had already been
  forced into the same internal one. Confirmed correct via
  `test_critical_projection_preserves_declarations_but_excludes_parameter_units`.
- Docs (`docs/source/units.md`, `docs/source/model_search.md`,
  `aidocs/KNOWLEDGE.md`) updated to match, and read accurately against the
  actual code.

## Findings

### Low: `_quantity_declarations_equal` reimplements what `unxt.Quantity` already does

```python
def _quantity_declarations_equal(left: Any, right: Any, path: str) -> bool:
    left_values, left_unit = _parse_quantity_declaration(left, path)
    right_values, right_unit = _parse_quantity_declaration(right, path)
    if left_values.shape != right_values.shape:
        return False
    if not left_unit.is_equivalent(right_unit):
        return False
    converted = np.asarray(left_unit.to(right_unit, left_values), dtype=float)
    return bool(np.array_equal(converted, right_values))
```

`left_unit`/`right_unit` here are plain `astropy.units` objects (from
`unxt.unit(...)`), and the comparison manually converts one side via
`Unit.to(...)` before comparing raw arrays. But `unxt.Quantity` already
compares across compatible units directly -- confirmed empirically:

```python
>>> u.Quantity(1.0, "kpc") == u.Quantity(1000.0, "pc")
True  # (elementwise for arrays)
```

The only behavior `unxt.Quantity.__eq__` doesn't give you directly: it
*raises* for incompatible dimensions rather than returning `False`
(confirmed: `Q(1,"kpc") == Q(1,"Myr")` raises `EquinoxTracetimeError`, not a
`False`/`array([False])`). The existing `left_unit.is_equivalent(right_unit)`
guard is exactly what's needed to preserve today's "return False, don't
raise" behavior for a dimension mismatch, so it stays. With that kept, the
fix is a one-line replacement of the manual conversion:

```python
if not left_unit.is_equivalent(right_unit):
    return False
return bool(np.all(u.Quantity(left_values, left_unit) == u.Quantity(right_values, right_unit)))
```

(`u` here would need to resolve to `unxt`, already imported as `import unxt
as u` in this module -- name collision with the existing `u.unit(...)` calls
elsewhere in the file is not an issue since both live under the same `unxt`
namespace.)

This doesn't change behavior -- the existing tests
(`test_quantity_comparison_handles_nested_arrays_and_exact_values`,
`_reports_shape_and_dimension_changes`, `_rejects_malformed_declarations`)
already pin the exact comparison semantics (shape must match, dimensions
must match, exact value equality, no epsilon) and would catch a regression
from this refactor without needing new tests. It's purely: stop hand-rolling
unit conversion when the type already reached for does it, which is the
more complete version of the PR's own stated principle.

Actionable location: `tnt/configuration/compatibility.py`,
`_quantity_declarations_equal`.

## Missing or weak tests

None found beyond what the finding above already covers -- the malformed/
shape/dimension test matrix is thorough and would double as regression
coverage for the suggested fix. No new tests are strictly required to make
that change safely.

## Checks run

All run directly on this branch, locally:

- Full `pytest`: 289 passed.
- `ruff check .`: passed.
- Sphinx build with warnings treated as errors: passed.
- Manual trace of every reference to the removed
  `normalize_configuration_quantities`/`normalize_potential_settings`/
  `_normalize_*`/`_scale_field` across the whole repository: none found
  (clean removal).
- Manual check that `normalize_unitful_value` (a different function, still
  used by `tnt.kinematics`) was correctly left in place.
- Manual trace of `_looks_like_quantity_declaration` against every section
  `_critical_configuration` actually includes, looking for a legitimate
  non-quantity `{value: ...}`-shaped mapping that could false-positive into
  the quantity-comparison path: none found.
- Empirically verified `unxt.Quantity`'s cross-unit `==` behavior (equal
  values, raising on incompatible dimensions) directly in a Python shell.

## Recommended review and correction sequence

1. Apply the one-line simplification in `_quantity_declarations_equal`
   above.
2. Optional: a short comment on the `is_equivalent` guard noting *why* it's
   still needed (Quantity equality raises rather than returning `False` for
   incompatible dimensions) so a future reader doesn't mistake it for dead
   code once the manual conversion is gone.
3. Rerun the existing test suite -- no new tests needed, per "Missing or
   weak tests" above.
