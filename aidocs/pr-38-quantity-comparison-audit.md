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

One finding, resolved into a scoped-out follow-up rather than a fix for
this PR: the new pairwise comparison still manually reimplements unit
conversion instead of using `unxt.Quantity`'s own equality operator. That
initially read like an obvious simplification -- but applying it breaks two
of this PR's own tests, because `unxt.Quantity` is JAX-backed and silently
downcasts to `float32` by default, and this PR deliberately tests for
precision well below what `float32` can represent ("exact comparison
semantics" is named explicitly in the commit message). The manual
`astropy`/`float64` conversion currently in `tnt/configuration/compatibility.py`
is therefore *necessary*, not leftover complexity -- see the finding below
for the full story and the proposed resolution (a project-wide
`jax_enable_x64` config default, tracked as separate follow-up work).

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

### Info: `_quantity_declarations_equal` could use `unxt.Quantity` directly -- blocked on a real precision gap, not just style

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

`left_unit`/`right_unit` are plain `astropy.units` objects, and the
comparison manually converts one side via `Unit.to(...)` before comparing
raw arrays. `unxt.Quantity` already compares across compatible units
directly (`Q(1,"kpc") == Q(1000,"pc")` is `True`), which reads like an
obvious simplification -- and initially I proposed exactly that, with the
existing `is_equivalent` guard kept (since `Quantity.__eq__` raises for
incompatible dimensions rather than returning `False`, and broadcasting can
mask a genuine shape mismatch -- e.g. `[5.0] kpc` vs `[5000.0, 5000.0] pc`
broadcasts to `[True, True]` even though the declarations are different
shapes; both guards are load-bearing, not incidental).

**Applying it broke two existing tests.** `unxt.Quantity` is JAX-backed and
silently downcasts to `float32` by default (confirmed:
`u.Quantity(np.array([2000.0000000001]), "pc").value` prints `[2000.]`,
`dtype=float32`). That's exactly what
`test_quantity_comparison_handles_nested_arrays_and_exact_values` checks
for -- it perturbs a value by `1e-10` (roughly six orders of magnitude below
float32's ~7-digit precision) and asserts that still registers as a
difference, and the commit message names "exact comparison semantics" as a
deliberate goal. So this isn't an oversight in PR #38; the manual
`astropy`/`float64` conversion is currently *necessary* to deliver what the
PR itself already tests for, not just historical leftover. Applied the
`unxt.Quantity` change locally, confirmed it broke
`test_quantity_comparison_handles_nested_arrays_and_exact_values` and
`test_equivalent_declared_units_are_compatible`, and reverted it before
committing -- `_quantity_declarations_equal` in the current code is
unchanged from what PR #38 shipped.

**Proposed resolution (Prash's, discussed 2026-08-27), out of scope for
PR #38 itself:** add a config-level `jax_enable_x64` switch, defaulting to
`True`, applied once at process start (before any JAX arrays exist -- almost
certainly in `configuration_session()`/`Configuration.read()`, needs its own
design pass on exactly where). With `x64` on by default, `unxt.Quantity`
carries full `float64` precision, so the direct-comparison simplification
above becomes correct as originally proposed, with no special-casing needed
-- `_quantity_declarations_equal` reduces to shape check + `is_equivalent`
guard + `bool(np.all(Quantity(...) == Quantity(...)))`. This also gives
TNT a project-wide precision policy it currently doesn't have at all (grepped
the whole codebase: nothing sets `jax_enable_x64` anywhere today, so
`float32` is the live default everywhere, silently), which matters well
beyond this one comparison -- e.g. it's plausibly relevant to the deferred
GPU-migration work, where `float32` vs `float64` is a real throughput
tradeoff worth making an explicit, visible choice about rather than
inheriting JAX's default.

This is a genuinely separate follow-up, not a PR #38 fix -- it needs: (a) a
new config setting and where it plugs into process startup, (b) confirming
nothing already implicitly depends on today's silent `float32` default
elsewhere in the codebase, (c) placing it under `numerics_settings`, whose
own comment already describes exactly this category ("Shared numerical
policies. Keeping these values explicit makes model identity checks...
reproducible across implementations") and which is already part of
`_critical_configuration`'s compared fields (the code already comments "the
current compatibility contract rejects every numerics change") -- so
putting `jax_enable_x64` there gets resume-compatibility enforcement for
free, with no further schema or manifest work needed. And (d) then the
one-line simplification here. Recommend opening a new issue for it
(matching how #36 was itself split out of #27) rather than folding it into
this PR.

Actionable location, once the above lands: `tnt/configuration/compatibility.py`,
`_quantity_declarations_equal`.

## Missing or weak tests

None for PR #38 as shipped -- the malformed/shape/dimension test matrix is
thorough, and (usefully) is exactly what caught the precision regression
above during review. The `jax_enable_x64` follow-up, once scoped, will need
its own coverage: the config default itself, that it's applied before any
JAX array is created, and that a resumed run with a different
`jax_enable_x64` than its baseline is actually rejected by
`ensure_resume_compatible`.

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

Nothing to change in PR #38 itself -- it's correct and consistent with its
own tests as shipped. For the `jax_enable_x64` follow-up:

1. Open a new issue capturing it (matching how #36 was split out of #27),
   covering: the new config setting and default under `numerics_settings`,
   where it applies during process startup, and its relevance to the
   deferred GPU-migration work.
2. Once landed, apply the one-line simplification in
   `_quantity_declarations_equal` (shape check + `is_equivalent` guard +
   `bool(np.all(Quantity(...) == Quantity(...)))`), and confirm the existing
   test matrix (in particular the `1e-10`-perturbation exact-comparison
   test) still passes under the new default.
