# PR #74 audit: Add traced validity and construction for native potentials

Branch: `codex/traceable-potential-validation` → `main`. Author: Thomas Maindl
(assisted by Codex). First implementation step for issue #72.

## Summary of the change

Native `galax` potential construction currently raises Python exceptions on
invalid numerical proposals, which is incompatible with `jax.jit`/`jax.vmap`
tracing. This PR adds a parallel, traceable path:

- `ParameterConstraint.valid()`: JAX scalar predicate version of the existing
  eager `violation()`, sharing one `_checks()` helper so both evaluate the
  identical predicates at the proposal's own floating-point precision.
- `ResolvedPotentialComponent._raw_parameters_valid()` /
  `build_with_validity()`: static structure (names/types/dimensions/shapes)
  still raises eagerly; numeric finiteness and registered bounds become a
  JAX boolean instead of a raise. Explicitly out of scope (raises
  `NotImplementedError`, eagerly, before tracing): components with a
  registered `parameterization` converter, and non-native (MGE) components.
- `Potential.build_with_validity()`: composes each component's flag into one
  combined validity boolean alongside the built `Potential`.
- New `ParameterConstraint.minimum_separation_eps_power`: a relative
  separation guard (`(a - b) > eps**power * max(|a|, |b|)`), applied to
  `StoneOstriker15Potential`'s `r_h > r_c` constraint, since that potential's
  formula subtracts near-equal terms and loses gradient accuracy close to
  `r_h == r_c`.

## Verification performed

- Checked out `codex/traceable-potential-validation`, ran the full suite:
  `551 passed` (unit + integration), matching the PR's own claim of 178
  passing tests in `test_potential.py` specifically (`178 tests collected`,
  confirmed exactly).
- `ruff check` on the four changed source files: clean.
- Traced the constraint-checking call graph by hand: `build()` (eager) calls
  `_check_parameter_set_contract` (raises on non-finite) before
  `_validate_parameter_constraints` (raises via `violation()`), so
  `violation()`'s new finiteness branch is provably unreachable from
  `build()` -- see Findings.
- Confirmed `build_with_validity` has no production caller yet (`grep` turns
  up nothing outside `core.py`/`components.py`) -- this is genuinely
  scaffolding for #72's next step, exactly as the PR description says.
- Spot-checked the numerics: `_raw_parameters_valid` iterates
  `self.raw_constraints`, the same set `build()`'s eager path validates via
  `self.raw_constraints` -- no coverage gap between the eager and traced
  constraint sets for the components this path supports.
- The Stone-Ostriker precision-margin tests carry an independently computed
  90-digit-precision reference gradient and test the exact float32/float64
  boundary the epsilon-power guard is meant to draw -- strong verification
  of that specific piece, not just a smoke test.

## Findings

**Minor, no fix needed:** `ParameterConstraint.violation()`'s new
`if not bool(checks["finite"])` branch (`registry.py`) is dead code in the
only production call path (`_validate_parameter_constraints`, always called
after `_check_parameter_set_contract`, which already raises on non-finite
values first). It's reachable only from a test that calls `violation()`
directly. Harmless -- not a correctness bug, just a branch with no live
caller -- and cheap to keep as a public-method safety net given `valid()`
and `violation()` intentionally share `_checks()`. Not worth blocking on.

No other issues found. Design is well-scoped (raises immediately, eagerly,
for anything outside "native, no converter" rather than silently mishandling
it under trace), the eager/traced predicate sharing avoids the two paths
drifting apart, and the new precision guard is independently justified and
tested rather than just asserted.

## Recommendation

Ready to merge.
