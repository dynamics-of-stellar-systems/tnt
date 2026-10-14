# PR #84: mge-traceable-inverses audit

Date: 2026-10-14

PR: https://github.com/dynamics-of-stellar-systems/tnt/pull/84

Reviewed head: `43f7c9dc93878526ce41c11de32244a27926a22b`.
Reviewed main: `4bb8711150a3a6511b8edc1d1b0d9c6fccf226f0`.
The head includes that main commit; the review does not require reconciling
an older base. The checkout was clean before this audit.

## Recommendation

**Ready to merge.** The core implementation is sound: the new
`_with_validity` inverse methods correctly extend the established
eager/`with_validity`/`_candidate` pattern already used for the forward
conversions, are genuinely well tested (including the hard differentiability
properties under `vmap`), and fix a real, previously-unguarded bug along the
way (see Design assessment). One test introduced by this PR failed on a
re-run — confirmed to be a test-only defect (a `pytest.approx` tuple
comparison mixing raw JAX arrays with Python floats), not a production
regression; the underlying numerics agree to float64 roundoff. Fixed directly
(F1's own response, below) since the correction was a one-line test change
with no implication for the reviewed production code.

This PR implements item 1 of issue #82's prerequisites (traceable inverse
conversions); it does not itself extend default priors to the newly-traceable
parameterizations, matching its own stated scope.

## Design assessment

- The `_with_validity` siblings (`q_min_from_inclination_with_validity`,
  `triaxial_intrinsic_shape_with_validity`,
  `T_Tmaj_Tmin_from_viewing_angles_with_validity`) follow the exact
  eager-raises / `with_validity`-returns-a-flag / shared-`_candidate` pattern
  already established for the forward conversions elsewhere in `tnt/mge.py` —
  consistent, not a new convention.
- `_inverse_reporting_values` (the glue used by the registered `invert`
  converters) correctly branches on `isinstance(valid, jax.core.Tracer)`,
  matching the same traced-vs-concrete detection idiom `_guard_construction`
  already uses in this file: eager calls still raise `MGEDeprojectionError`
  (preserving the pre-PR reporting contract for `AllModels`), traced calls
  return `NaN` leaves via `jax.tree.map` instead.
- **Bonus fix, not just a refactor:** the pre-PR eager
  `triaxial_intrinsic_shape` had *no validity check at all* — it indexed the
  anchor's raw per-component ratios and returned them unconditionally,
  however degenerate. This PR adds the missing check
  (`_intrinsic_shape_candidate`'s own named predicate) and makes it raise
  `MGEDeprojectionError` instead of silently reporting garbage. Worth calling
  out explicitly since it reads as a refactor-only diff otherwise.
- `_intrinsic_shape_candidate`'s "compute validity from a
  `jax.lax.stop_gradient`-detached probe, then recompute the real output from
  a `jnp.where`-guarded safe substitute" construction was traced through by
  hand; it correctly keeps rejected batch lanes finite and zero-gradient
  without affecting accepted lanes, confirmed independently against
  `test_valid_inverse_anchor_is_independent_of_other_gaussians_and_normalization`
  and `test_triaxial_inverse_rejections_batch_with_zero_gradients`.
- Scope of the new triaxial inverse validity is deliberately narrower than
  full construction: it checks only the selected anchor Gaussian's geometry
  and covariance-inversion accuracy, not every Gaussian's deprojection. This
  is correctly documented in three places (the method docstrings,
  `docs/source/potential.md`, `aidocs/KNOWLEDGE.md`) and is the right scope
  for a *shape-recovery* check, not a construction-validity check — the two
  are intentionally different concerns, and `raw_potential_parameters`'s own
  docstring states the precondition ("use it only after successful
  construction") plainly.
- New test file (`test_mge_inverse_traceable.py`, 394 lines) is thorough, not
  just happy-path: analytic-value/derivative checks at both JAX precisions,
  mixed-batch rejection with explicit gradient-finiteness assertions, anchor
  selection/tie-breaking under tracing, a circular-anchor edge case (`(p, q,
  u)` well-defined but `(T, T_maj, T_min)` singular, since the latter divides
  by `1 - q^2`), forward/inverse round-trip identity derivatives, and static
  type-rejection for boolean/complex angle inputs.

## Merge findings

### F1 — P2: a regression test fails; the PR's own "528 tests passed" claim is inaccurate

Location: `tests/unit_tests/test_mge.py:800-809`
(`test_T_Tmaj_Tmin_and_viewing_angles_are_inverses`).

Confirmed by running the test directly (not just reading the PR description):

```
FAILED tests/unit_tests/test_mge.py::test_T_Tmaj_Tmin_and_viewing_angles_are_inverses
1 failed, 451 passed, 28 warnings in 292.91s
```

**Root cause, traced to source:** this test's own refactor (part of this PR)
replaced the old eager wrapper `_T_Tmaj_Tmin_from_p_q_u` (which cast its
result to Python floats) with the lower-level `_T_from_pqu_candidate` (which
returns raw JAX array elements, by design — it is the shared `_candidate`,
not a public eager API):

```python
coordinates, valid = _T_from_pqu_candidate(0.85, 0.60, 0.93, q_obs=0.76)
T, T_maj, T_min = coordinates          # 0-d jax arrays, not floats
...
T_r, T_maj_r, T_min_r = mge.T_Tmaj_Tmin_from_viewing_angles(theta, phi, psi)
assert (T_r, T_maj_r, T_min_r) == pytest.approx((T, T_maj, T_min), abs=1e-9)
```

`pytest.approx` mishandles a tuple mixing Python floats (`T_r` etc. — the
real `T_Tmaj_Tmin_from_viewing_angles` still casts to float) with raw 0-d JAX
arrays (`T` etc.). Verified directly, isolating each possible cause:

```python
>>> T_r == pytest.approx(T, abs=1e-9)              # single pair: True
>>> (T_r, T_maj_r, T_min_r) == pytest.approx((T, T_maj, T_min), abs=1e-9)   # tuple: False
>>> (T_r, T_maj_r, T_min_r) == pytest.approx((float(T), float(T_maj), float(T_min)), abs=1e-9)  # cast: True
```

The actual numerical discrepancy is `~3e-16` (float64 roundoff), seven orders
of magnitude inside the stated `abs=1e-9` tolerance — **not** a precision
regression in `tnt/mge.py`. This is isolated to this one test:
`test_T_Tmaj_Tmin_round_trips_through_p_q_u` (same file) hits the identical
raw-`_T_from_pqu_candidate` pattern but compares via `np.testing
.assert_allclose`, which has no such issue and passes;
`test_viewing_angles_from_T_Tmaj_Tmin_agrees_with_the_pqu_path` compares
single `.ustrip("rad")` scalars rather than a mixed tuple, and also passes.

**Required change:** cast `coordinates` to Python floats before the
`pytest.approx` tuple comparison (matching the pattern already used
elsewhere in this same file), e.g. `T, T_maj, T_min = (float(v) for v in
coordinates)`. Re-verify the PR's own validation claim after the fix —
either the author's test run didn't actually include this test, or ran under
conditions that happened not to trigger the `pytest.approx`/JAX-array
interaction; worth confirming which, since "528 tests passed" should mean
what it says.

**Response -- fixed.** `test_T_Tmaj_Tmin_and_viewing_angles_are_inverses` now
unpacks `coordinates` as `T, T_maj, T_min = (float(value) for value in
coordinates)`, exactly the suggested change, with a comment noting *why*
(the tuple-vs-single-pair `pytest.approx` discrepancy, not a floatness
requirement downstream -- `_T_from_pqu_candidate` stays the shared, uncast
candidate it's meant to be). Re-ran the full set of test modules this audit
covered: **452 passed, 0 failed** (up from 451 passed/1 failed). Ruff and a
strict Sphinx build still pass. This was a one-line test-only correction
with no implication for any reviewed production code, so it was made
directly rather than left for the PR's own author.

## Additional review notes

- The registered `invert` converters' eager failure path now raises a
  generic `"Cannot report an invalid inverse MGE shape."`
  (`_inverse_reporting_values`) instead of the original, more specific
  message the direct eager method used to raise (e.g. "No reliable triaxial
  anchor geometry at these angles and precision."). Confirmed by direct
  reproduction. A minor diagnostic-quality step down, not a correctness
  issue — worth considering whether `_inverse_reporting_values` should thread
  the real `checks` dict through to the eager branch instead of a fixed
  string, so a failed `AllModels` report (e.g. when inspecting a historical
  run) still names what actually failed.
- Issue #82's other prerequisite (a generic default-prior registration path
  for native, no-`parameterization` types) is correctly left untouched by
  this PR, consistent with its stated scope.

## Validation

Environment: macOS `arm64`, Python 3.12.5, JAX 0.10.2, NumPyro 0.21.0,
Matplotlib 3.11.2, local `uv` environment (not the project's Colima image).

- `uv run ruff check tnt/ tests/`: passed.
- `uv run sphinx-build -q -E -b html -W docs/source`: passed.
- `uv run pytest tests/unit_tests/test_mge.py
  tests/unit_tests/test_mge_inverse_traceable.py
  tests/unit_tests/test_mge_inverse_reporting.py
  tests/unit_tests/test_mge_native_traceable.py
  tests/unit_tests/test_mge_parameterized_traceable.py
  tests/unit_tests/test_potential.py`: **451 passed, 1 failed** (F1 above).
- Hand-traced `_intrinsic_shape_candidate`'s stop-gradient/safe-substitution
  construction and independently reproduced its differentiability properties
  via a standalone script, matching the new test file's own assertions.
- Reproduced F1's root cause directly (isolated single-pair vs. tuple vs.
  cast-tuple comparisons) rather than inferring it from the traceback alone.
- The full repository suite and the project's Colima/Docker image were not
  used for this review; this audit's own validation is local-environment
  only, same caveat the PR's own description carries for its environment.
- PR head was unchanged during this review.

## Completion checklist

- [x] Read project context and reviewed every changed production module
      (`tnt/mge.py`, `tnt/potential/core.py`, `tnt/potential/oblate_mge.py`,
      `tnt/potential/triaxial_mge.py`).
- [x] Ran the directly-relevant test modules, lint, and a strict
      documentation build (not the full repository suite).
- [x] Reproduced and documented F1 from source, not from the traceback alone.
- [x] Rechecked that the PR head remained unchanged during the audit.

No production code was changed.
Following the project's review workflow, remove this temporary audit document
once its findings have been addressed, before merging.

Assisted by Claude Code (Anthropic).
