# PR #75 audit: Guard NFW conversion in traced potential construction

Branch: `codex/traced-nfw-construction` → `main`. Author: Thomas Maindl
(assisted by Codex). Stacked on #74 (merged); continues issue #72.

## Summary of the change

`Potential.build_with_validity()` (from #74) rejected any component with a
registered `parameterization` converter. This extends it to cover traceable
registered conversions (currently: NFW's `concentration_m200`), using a
probe-then-convert pattern in `ResolvedPotentialComponent.build_with_validity`:

1. Raw parameters are checked (finiteness + registered raw bounds) exactly as
   in #74, giving `valid`.
2. If a converter is registered: `convert()` (the registered converter, plus
   a static structural check of its own output -- names/types/dimensions/
   scalar-shapes) is abstractly shaped via `jax.eval_shape` to build a
   correctly-typed zero `placeholder`, unconditionally -- so a malformed
   converter (a programming error) always raises, regardless of whether the
   raw proposal is itself numerically valid.
3. A `jax.lax.cond(valid, convert, placeholder)`, wrapped in
   `stop_gradient`, *probes* the converted values' own finiteness/bounds
   (`self.canonical_constraints`, the same set `build()`'s eager path uses)
   without differentiating through `convert` -- this is what catches raw
   proposals that are positive/finite but whose conversion overflows or
   cancels into something unusable.
4. `valid` is updated to include that probe result, and only then is a
   second, *differentiable* `lax.cond(valid, convert, placeholder)` used to
   build the actual `canonical` parameters. `convert`'s numerics are
   therefore never concretely executed, forward or backward, for a proposal
   already known invalid -- avoiding the classic "compute both branches then
   mask" pitfall where an invalid branch's internal NaN/Inf contaminates the
   gradient of the valid one even after the value itself is discarded.

## Verification performed

- Checked out `codex/traced-nfw-construction` (still parented on the
  pre-squash `codex/traceable-potential-validation` history; diffed against
  `main`'s current tip with a plain two-ref `git diff main HEAD`, which
  correctly isolates just this PR's own changes since #74's content is
  identical either way).
- Full suite: `561 passed`. `ruff check`: clean.
- Traced the constraint-set wiring by hand: `_parameter_values_valid` (new,
  shared by both raw and canonical checks) is called with
  `self.canonical_constraints` in the traced path, the exact same field
  `build()`'s eager path passes to `_validate_parameter_constraints` --  no
  drift between the eager and traced canonical constraint sets.
- Confirmed the "malformed converter output must always raise" property
  directly follows from `jax.eval_shape(convert, raw)` running
  unconditionally (not inside the `valid`-gated `cond`), and is exercised by
  `test_traced_nfw_conversion_contract_errors_raise` for all four structural
  failure modes (names/type/dimension/shape), each checked at both a valid
  and an invalid raw `c`.
- The gradient-correctness test
  (`test_traced_nfw_build_guards_conversion_and_preserves_gradients`) checks
  three independent gradients (concentration, mass, Hubble parameter)
  against a closed-form finite-difference reference in both float32 and
  float64, across two different declared-unit spellings, and separately
  asserts every gradient is *exactly* zero (not just finite) for a
  comprehensive invalid-input list spanning raw-invalid, converted-overflow,
  and bad-cosmology cases -- this is the correctness-critical claim of the
  whole PR and it's tested at the right level of rigor, not just smoke-tested.

## Findings

None. No correctness issues found. The eager/traced constraint-set sharing
established in #74 is preserved here without new gaps, and the probe/mask
split is exactly the right JAX pattern for the NaN-gradient hazard it's
built to avoid (independently reasoned through above, not just taken on
the PR description's word).

## Recommendation

Ready to merge.
