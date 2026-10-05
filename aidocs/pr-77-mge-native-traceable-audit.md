# PR #77: Make native MGE validation and construction JAX-traceable — audit

- Date: 2026-10-05
- Reviewer: Claude, for Thomas Maindl
- PR: [Make native MGE validation and construction JAX-traceable](https://github.com/dynamics-of-stellar-systems/tnt/pull/77)
- Author: `maindlt` (Assisted by Codex)
- Reviewed head: `7f78f98c4248c0c5ef47c97d6912e21f1867a159`
- PR base (merge-base with `main`): `a5285b4687d74824a531b83c44f15537046f89cd`

## Recommendation

The core design — one scalar validity flag per proposal, numerical probes
detached from autodiff before conditional construction, shared geometry/unit
checks across oblate/triaxial light/mass — is sound and the tests back it up.
One finding (F1) looks like an unintended regression and is worth fixing or
explicitly justifying before merge; the rest are maintainability/performance
observations, not correctness blockers.

## Findings

### F1 — P2: `PA_twist` axisymmetry check tightened from tolerant to exact equality, with no stated reason

Locations: `tnt/mge.py:624` (eager `deproject_oblate`) and `tnt/mge.py:695`
(traced `_oblate_candidate`'s `valid` expression, used by
`deproject_oblate_with_validity` / `_build_mge_with_validity`).

The diff against `main` changes:

```python
if not bool(jnp.allclose(self.PA_twist.ustrip("rad"), 0.0)):
```

to:

```python
if not bool(jnp.all(self.PA_twist.ustrip("rad") == 0)):
```

and adds the same exact-equality form to the traced validity flag. `allclose`
is equally JAX-traceable, so this isn't required for the traceability goal of
the PR, and nothing in the PR description or commit messages explains the
tightening.

Reproduced directly against this branch: a `LightMGE` with
`PA_twist = [0, 1e-10] deg` (i.e. effectively-zero isophote twist, the kind of
value that can arise from floating-point roundoff in an upstream unit
conversion) passed the old `allclose` check (`True`) but now makes
`deproject_oblate` raise `ValueError` eagerly, and makes
`deproject_oblate_with_validity`'s flag come back `False`:

```
deproject_oblate RAISED: ValueError - deproject_oblate requires PA_twist == 0 ...
traced valid flag: False
allclose(1e-10 deg in rad, 0) = True
```

An MGE that would have deprojected fine before this PR is now rejected.

Requested fix: either revert to `jnp.allclose` (default tolerance) for both
checks, or, if exact equality is intentional (e.g. to match a stricter
upstream contract), say so in the docstring/commit message and confirm no
existing fixture or caller relies on near-zero-but-nonzero twist.

### F2 — P2: Diagnostic fallback in `build()` doesn't cover all of `probe()`'s validity conditions

Location: `tnt/potential/components.py:196-209`.

```python
component, valid = _build_mge_with_validity(
    self.component_cls, canonical, self.extra_fields, jnp.asarray(True)
)
if not bool(valid):
    # Preserve the eager geometry diagnostics from the same checks.
    self.component_cls._build(
        canonical, cosmological_parameters, self.extra_fields
    )
    raise InvalidPotentialParametersError(
        f"Invalid construction for {self.path}: converted values or "
        "derivatives are not representable at this numerical precision."
    )
```

The comment says this re-call to `_build` exists to "preserve the eager
geometry diagnostics" — i.e. if `_build` would raise a specific `ValueError`
for a bad geometry, that more informative exception propagates instead of the
generic one below it. But `probe()` (`components.py:258-288`) ANDs the
geometry/domain checks together with forward/reverse-autodiff finiteness and
agreement checks that `_build` never runs. When `valid` is `False` only
because of a derivative-agreement failure (e.g. a borderline-conditioned
viewing angle where `jacfwd`/`jacrev` disagree beyond
`50 * sqrt(eps) * scale`), `_build` returns normally with no exception, its
result is discarded, and the generic `InvalidPotentialParametersError` fires
anyway — contrary to what the comment claims for that case. This doesn't
cause incorrect accept/reject behavior (the proposal is still correctly
rejected), but the diagnostic is weaker than advertised for an entire class
of rejections.

Requested fix: either have `probe()`'s derivative checks also raise/report
*which* condition failed (geometry vs. derivative agreement vs. derivative
finiteness) so `build()` can forward the right message, or adjust the comment
to say the diagnostic re-call only helps for geometry/domain failures.

### F3 — P3: Domain checks are hand-duplicated between the eager and traced construction paths

Locations: `tnt/mge.py:612-625` (eager, Python `if`/`raise`) vs.
`tnt/mge.py:690-698` (traced, `_oblate_candidate`'s `valid` expression) —
same pattern for the triaxial counterpart at `tnt/mge.py:820` /
`tnt/mge.py:857`.

Inclination range, `sigma`-is-length, and `PA_twist == 0` are each checked
once by hand in eager Python and again as a JAX boolean predicate in the
traced `valid` flag. F1 shows this is not just theoretical: the two copies
already drifted out of sync for one line (`allclose` vs. exact `==`) in this
same PR — had only one been updated, eager and traced construction would
silently accept/reject different inputs for the same MGE.

Requested fix (non-blocking): extract shared boundary predicates (e.g. an
`_inclination_valid(angle) -> jax.Array` usable both as a raise-condition and
inside `valid`) so eager and traced paths read from one definition.

### F4 — P3: `_build_mge_with_validity`'s traced scaffold reimplements `_guard_deprojection` by hand

Locations: `tnt/potential/components.py:290-312` vs.
`tnt/mge.py:92-107` (`_guard_deprojection`, already used by
`deproject_oblate_with_validity` / `deproject_triaxial_with_validity`).

Both do the same thing — `jax.eval_shape` a zero placeholder, probe under
`jax.lax.stop_gradient`, then `jax.lax.cond` into either the real
construction or the placeholder — but as two independently written copies.
A future fix to this scaffolding (e.g. a new edge case in how invalid
proposals are kept out of the differentiated path) made in `tnt/mge.py` would
not propagate to `components.py`'s copy.

Requested fix (non-blocking): have `_build_mge_with_validity` call
`_guard_deprojection` (or a shared generalization of it) instead of
re-deriving the same control flow.

### F5 — P3: `probe()` runs full forward+reverse Jacobians unconditionally, even from plain eager `build()`

Location: `tnt/potential/components.py:258-269`, called from both
`build_with_validity` and the untraced branch of `_build_mge_with_validity`
(line ~302, reached from plain `build()` at line 196).

`probe()` always computes `jax.jacfwd(numerical_outputs)` and
`jax.jacrev(numerical_outputs)` over an 8-output function, including when
`build()` is called eagerly outside any `jax.jit`/`jax.vmap` trace and the
caller has no use for gradients — e.g. once per model a sampler materializes.
Within one call, the deprojection `candidate()` itself is also recomputed
several times (once directly in `probe`, again inside `numerical_outputs`
under tracing for `jacfwd`, again under `jacrev`, and once more to build the
final returned model), multiplying the per-proposal cost of a construction
path meant to build one potential.

Requested fix (non-blocking, perf only): consider gating the Jacobian
checks to the paths that actually need gradient validity (`build_with_validity`
/ the traced branch), and reusing one `candidate()` evaluation for the final
model instead of recomputing it.

### F6 — Minor: redundant unit check duplicated via two mechanisms

Location: `tnt/mge.py:619` vs. `tnt/mge.py:652-659`
(`_check_deprojection_structure`).

`deproject_oblate`/`deproject_triaxial` each do
`self.sigma.unit.is_equivalent(au.m)` immediately before calling
`_check_deprojection_structure`, which a few lines later validates the same
requirement via `validate_dimension(value.unit, "length", ...)`. Harmless
today (both checks agree), but two sources of truth for one rule; not a
blocker.

## Scientific and implementation review

- The physical-surface-density convention from PR #76 is correctly carried
  through: `I` is unaffected by `angular_to_physical`, only `sigma` converts.
- Sharing geometry/representability checks between eager construction and the
  traced validity flag (modulo F1/F3) is the right shape for this problem —
  static contract errors raise, numerical proposal failures become a guarded
  `valid=False` with zeroed placeholders that never reach differentiated
  code.
- The forward/reverse-mode agreement check in `probe()` is a reasonable way
  to catch derivative blow-ups that a finite-forward-value check alone would
  miss (same pattern as PR #75's NFW gradient-finiteness check).
- Scope is honestly stated: Galax's own potential quadrature is explicitly
  out of scope, and `q_min`/`pqu`/`(T, T_maj, T_min)` traceable conversions
  are deferred, consistent with the PR description.

## Verification performed

Run locally (not via Docker/Colima; this machine has the project's venv with
all dependencies):

```sh
python -m pytest -q tests/unit_tests/test_mge.py tests/unit_tests/test_mge_native_traceable.py
python -m pytest -q tests/unit_tests/test_potential.py
```

- `tests/unit_tests/test_mge.py` + `test_mge_native_traceable.py`: **162
  passed**, 22 warnings (JAX x64-truncation warnings, pre-existing and
  unrelated to this PR).
- `tests/unit_tests/test_potential.py`: **193 passed**, 4 warnings (same
  cause).
- Reproduced F1 directly with a small script constructing a `LightMGE` with
  `PA_twist = [0, 1e-10] deg`: confirmed `jnp.allclose(...) == True` but
  `jnp.all(... == 0) == False`, and that `deproject_oblate` raises on this
  branch where it would have succeeded under the pre-PR check (see F1 for the
  exact output).
- Did not run the full repository suite (matches the PR description's own
  note that "the full repository test suite was not run for the final
  correction"); did not run `ruff`/Sphinx independently in this pass.

## Scope and follow-up

Only this audit document was added to the branch. Implementation fixes,
posting review comments, and merging are separate follow-up actions. After
F1/F2 are resolved (and F3-F6 addressed or explicitly deferred), rerun the
affected tests and remove this audit from the PR branch per the project
workflow before merging to `main`.
