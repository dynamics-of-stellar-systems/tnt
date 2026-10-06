# PR #77: Make native MGE validation and construction JAX-traceable — audit

- Date: 2026-10-06
- Reviewer: Claude, for Thomas Maindl
- PR: [Make native MGE validation and construction JAX-traceable](https://github.com/dynamics-of-stellar-systems/tnt/pull/77)
- Author: `maindlt` (Assisted by Codex)
- Reviewed head: `7f78f98c4248c0c5ef47c97d6912e21f1867a159`
- PR base (merge-base with `main`): `a5285b4687d74824a531b83c44f15537046f89cd`

## Implementation response (2026-10-06)

The audit below describes the reviewed head above. Thomas subsequently approved
one shared builder, removal of routine derivative cross-checks, completion of
all registered MGE parameterizations, and iterator migration.

- **Q2 / F4:** commit `51e15ad` removes full forward/reverse construction
  Jacobians from each proposal. Geometry and value representability checks
  remain. Validity no longer promises construction derivative agreement.
- **Q1 / F1:** `Potential.build_with_validity` replaces `build_potential`,
  `Potential.build`, `Potential.from_settings`, component `build`, and the
  private MGE construction factories. There is no diagnostic fallback call.
  The iterator checks the flag before numerical use, records rejected raw
  parameters, and logs a generic numerical-validation failure. Static setup
  errors propagate instead of being recorded as rejected proposals.
- **Q1 completion:** `q_min`, `pqu`, and `T_maj_min` are traceable for both light
  and mass components. Shared candidates preserve anchor selection/twist,
  numerical endpoint margins, and shape round-trip checks. Iterator processing
  remains one proposal at a time; batching is not required to retire the old
  construction interface. Prior integration remains separate work.
- **F2 / F5:** oblate domain predicates are shared with standalone diagnostics;
  the duplicate physical-width checks are removed in favor of the structural
  unit validator. Standalone scientific conversions also share the same
  numerical candidates as the registered converters.
- **F3:** `_guard_construction` owns detached probing, placeholder construction,
  and conditional differentiation for conversions and MGE deprojection/building.
  Ordinary and compiled execution use the same candidate and predicates.
- **Test cost:** mixed valid/invalid `vmap` coverage covers all four native
  types and all six registered MGE type/parameterization combinations at both
  precisions. Unit and integer-column gradient tests use representative types;
  focused shape-gradient tests remain, including direct versus compiled
  differentiation and finite-difference references. No full-suite speedup
  factor is claimed without a comparable baseline measurement.

Final affected tests: **404 passed** in sequential Colima processes: 115 MGE,
39 native-construction, 15 converted-construction, and 235 potential/iterator/
model-search tests. The full repository test suite was not run. Existing JAX
precision and upstream deprecation warnings remain.

This document remains as the review handoff; remove it before merging, per the
project workflow. Lint and documentation verification are recorded in the
implementation commit.

### Before/after benchmark (2026-10-06)

The native-construction test file averaged **920.25 seconds before** and
**204.54 seconds after** the Q1/Q2 implementation: **4.50× faster**, a **77.8%**
runtime reduction, saving approximately **11 minutes 56 seconds per run**.

Compared revisions `b46b0c06563646ecbdb9205c2d21ea7f2bf781c5` (before) and
`f448e5fe2d6910551f56878bc11dd3719de0c817` (after). Each revision ran twice,
sequentially in before–after–after–before order. Each run used a fresh Colima
Linux x86_64 container, a read-only source mount, the same pinned image
(`sha256:37ae2dab37655a2590f2a9003cddff9a1b132cab6f89e69e9ce30d2f264c0727`),
and `JAX_ENABLE_COMPILATION_CACHE=false`. Compilation time is included.

```sh
python -m pytest -q tests/unit_tests/test_mge_native_traceable.py --durations=10 -p no:cacheprovider
```

| Run | Tests passed | Pytest runtime (seconds) |
| --- | ---: | ---: |
| Before 1 | 47 | 951.40 |
| After 1 | 39 | 208.05 |
| After 2 | 39 | 201.02 |
| Before 2 | 47 | 889.09 |

All four runs passed. This measures the combined implementation and test
coverage changes in this one file, whose test count decreased from 47 to 39;
it does not establish a full-repository speedup or isolate the implementation
gain with identical tests. Two repeats provide an estimate, not statistical
certainty. No other containers were running at the benchmark start; other
host activity was not controlled.

## Questions from Prash

### Q1 — Should `build_with_validity` replace `build_potential` entirely?

We shouldn't have two ways of doing the same thing. I think `build_with_validity`
should eventually replace the eager `build_potential` path outright. If we do
that: how does `build_with_validity` get threaded into `ModelIterator`? My
guess is it needs a `vmap` — build and validity-check a batch of proposals at
once, filter out the invalid ones, then only run the (non-traceable) orbit
integration / weight solve on the survivors.

*Context gathered during this review:* today, the one real production
consumer is `ModelIterator._evaluate` (`tnt/model_iterator.py:417-460`), which
calls the eager `build_potential` specifically to catch named Python
exceptions (`MGEDeprojectionError`, `InvalidPotentialParametersError`) with
human-readable messages for its per-point warning log — `build_with_validity`'s
boolean flag has no message to attach to that log line. `build_with_validity`
has no consumer anywhere in the codebase yet; the PR description itself defers
"prior/model-iterator integration" to a later phase. Resolving this question
would directly settle F1 below (the diagnostic-fallback gap only exists
because both paths currently coexist) and F2 (the hand-duplicated domain
checks only risk drifting because there are two paths to keep in sync), and
would reframe F4 from "`probe()` shouldn't run from eager `build()`" to
"eager `build()` shouldn't exist at all."

### Q2 — Are the derivative checks overkill?

The test cases where forward- and reverse-mode actually diverge are pretty
pathological — either deliberately extreme parameter values, or an ordinary
galaxy declared in an extreme unit (kilometres). That's a lot of added
complexity to catch inputs that look like edge cases by construction. The
derivatives being checked (the Jacobian of `I`/`sigma`/`p`/`q`/mass with
respect to the raw parameters) may not even be the ones that end up
differentiated in the real modelling workflow. And since gradients will be
consumed exclusively by the prior module (an MCMC/HMC-style sampler),
computing an `(8,4)` Jacobian in *both* forward and reverse mode on every
sampler step seems like the wrong trade.

**Suggestion: drop the derivative checks.**

*Context gathered during this review:* the two documented divergence cases
are (1) `test_reverse_mode_overflow_is_rejected_eagerly_and_under_jit`, built
from `I ~ 1e-100` / `sigma ~ 1e103 kpc` specifically to overflow, and (2) the
`km`-unit case in `test_equivalent_width_units_preserve_native_mge_values_and_gradients`,
which was already fixed at the root by the second commit's "local astronomical
units" change rather than merely gated — so in the realistic-proposal regime
this check should rarely fire today. Separately: once `jax.jit`-compiled,
`valid`'s computation can't be dead-code-eliminated (it feeds the returned
flag), so the forward-mode Jacobian is re-executed on every call to the
compiled function, not just once at trace time — in an MCMC sampler that's
every leapfrog step, indefinitely, paying for a cross-check whose only
consumer is the check itself (the sampler's own gradient need is a single
reverse-mode VJP). Timed empirically on this branch: removing the derivative
block cut `test_equivalent_width_units_preserve_native_mge_values_and_gradients`
from ~17.5s to ~5.6s for the triaxial cases (~9.3s to ~3.5s for oblate),
consistent with the `(8,4)`/`(8,2)` dual-mode Jacobian being the dominant
cost.

## Recommendation

The core design — one scalar validity flag per proposal, numerical probes
detached from autodiff before conditional construction, shared geometry/unit
checks across oblate/triaxial light/mass — is sound and the tests back it up.
Before merging, it's worth settling Q1/Q2 above with Thomas, since the answers
change how much of F1/F2/F4 below is worth fixing in this PR versus moot under
a redesign. The remaining findings are maintainability observations, not
correctness blockers.

## Findings

### F1 — P2: Diagnostic fallback in `build()` doesn't cover all of `probe()`'s validity conditions

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
to say the diagnostic re-call only helps for geometry/domain failures. (Moot
if Q1 resolves toward removing eager `build()` entirely.)

### F2 — P3: Domain checks are hand-duplicated between the eager and traced construction paths

Locations: `tnt/mge.py:612-625` (eager, Python `if`/`raise`) vs.
`tnt/mge.py:690-698` (traced, `_oblate_candidate`'s `valid` expression) —
same pattern for the triaxial counterpart at `tnt/mge.py:820` /
`tnt/mge.py:857`.

Inclination range, `sigma`-is-length, and `PA_twist == 0` are each checked
once by hand in eager Python and again as a JAX boolean predicate in the
traced `valid` flag — two sources of truth for the same rule, with nothing
tying them together.

Requested fix (non-blocking): extract shared boundary predicates (e.g. an
`_inclination_valid(angle) -> jax.Array` usable both as a raise-condition and
inside `valid`) so eager and traced paths read from one definition. (Moot if
Q1 resolves toward removing eager `build()` entirely.)

### F3 — P3: `_build_mge_with_validity`'s traced scaffold reimplements `_guard_deprojection` by hand

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

### F4 — P3: `probe()` runs full forward+reverse Jacobians unconditionally, even from plain eager `build()`

Location: `tnt/potential/components.py:258-269`, called from both
`build_with_validity` and the untraced branch of `_build_mge_with_validity`
(line ~302, reached from plain `build()` at line 196).

`probe()` always computes `jax.jacfwd(numerical_outputs)` and
`jax.jacrev(numerical_outputs)` over an 8-output function, including when
`build()` is called eagerly outside any `jax.jit`/`jax.vmap` trace. `build()`'s
own docstring says this path "must remain outside `jax.jit`/`jax.vmap`
traces; only the resulting potential enters compiled numerical work" — i.e.
by contract, nothing downstream of an eager `build()` call ever differentiates
through it, so this cost is unconditional and provably unused in that path
(see Q1/Q2). Within one call, the deprojection `candidate()` itself is also
recomputed several times (once directly in `probe`, again inside
`numerical_outputs` under tracing for `jacfwd`, again under `jacrev`, and once
more to build the final returned model), multiplying the per-proposal cost of
a construction path meant to build one potential.

Requested fix: drop the probe from the eager path entirely (not just gate it)
once Q1 is settled; see Q2 for whether the dual-mode check belongs in the
traced path at all.

### F5 — Minor: redundant unit check duplicated via two mechanisms

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
  traced validity flag (modulo F2) is the right shape for this problem —
  static contract errors raise, numerical proposal failures become a guarded
  `valid=False` with zeroed placeholders that never reach differentiated
  code.
- The forward/reverse-mode agreement check in `probe()` is a reasonable way
  to catch derivative blow-ups that a finite-forward-value check alone would
  miss (same pattern as PR #75's NFW gradient-finiteness check) — but see Q2
  for whether its cost is justified given where it actually fires and who
  actually consumes the gradients it protects.
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
- Timed `test_equivalent_width_units_preserve_native_mge_values_and_gradients`
  with the `probe()` derivative block (forward/reverse Jacobians and
  agreement check) temporarily stripped out: triaxial cases dropped from
  ~17.5s to ~5.6s, oblate from ~9.3s to ~3.5s (see Q2).
- Did not run the full repository suite (matches the PR description's own
  note that "the full repository test suite was not run for the final
  correction"); did not run `ruff`/Sphinx independently in this pass.

## Scope and follow-up

Only this audit document was added to the branch. Implementation fixes,
posting review comments, and merging are separate follow-up actions. Q1 and
Q2 are open design questions for Thomas, not findings to fix unilaterally —
resolve those first, since they determine which of F1-F4 are worth fixing
here versus superseded by a `build_potential` removal / derivative-check
simplification. After that, rerun the affected tests and remove this audit
from the PR branch per the project workflow before merging to `main`.
