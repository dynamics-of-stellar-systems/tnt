# PR 60: PQU parameterization audit

PR: **Add a (p, q, u) parameterization for the triaxial MGE potentials**

Branch: `pqu-parameterization`

Base: `main`

Audited head: `73a0fe56e8b67e6c6321b63cf02e423a2bfb38b3`

Audit date: 2026-09-02

## Decision

PR 60 is **not ready to merge yet**. Its overall design is reasonable, and
the ordinary interior-point conversion agrees with the referenced DYNAMITE
implementation, but two reproducible High-severity numerical/astrophysical
correctness problems require correction. These are focused problems rather
than a need for architectural revision.

GitHub reports the PR conflict-free and technically mergeable. It has no CI
checks, no review comments, and is blocked by the required review.

## Architectural summary

The PR adds `parameterization: "pqu"` to
`TriaxialLightMGEPotential` and `TriaxialMassMGEPotential`. A configuration
can therefore state the intrinsic ratios `p = B/A`, `q = C/A`, and the
scale-length compression `u` instead of the canonical global viewing angles
`theta`, `phi`, and `psi`.

The forward conversion uses the van den Bosch et al. (2008) relations at the
anchor `q' = min(component q)`. The inverse reuses the triaxial
angle-to-intrinsic-shape calculation extracted from `AbstractMGE` into
`_triaxial_intrinsic_axis_ratios()`. Runtime construction passes the named MGE
to converters, applies data-independent `ParameterConstraint`s before
conversion, applies MGE-dependent checks in `_pqu_to_tpp()`, and then builds
the existing canonical triaxial MGE component. Invalid physical candidates
raise `InvalidPotentialParametersError`, which `ModelIterator` records as an
invalid model.

To support the new parameterization, `register_parameterization()` now accepts
registered TNT component types as well as curated native `galax` types. The
two triaxial MGE classes override `raw_parameters()` so `AllModels` can report
the configured PQU representation after construction or mass rescaling.

## Findings

### High 1: documented inclusive upper boundaries do not survive construction

The converter treats `u == 1` as valid by returning exactly
`phi = psi = pi/2` in `tnt/potential/triaxial_mge.py` around lines 344-348.
Those exact angles are singular in `_triaxial_intrinsic_axis_ratios()` in
`tnt/mge.py` around lines 88-123.

Two failures were reproduced in the Linux development container:

- Full `ResolvedPotentialComponent.build()` with
  `(p, q, u) = (0.85, 0.60, 1.0)` raises `MGEDeprojectionError` because the
  resulting intrinsic `p` and `q` are `NaN`.
- `_tpp_to_pqu()` applied to those supposedly valid angles returns
  `p = q = u = NaN`.

The test `test_pqu_to_tpp_accepts_u_equal_to_one()` in
`tests/unit_tests/test_potential.py` around line 1809 checks only that the
converter returns finite angles. It never builds a component, inspects its
deprojection, or runs the inverse converter.

The other documented inclusive upper boundary is also numerically fragile.
For `q'=0.9`, `p=0.85`, `q=0.60`, and `u=p/q'`, roundoff produced
approximately `w2=-1.1e-15` and `w3=-2.5e-15`; `_pqu_to_tpp()` rejected the
point even though `u <= min(p/q', 1)` includes it.

Upstream DYNAMITE avoids the `u=1` singularity by moving `u` one
floating-point step below one before evaluating the formulas:
<https://github.com/dynamics-of-stellar-systems/dynamite/blob/master/dynamite/physical_system.py#L652-L719>.

Required correction:

- preserve the limiting geometry with a nonsingular near-one evaluation
  rather than returning the exactly singular angles;
- handle tiny roundoff excursions around mathematically valid zero weights;
- add full forward/build/deprojection/inverse tests for `u=1` and `u=p/q'`,
  for both light- and mass-MGE types.

The inverse/reporting semantics at `u=1` should be explicit: if a one-ULP
internal adjustment is used, decide whether `AllModels` should report the
nearly-one recovered value or preserve the exactly declared value.

### High 2: a twisted anchor silently changes the configured intrinsic shape

`_mge_min_observed_q()` in `tnt/potential/triaxial_mge.py` around lines
283-285 retains only `min(mge.q)`. `_pqu_to_tpp()` therefore calculates angles
as though that anchor Gaussian has zero twist. Actual deprojection in
`tnt/mge.py` around lines 437-443 adds each component's `PA_twist` to the
global `psi`.

With a minimum-`q'` anchor having `q'=0.76` and `PA_twist=0.2 rad`, requesting
`(p,q)=(0.85,0.60)` silently produced approximately
`(p,q)=(0.7366,0.6337)` for that anchor. The resulting shape is still
physically valid, so no exception reveals that a different intrinsic model
was constructed.

All new PQU fixtures use zero twist; see `_triaxial_light_mge()` in
`tests/unit_tests/test_potential.py` around lines 1736-1742.

Required design decision and correction:

- either require the selected minimum-`q'` reference Gaussian to have zero
  `PA_twist`, with a clear runtime error and defined handling of tied minima;
- or include the anchor twist consistently in both forward and inverse
  transformations.

Whichever policy is chosen needs a regression test with a nonzero anchor
twist. Until then, the documented meanings of `p`, `q`, and `u` are not
reliable for every valid TNT MGE.

### Medium: registration is more general than inverse dispatch

`register_parameterization()` in `tnt/potential/registry.py` around lines
503-524 now accepts every registered TNT component type. However,
`AbstractPotentialComponent.raw_parameters()` in
`tnt/potential/components.py` around lines 352-380 still returns canonical
parameters unchanged. Only the two triaxial MGE classes compensate with
duplicated overrides in `tnt/potential/triaxial_mge.py` around lines 116-124
and 199-207.

A future parameterization registered for another TNT component would
therefore be accepted but could silently report canonical rather than
configured parameters. The PR notes central inverse dispatch as a follow-up,
but the public registration rule is already broader than the implementation
that makes it safe.

Recommended correction:

- centralize registered inverse dispatch on `AbstractPotentialComponent`; or
- make registration reject a TNT component that does not provide the needed
  inverse path.

If this is deliberately deferred, create a concrete follow-up issue and make
the current API limitation explicit.

The converter type aliases were also widened to `Callable[..., ...]`, while
the registration tests' `_identity_forward` and `_identity_inverse` helpers in
`tests/unit_tests/test_potential.py` around lines 1010-1015 still have the
shorter signatures. Those tests verify registry storage but would fail if the
registered converters were actually called. Exact callable protocols and an
invocation test would make the converter contract clearer.

### Low: documentation and test comments contain contradictions

- `tnt/potential/components.py` around lines 365-366 still says the four MGE
  composite types do not support a parameterization.
- `docs/source/potential.md` around line 244 advertises `q <= p`, although
  `_pqu_to_tpp()` rejects `q == p` as a non-unique prolate geometry. The
  effective supported triaxial domain should be stated directly.
- `docs/source/potential.md` around line 253 describes previous TNT behavior
  (“was previously native-`galax`-only”). TNT is a new package, so this should
  be a direct description of current behavior.
- `tests/unit_tests/test_potential.py` around lines 1810-1811 says the code
  uses a one-ULP nudge at `u=1`, but the implementation returns exact analytic
  angles instead.
- The invalid-case comment around line 1830 says `0.98 > min(...)=1`; the
  actual violation is the lower bound because `q/q'` is approximately 1.053.
- The new `aidocs/KNOWLEDGE.md` material contains migration-style wording such
  as “were already”, “gained”, and “was always”. It should describe the
  current parameterization architecture directly.

## Test coverage assessment

The PR has good coverage of:

- configuration schema selection and exact parameter names;
- the ordinary interior-point PQU round trip;
- one known DYNAMITE conversion point;
- equivalent PQU and angle-based deprojections for light and mass MGEs;
- PQU reporting after mass rescaling;
- data-independent and MGE-dependent invalid candidates;
- `ModelIterator` recording a domain-invalid PQU point rather than crashing;
- preservation of the existing triaxial MGE deprojection behavior after the
  helper extraction.

Missing or weak coverage:

- `u=1` through complete component construction and inverse reporting;
- the exact `u=p/q'` upper boundary;
- a minimum-`q'` anchor with nonzero `PA_twist`;
- tied minimum-`q'` components with different twists;
- actual invocation of converter functions registered through the generalized
  registry tests;
- optional direct coverage under `numerics_settings.jax_enable_x64: false` for
  boundary-sensitive calculations.

## JAX, units, and runtime-boundary assessment

The PQU conversion uses Python `float`/`math` operations and remains eager.
That is consistent with the existing documented boundary:
`ResolvedPotentialComponent.build()` runs outside `jax.jit`/`jax.vmap`, and
only a constructed potential enters compiled numerical work. The extraction
of `_triaxial_intrinsic_axis_ratios()` preserves the existing JAX array
calculation used by MGE deprojection.

`p`, `q`, and `u` are correctly dimensionless. `ml` retains its declared
mass-to-light unit, while `mge_mass_scale` remains dimensionless. The
converter produces angles in radians without normalizing configuration
declarations, consistent with TNT's unit boundary. No new Equinox mutability
or PyTree problem was found.

## Checks run

Using the Colima Linux development environment at the audited head:

- full suite: **388 passed**, with one dependency-owned TensorFlow
  Probability/JAX deprecation warning;
- `ruff check .`: passed;
- strict Sphinx (`sphinx-build -E -b html -W`): passed;
- `git diff --check main...HEAD`: passed;
- all Python files changed by PR 60 pass `ruff format --check`.

Repository-wide `ruff format --check .` still reports eight files inherited
from `main`; none is changed by PR 60. Apple Silicon was not rerun during this
audit. GitHub currently provides no CI results for this PR.

The two adversarial calculations described in the High findings were run
separately in the same Linux container and reproduce reliably.

## Recommended merge path

1. Fix and regression-test `u=1` and the inclusive `u=p/q'` boundary through
   full construction and inverse reporting.
2. Decide and enforce the minimum-`q'` anchor-twist policy.
3. Correct the stale documentation, docstrings, and test comments.
4. Preferably centralize or constrain inverse dispatch; otherwise record the
   limitation in a concrete issue.
5. Rerun the full Linux suite, Ruff, strict Sphinx, and the adversarial
   boundary/twist cases.
6. Obtain the required review approval, then remove this audit file in its own
   commit before merging.

## Questions for Prash and Thomas

1. Should PQU support a nonzero `PA_twist` on the minimum-`q'` reference
   Gaussian by shifting the reference frame, or should such an MGE be rejected?
2. If more than one component shares the minimum `q'`, how is the anchor chosen
   when their twists differ?
3. For declared `u=1`, should `AllModels` preserve exactly `1.0`, or is the
   one-ULP-below-one value used internally to retain an invertible geometry an
   acceptable reported value?
4. Should safe inverse dispatch for every registered TNT component be completed
   in this PR, or tracked as an immediate follow-up?

## Response (2026-09-10, at head `f7afa64`)

The branch was rebased onto `main` (past #63 and #64) and all findings are
addressed. Full suite 429 passed, `ruff` clean, strict `sphinx-build` clean
(macOS).

Line references below are to the current branch head, not the audited head.

### Where the conversion now lives

Both directions of the van den Bosch relation are now `AbstractMGE` methods,
sharing one core:

- `_triaxial_component_ratios(theta, phi, psi) -> (p[], q[], u[])` -- each
  Gaussian's line-of-sight angle `psi + PA_twist` (vdB eq. 6), then
  `_triaxial_intrinsic_axis_ratios`. `deproject_triaxial` and
  `triaxial_intrinsic_shape` both call it, so that convention has one home.
- `triaxial_viewing_angles(p, q, u) -> (theta, phi, psi)` -- the vdB
  `triax_pqu2tpp` math, anchor selection (`_triaxial_anchor`), `u`-boundary
  nudge, weight noise-floor, and the anchor-twist fold; raises
  `MGEDeprojectionError`.
- `triaxial_intrinsic_shape(theta, phi, psi) -> (p, q, u)` -- the
  anchor-component slice of `_triaxial_component_ratios`, the exact inverse.

`_pqu_to_tpp` / `_tpp_to_pqu` in `tnt.potential.triaxial_mge` are now
~10-line registry adapters: call the method, package the result, and
translate `MGEDeprojectionError` -> `InvalidPotentialParametersError`.
`_triaxial_intrinsic_axis_ratios` is no longer imported outside `tnt.mge`, so
the potential layer no longer reverse-engineers `deproject_triaxial`'s
`psi + PA_twist` to invert it (raised in review, not in the audit).

### Decisions

1. **Nonzero anchor twist (High 2): fold it in, don't reject.**
   `triaxial_viewing_angles` subtracts the anchor Gaussian's `PA_twist`
   (`delta`) from the global `psi` it returns; `_triaxial_component_ratios`
   adds `delta` straight back for the anchor, so `(p, q, u)` name the
   anchor's intrinsic shape whatever the MGE's twist profile.
   `triaxial_intrinsic_shape` is the exact inverse.
2. **Tied minimum `q'`: first by component order** (`jnp.argmin`), documented
   on `AbstractMGE._triaxial_anchor` and in `KNOWLEDGE.md`.
3. **`u = 1` reporting: no artificial snapping.** Forward evaluates `u` a hair
   inside the domain; the inverse recovers `u = 1` to within roundoff
   (`~1e-8`), and that is what `AllModels` reports. Preserving an exact `1.0`
   would need a matching special case in the inverse and was not judged worth
   it; revisit if resume-compatibility on a declared `u = 1` proves noisy.
4. **Inverse dispatch (Medium): follow-up.** Issue #65 tracks centralizing it
   on `AbstractPotentialComponent` (and the `mge`/`cosmological_parameters`
   converter-context fold). `register_parameterization`'s docstring and
   `KNOWLEDGE.md` now state that a parameterization on any other TNT component
   type would silently report canonical parameters until then.

### High 1 -- boundary geometries

The de Zeeuw & Franx weights are singular exactly on every `u` boundary
(`u` in `{p, q/q', p/q', 1}`), each a valid limiting geometry. Rather than
special-casing `u == 1` to the exactly-singular `phi = psi = pi/2` (which the
audit showed deprojects to `NaN`), `triaxial_viewing_angles` clamps `u` into
the open interval `(lo, hi)` by a relative `1e-9` margin -- interior values
untouched, DYNAMITE's `u == 1` nudge generalized. `_TRIAXIAL_WEIGHT_ATOL`
(`1e-9`) absorbs any residual roundoff on the weights; larger excursions
still raise. New tests `test_pqu_u_equal_to_one_builds_and_inverts` and
`test_pqu_accepts_the_upper_boundary_u_equals_p_over_qprime` cover
forward + build + inverse, plus `test_mge.py` method-level coverage; the
domain-rejection cases gained a real `u > min(p/q', 1)` case and their stale
comments were fixed.

### High 2 -- anchor twist

`_triaxial_anchor` returns `(q', PA_twist)`. Tests:
`test_pqu_folds_a_non_zero_anchor_twist_into_psi` and
`test_pqu_ignores_twist_on_non_anchor_gaussians` in `test_potential.py`;
`test_triaxial_viewing_angles_fold_the_anchor_pa_twist` and the
round-trip / `deproject_triaxial`-agreement tests in `test_mge.py`.

### Low / Medium docs

`potential.md` (twist folding, `q < p` for a triaxial solution, inclusive `u`
endpoints valid, dropped "previously native-galax-only"), `components.py`
`raw_parameters` docstring (no longer "four MGE composite types don't support
one"), `registry.py` + `KNOWLEDGE.md` (issue #65 limitation), and the
converter/module docstrings all updated.

The converter type-alias / `_identity_*` test-helper signature mismatch noted
under Medium is left as-is: those helpers exercise registry storage only, and
issue #65's centralization will settle the converter protocol.

---

## Re-audit — 2026-09-10

**Decision: not ready to merge.** The anchor-twist correction and shared MGE
conversion design are sound. The boundary correction fixes the original
default-precision examples, but still permits an uncaught division by zero
and does not reliably preserve the requested shape at 32-bit precision.
These are reproducible numerical correctness findings, independent of the
test-run timing limitation described below.

The re-audit reviewed and tested commit
`f2f9a1394262e63fab88c831fcbb47398196d901`, the latest remote
`origin/pqu-parameterization` commit and GitHub PR 60 head as of this review
on 2026-09-10. The remote was fetched before review, and GitHub was checked
again at the end to confirm that the PR head had not changed. All code
findings and validation results in this section refer to that exact commit,
not the outdated local branch. The comparison base was `origin/main` at
`d51fe2c`, which is an ancestor of the reviewed commit.

No project `AGENTS.md`, `CLAUDE.md`, or `aidocs/INDEX.md` was present;
the supplied user instructions, `README.md`, relevant sections of
`aidocs/KNOWLEDGE.md`, this audit and the author response guided the review.

This section is the current merge assessment. The earlier audit and author
response above are retained as review evidence, not TNT version-compatibility
or migration documentation.

### Status of the existing findings

| Finding | Assessment at the reviewed head |
| --- | --- |
| High 1: upper boundaries | Partially addressed; remains merge-blocking for the cases below. |
| High 2: anchor twist | Addressed. Forward subtracts the selected anchor twist; the shared per-component calculation adds it back. Existing regression tests pass. |
| Medium: inverse dispatch | Acceptable as an explicit follow-up under the original audit's deferral option. [Issue 65](https://github.com/dynamics-of-stellar-systems/tnt/issues/65) is open and covers centralized inverse dispatch or registration restrictions, plus converter context. The limitation is documented. |
| Low: documentation | Mostly addressed; remaining accuracy and formatting notes appear below. |

Both conversion directions belong to `AbstractMGE` and share
`_triaxial_component_ratios` with `deproject_triaxial`. The minimum observed
axis ratio selects the anchor, with the first component winning ties.
Only that anchor's twist enters the forward conversion; every component's
twist still enters its own deprojection and physical validity check.
The two triaxial potential classes share `_mge_raw_parameters` for inverse
reporting. This review found no additional architectural blocker.

### High 1a: the boundary adjustment can round away and crash model evaluation

Location: `tnt/mge.py:580-596`, `AbstractMGE.triaxial_viewing_angles`.

With default 64-bit precision, a one-Gaussian MGE with `q'=0.76`, zero twist,
and requested `(p,q,u)=(0.99999999,0.60,1.0)` satisfies the declared domain.
Full `ResolvedPotentialComponent.build()` raises
`ZeroDivisionError: float division by zero` for **both**
`TriaxialLightMGEPotential` and `TriaxialMassMGEPotential`.

Here `lo=0.99999999`, `hi=1`, and the margin is approximately `1e-17`.
`hi - margin` rounds back to exactly `1.0`, leaving `u_calc=1` and a zero
`1-u_calc**2` denominator in `w2`. The weight tolerance is applied after
the division and cannot prevent this failure. `_pqu_to_tpp` translates only
`MGEDeprojectionError`; `ModelIterator._evaluate` catches only that exception
and `InvalidPotentialParametersError` around potential construction
(`tnt/model_iterator.py:451-461`). Therefore this arithmetic exception escapes
instead of producing an invalid-model record. The build failure was executed;
the iterator consequence follows directly from that exception boundary.

Required correction: ensure the evaluated geometry is representably inside
the supported domain, including narrow intervals. Handle an interval that
cannot be evaluated reliably with an explicit domain error, and regression-test
the full build plus iterator behavior for both component types. Do not merely
hide the division error or return unchecked angles.

### High 1b: the boundary treatment is unreliable with 32-bit precision

Locations: `tnt/mge.py:580-615` and the shared intrinsic-shape calculation.

In a fresh process, set `jax_enable_x64=False` before constructing any probe
quantities. With one Gaussian and zero twist, full construction gives these
results for **both** light and mass types:

| Observed `q'` | Requested `(p,q,u)` | Actual result |
| --- | --- | --- |
| `0.76` | `(0.85, 0.60, 1.0)` | Build succeeds but the anchor has `p=0.8501268625`, `q=0.5997300744`, recovered `u=1.0`. |
| `0.76` | `(0.76, 0.60, 1.0)` | Build raises `MGEDeprojectionError`; recovered `p` and `q` are `NaN`. |
| `0.80` | `(0.80, 0.50, 1.0)` | Build raises `MGEDeprojectionError`; calculated `p≈0.9999981`, `q≈7.423384`. |

The first row silently changes the requested intrinsic shape by approximately
`1.27e-4` in `p` and `2.70e-4` in `q`, much larger than ordinary float32
roundoff. For comparison, the interior point `(0.85,0.60,0.93)` recovers
`(0.8500000238,0.6000000834,0.9300000668)` in the same process.
The coincident upper boundaries `u=1=p/q'` are particularly ill-conditioned.
The domain-width margin is calculated with Python floats, but the resulting
angles and inverse calculation use the configured JAX precision; the current
margin does not control error through that complete calculation.

Required correction: use a boundary evaluation strategy with verified accuracy
at each supported precision, including coincident boundaries. Regression tests
must inspect the built anchor shape, compression and inverse reporting, not
just angle finiteness. If a geometry cannot be represented reliably, reject it
explicitly and document the actual supported domain rather than silently
constructing a materially different model. Include both component types.

### Reproducing the blocking cases

Run this in the Linux development container with `python -` (standard input),
once with `ENABLE_X64=True` and once with `False`. The flag must be set before
constructing the MGE and parameter quantities. Each probe uses a physical
sigma of 1 kpc and intensity of 1 solar luminosity per square parsec.

```python
import tnt
import jax
import jax.numpy as jnp
from unxt import Quantity as Q
from tnt.mge import LightMGE
from tnt.potential.components import AbstractPotentialComponent

ENABLE_X64 = True
jax.config.update("jax_enable_x64", ENABLE_X64)
cases = [(0.76, 0.99999999, 0.60, 1.0)] if ENABLE_X64 else [
    (0.76, 0.85, 0.60, 1.0),
    (0.76, 0.76, 0.60, 1.0),
    (0.80, 0.80, 0.50, 1.0),
]
for observed_q, p, q, u in cases:
    for kind in ("Light", "Mass"):
        mge = LightMGE(
            I=Q(jnp.array([1.0]), "Lsun / pc2"),
            sigma=Q(jnp.array([1.0]), "kpc"),
            q=Q(jnp.array([observed_q]), ""),
            PA_twist=Q(jnp.array([0.0]), "rad"),
            major_axis_pa=Q(0.0, "deg"),
        )
        mass_name, mass_unit = "ml", "Msun / Lsun"
        if kind == "Mass":
            mge = mge.to_mass(Q(1.0, mass_unit))
            mass_name, mass_unit = "mge_mass_scale", ""
        resolved = AbstractPotentialComponent.resolve(
            {"type": f"Triaxial{kind}MGEPotential",
             "parameterization": "pqu", "mge": "m", "parameters": {}},
            {"m": mge}, path="potential.stars",
        )
        try:
            component = resolved.build(
                {mass_name: Q(1.0, mass_unit), "p": Q(p, ""),
                 "q": Q(q, ""), "u": Q(u, "")}, {},
            )
            print(kind, (p, q, u), mge.triaxial_intrinsic_shape(
                *(component.parameters[k] for k in ("theta", "phi", "psi"))
            ))
        except Exception as error:
            print(kind, (p, q, u), type(error).__name__, str(error))
```

### Coverage and documentation accuracy

- The original default-precision cases now pass independent full-build and
  inverse probes for both types: `(q',p,q,u)=(0.76,0.85,0.60,1)` and
  `(0.90,0.85,0.60,0.85/0.90)`. The added upper-boundary test's point
  `(0.80,0.70,0.55,0.70/0.80)` also passes a full one-Gaussian build.
- `test_pqu_accepts_the_upper_boundary_u_equals_p_over_qprime` itself calls
  only forward and inverse converters, despite the response's claim of
  forward/build/inverse coverage. The `u=1` build test covers only the light
  type. Add full boundary coverage for both types and both precisions.
- First-component tie selection is clear from `jnp.argmin`, but there is no
  dedicated tied-minimum/different-twist regression test. The short
  `_identity_*` registry test signatures and broad `Callable[..., ...]`
  aliases remain a non-blocking API/testing concern.
- `test_pqu_to_tpp_accepts_u_equal_to_one` still says the calculation uses
  the largest float below one; it actually uses a domain-width-dependent
  margin. The comment that interior values are untouched is also too broad:
  values within the margin of either boundary are adjusted.
- `triaxial_intrinsic_shape` is a numerical inverse, not an "exact inverse"
  at adjusted boundary points. Describe the adjustment, actual precision
  limits and recovered-value reporting explicitly in current-behavior docs.
  `aidocs/KNOWLEDGE.md` should say that violations of the MGE-dependent
  inequality are rejected, rather than saying the inequality is rejected.
  Lower endpoints remain excluded by `lo < u`, despite prose about all
  boundary geometries. Remove the incidental "first non-native
  parameterization" chronology in `docs/source/potential.md`; describe the
  supported component types directly. No TNT version migration or
  backward-compatibility narrative is needed.

### Validation at the reviewed head

- Linux/amd64, Colima, documented Compose development environment:
  `pytest -q` completed with **427 passed, 2 failed** in 365.11 seconds.
  Both failures are the existing parameterized
  `test_read_applies_jax_precision_policy_in_isolated_process` tests
  (`False` and `True`), each reaching its 30-second subprocess timeout.
  They are not failed scientific assertions. Rerunning those two cases alone
  passed: **2 passed** in 56.84 seconds. Thus all 429 collected tests passed
  across the initial run and focused rerun, but the single full-suite run
  was not clean. One dependency-owned TensorFlow Probability/JAX deprecation
  warning was emitted in each run.
- `ruff check .`: passed.
- `sphinx-build -E -b html -W docs/source /tmp/pr60-sphinx`: passed.
- `git diff --check origin/main...HEAD`: passed.
- `git diff --check` after the audit-only edit: passed.
- Formatting check of the nine changed Python files: three files would be
  reformatted. The new unformatted blocks are in `test_potential.py`;
  the reported blocks in `test_mge.py` and `tnt/mge.py` are inherited from
  `main`. This is a cleanup item, not the numerical merge blocker.
- The separate 64-bit and 32-bit construction/inverse probes above were
  executed for both component types. Native macOS validation was not rerun.
- GitHub reports no status checks for this head. The prior
  `CHANGES_REQUESTED` review remains present. This re-audit records its
  findings in this document; no GitHub review, comment or merge was submitted.

### Required before approval

Correct and regression-test the remaining boundary failures, accurately
document the precision/domain behavior, and rerun the relevant checks.
The twist finding can remain closed and issue 65 can remain a scoped
follow-up. Keep this audit available for the next review; do not treat the
author response's "all findings are addressed" statement as the current
merge decision.
