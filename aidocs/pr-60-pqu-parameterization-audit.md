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
