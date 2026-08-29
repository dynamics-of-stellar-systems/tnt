# PR 48 audit: axisymmetric MGE composite potentials

Audit date: 2026-08-28

Pull request: #48, `Implement to_galax() for the axisymmetric MGE composite potentials`

Base/head reviewed: `main` / `axisym-mge-v2`, after merging current `main` in
commit `59a8f8e`.

## Overall judgment

PR 48 is structurally consistent with TNT's current potential architecture and
the implemented mass normalization is mathematically correct for TNT's oblate
axisymmetric deprojection convention. I found no critical or high-severity
defect and no need for architectural revision.

I recommend **focused corrections before merge**, rather than merging exactly
as-is:

1. add one genuinely independent, flattened-axisymmetric numerical regression;
2. decide and document that the new public types are oblate-only, or rename
   them if a generic `Axisymmetric...` name is considered to promise future
   prolate support; and
3. remove the small set of stale internal comments that still describe only two
   MGE types or call the axisymmetric types future work.

The first two points matter most because TNT is new and this is the best time to
make the scientific contract and public names unambiguous. If the maintainers
explicitly accept `Axisymmetric...` as the name for an oblate-only convention,
the implementation itself appears fit to merge after the regression and
wording corrections.

## Architectural and mathematical summary

PR 48 adds `AxisymmetricLightMGEPotential` and
`AxisymmetricMassMGEPotential`, registered through the existing component
registry. Their configuration parameters are:

- light MGE: `ml` (mass/power) and `inclination` (angle);
- mass MGE: `mge_mass_scale` (dimensionless) and `inclination` (angle).

Resolution verifies the referenced MGE's runtime type. Construction preserves
the configured parameter units, applies `ml` or `mge_mass_scale`, and calls
`AbstractMGE.deproject_axisymmetric()` once. The resulting
`Deprojected3DMGE` is stored on the immutable Equinox component. The shared
unit system enters only later at `to_galax()`, where one
`galax.potential.AxisymmetricGaussianPotential` is created per Gaussian and
the terms are assembled into a `CompositePotential`. This is consistent with
the unit boundary merged from PR 47 and with the existing triaxial components.

For each deprojected Gaussian, TNT has central density `I`, intrinsic width
`sigma`, and intrinsic axial ratio `q`, with `p = 1`. `galax` defines

```text
rho_0 = m_tot / (q2 * (2*pi)^(3/2) * r_s^3).
```

Therefore the implementation's mapping

```text
r_s = sigma
q2 = q
m_tot = I * q * (2*pi)^(3/2) * sigma^3
```

is correct. Combining this with TNT's axisymmetric deprojection also recovers
the projected Gaussian total `2*pi*I_projected*q_observed*sigma^2`, independent
of inclination, as required.

`rescale()` copies the parameter mapping and uses `eqx.tree_at()` to multiply
only the stored three-dimensional density normalization. Shape parameters and
the original named MGE remain unchanged. The pattern matches the triaxial
implementation and is compatible with TNT's immutable runtime objects.

Orbit-library generation remains scaffolding in the pre-existing
`Potential.generate_orbit_library`; PR 48 does not change that boundary.

## Prioritized findings

### Critical

None.

### High

None.

### Medium 1: the flattened conversion has no independent numerical regression

References:

- `tests/unit_tests/test_potential.py:1064`
- `tests/unit_tests/test_potential.py:1104`
- `tests/unit_tests/test_potential.py:1147`
- `tnt/potential/axisym_mge.py:175`

The spherical cross-check uses `q = 1`. It can verify the spherical limit, but
it cannot detect a missing or incorrect flattening factor or an incorrect
`q -> q2` mapping. The multi-component flattened test constructs its expected
terms with the same `AxisymmetricGaussianPotential`, the same mapping, and the
same mass formula as the production helper; it verifies composition, not the
scientific mapping. The mass-MGE comparison is spherical as well.

The PR description says the tests cross-check the flattened implementation
against `TriaxialGaussianPotential(q1=1)`, but no such regression is present in
the reviewed test file.

Before merge, add a flattened case (`q != 1`) evaluated away from the principal
axes and compare it with an independent reference. A compact test can do both:

1. compare the axisymmetric potential with
   `TriaxialGaussianPotential(q1=1, q2=q)` at several coordinates; and
2. compare `potential.density()` with the analytic density represented by the
   stored `Deprojected3DMGE`.

This would detect errors in `q2`, `m_tot`, `r_s`, coordinate ordering, and the
normalization. It would also make the PR description accurate.

### Medium 2: the public scope is ambiguous, while one docstring incorrectly says prolate is supported

References:

- `tnt/potential/axisym_mge.py:3`
- `tnt/potential/axisym_mge.py:7`
- `tnt/mge.py:257`
- `tnt/mge.py:263`
- `tnt/mge.py:266`
- `tnt/mge.py:299`
- `tnt/mge.py:301`
- `docs/source/potential.md:120`
- `docs/source/potential.md:179`

The new module calls this the common "oblate/prolate case." TNT's actual
deprojection is the oblate convention: `p = B/A = 1`, `q = C/A <= 1`, with

```text
q_intrinsic^2 = (q_observed^2 - cos(inclination)^2) / sin(inclination)^2.
```

That represents `A = B >= C`. A prolate spheroid would instead have its long
axis as the symmetry axis (`q2 > 1` in the installed `galax` class, or an
equivalent relabelling), which TNT's `0 < q <= p <= 1` convention and `p = 1`
path do not produce.

At minimum, replace the `oblate/prolate` claim and state clearly in the public
potential documentation that these types currently implement oblate
axisymmetric deprojection only. Because TNT is new, the maintainers should
also decide now whether the generic names are desirable:

- keep `Axisymmetric...` and explicitly define it as TNT's present oblate-only
  axisymmetric convention; or
- rename to `OblateAxisymmetric...` if later prolate support should be exposed
  by a distinct type without changing the meaning of this API.

This is a contract/naming decision, not evidence that the implemented oblate
mathematics is wrong.

### Low 1: internal module documentation still describes the pre-PR state

References:

- `tnt/potential/triaxial_mge.py:10`
- `tnt/potential/components.py:3`
- `tnt/potential/components.py:77`
- `tnt/potential/components.py:108`
- `tnt/potential/components.py:145`
- `tnt/potential/components.py:222`
- `tnt/potential/components.py:275`
- `tnt/potential/registry.py:7`
- `tnt/potential/registry.py:185`

Several docstrings and comments still say there are two MGE composite types,
refer only to `deproject_triaxial`, or call the axisymmetric counterparts a
planned addition. Runtime behavior is unaffected, but these statements are
now contradictory and should be normalized in the same focused cleanup.

## Validation, parameter domains, and edge cases

The registry correctly supplies exact parameter names and dimensions to
configuration validation. Light and mass MGE references are type-checked, and
the new tests cover rejection of triaxial angle names and the requirement for
`inclination`.

Physical-domain validation remains incomplete across TNT and is already
tracked by issue #30. PR 48 adds cases that issue should explicitly cover:

- the accepted inclination convention/range, normally `0 < i <= 90 deg` for
  this oblate representation;
- finite `ml`, `mge_mass_scale`, and inclination;
- positive `ml` and `mge_mass_scale`; and
- numerically stable handling immediately above the minimum inclination
  `cos(i) = q_observed`, especially when 32-bit JAX precision is selected.

The existing eager `MGEDeprojectionError` correctly rejects impossible
geometries, `q_intrinsic = 0`, `NaN`, and intrinsic ratios outside TNT's
convention. Deferring the broader constraint framework to issue #30 is
reasonable and is not, by itself, a blocker for PR 48.

The MGE method already tests nonzero `PA_twist`, physical-length requirements,
flux conservation, and invalid inclinations in `tests/unit_tests/test_mge.py`.
Duplicating all of those tests at the potential wrapper level is unnecessary.

## JAX, Equinox, units, and platforms

- The new components are ordinary Equinox modules and follow the established
  immutable `eqx.tree_at()` rescaling pattern.
- Deprojection deliberately occurs in host-side construction, not under
  `jax.jit` or `jax.vmap`, because validation contains Python control flow and
  detailed exceptions. This is already TNT's documented runtime boundary.
- `to_galax()` constructs the static composite before orbit integration;
  `galax` owns JIT-compiled potential/density evaluation.
- Configured quantities and MGE source units remain preserved through
  construction. The configured internal unit system is supplied only to the
  `galax` constructors in `to_galax()`, matching current TNT policy.
- No OS-specific code or dependency change is introduced. Linux-container
  validation passed. A native Apple Silicon run was not performed for this
  audit; the changes use the same JAX/Equinox/galax stack already supported on
  that platform.

## Tests and checks performed

On the merged code at `59a8f8e`:

- `docker --context colima compose run --rm dev pytest`
  - 327 passed; three pre-existing dependency/multiprocessing warnings.
- `docker --context colima compose run --rm dev ruff check .`
  - passed.
- `docker --context colima compose run --rm dev sphinx-build -E -b html -W docs/source docs/build/html`
  - passed with warnings treated as errors.
- Focused axisymmetric/configuration selection:
  - 7 passed, 92 deselected.
- Inspected the installed `galax.potential.AxisymmetricGaussianPotential` and
  `TriaxialGaussianPotential` implementations to confirm their density,
  normalization, axis-ratio, and unit conventions.

GitHub showed no CI status entries, no reviews, and no review comments at audit
time. The PR was conflict-free and reported `MERGEABLE`; its `BLOCKED` state
was due to required review rather than a merge conflict.

## Recommended human review sequence

1. Decide whether `Axisymmetric...` intentionally means oblate-only in TNT.
2. Inspect `AbstractMGE.deproject_axisymmetric()` and the mass-normalization
   derivation above.
3. Add/review the independent flattened numerical regression.
4. Correct the public scope wording and stale internal docstrings/comments.
5. Confirm issue #30 explicitly includes the new inclination and normalization
   parameters.
6. Re-run the focused potential/configuration tests, full Linux suite, Ruff,
   and strict Sphinx build.
7. Merge after the required GitHub review is supplied.

## Decisions/questions for Thomas or Prash

1. Should TNT's generic `Axisymmetric...` public type names deliberately mean
   the present oblate-only convention, or should the names state `Oblate`?
2. Is the physical inclination domain canonically `0 < i <= 90 deg`, with
   equivalent angles outside that range rejected rather than silently mapped
   through squared trigonometric functions?
3. Should the independent flattened comparison be required in PR 48 (my
   recommendation) or accepted as immediate follow-up work?

## Response

All three points accepted and addressed in this PR.

### Naming: `Oblate...`, not `Axisymmetric...` (Medium 2, question 1)

The generic name over-promised. `AbstractMGE.deproject_axisymmetric` only ever
implemented the oblate convention (`p = B/A = 1`, `q = C/A <= 1`, i.e.
`A = B >= C`). A prolate spheroid has its long axis as the symmetry axis and
obeys a different projection relation
(`q_intr**2 = q_obs**2 sin(i)**2 / (1 - q_obs**2 cos(i)**2)`), which this code
path does not produce.

Renamed throughout:

- `AxisymmetricLightMGEPotential` -> `OblateLightMGEPotential`
- `AxisymmetricMassMGEPotential` -> `OblateMassMGEPotential`
- `AbstractMGE.deproject_axisymmetric` -> `AbstractMGE.deproject_oblate`
- module `tnt/potential/axisym_mge.py` -> `tnt/potential/oblate_mge.py`,
  helper `_galax_potential_from_axisym_deprojected` ->
  `_galax_potential_from_oblate_deprojected`

`Oblate` parallels the existing `Triaxial` pair and leaves `Prolate...` free
as a distinct future type. Prolate axisymmetric deprojection is filed as a
non-urgent follow-up (#49); it is not a capability gap today because
`deproject_triaxial` already reaches the prolate shape as its `p = q` limit.

### Inclination domain: `(0, 90]` deg, rejected not folded (question 2)

Confirmed canonical. It is the fundamental domain forced by two symmetries,
not a convention: equatorial mirror symmetry makes `i` and `180 deg - i`
give an identical projection (upper half-space redundant), and `i = 0`
(face-on) is singular -- the projection is circular and `sin(i) = 0` in the
deprojection denominator. It is the domain used in Monnet, Bacon & Emsellem
(1992) and Cappellari (2002), already cited by the method.

`deproject_oblate` now checks `0 < i <= 90 deg` before anything else and
raises `ValueError` otherwise, rather than folding `i in (90, 180)` deg
through the squared trigonometry (previously `120 deg` silently acted as
`60 deg`; `< i_min` and `i >= 180 deg` were already caught downstream by the
NaN / negative-`q` checks in `_check_axial_ratios`). Regression:
`test_deproject_oblate_rejects_inclination_outside_0_90` in `test_mge.py`.

The broader finite/positive `ml` / `mge_mass_scale` domain checks remain with
issue #30, as recommended.

### Flattened numerical regression: added to this PR (Medium 1, question 3)

`test_oblate_light_mge_to_galax_flattened_cross_checks` in `test_potential.py`
builds a genuinely flattened deprojection (`q_obs = 0.7` at `i = 70 deg` ->
`q_intr ~ 0.65`, asserted `< 0.99`) and checks `to_galax()` two independent
ways:

1. against `galax.potential.TriaxialGaussianPotential(q1=1, q2=q_intr)` -- a
   different `galax` class and code path from `AxisymmetricGaussianPotential`
   -- at three off-principal-axis coordinates; and
2. against the closed-form Gaussian density along each intrinsic axis, which
   only matches if `q2` maps to `z` (with `p = 1` on `x`/`y`).

Together these exercise `q2`, `m_tot`, `r_s`, and axis ordering. The stale PR
description claim about a `TriaxialGaussianPotential(q1=1)` cross-check is now
true; the description has been updated to match.

### Low 1: stale internal docs

Normalized in the same pass: `triaxial_mge.py` (axisymmetric counterparts
"planned" -> "live in `tnt.potential.oblate_mge`"), `components.py` and
`registry.py` ("two MGE composite types" -> "four"; `_VIEWING_ANGLES` comment
now notes the oblate pair uses `inclination`), `docs/source/potential.md`,
`docs/source/configuration.md`, `aidocs/KNOWLEDGE.md`.

### Checks re-run

`ruff check` clean; full `pytest` 329 passed (327 + the 2 new tests); strict
`sphinx-build -W` succeeds.
