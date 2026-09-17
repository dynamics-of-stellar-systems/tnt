# PR 67 audit: T-major/minor parameterization

Audit date: 2026-09-15. Reviewer: Codex, at Thomas's request.

**Recommendation: do not merge yet.** Two correctness findings need addressing:
accepted interior inputs can construct a materially different viewing geometry,
and the conversion accepts an exactly zero intrinsic minor-axis ratio.

## Reviewed repository state

- PR: [#67 — Add the (T, T_maj, T_min) triaxial shape parameterization](https://github.com/dynamics-of-stellar-systems/tnt/pull/67).
- Reviewed branch: `T-maj-min-parameterization`.
- Reviewed latest remote PR commit: `c7f3e97204de1254b7b10a941c0ca4466935af69`.
- Base: `main`, commit `7429509d1fb59d45a599e17170ac78c3793e48c2`.
- After fetching origin, local `HEAD`, `origin/T-maj-min-parameterization`,
  and GitHub's PR head all matched the reviewed commit. GitHub's head was
  checked again after the full test suite; it was unchanged.
- The existing checkout was clean before the audit, with one worktree and no
  separate audit branch. This audit file is the only review edit.
- GitHub reports the PR as mergeable, with review required and no status checks
  listed. That establishes absence of a Git merge conflict, not correctness.

## Scope and completed work

- [x] Read README and relevant `aidocs/KNOWLEDGE.md` architecture, numerical
  behavior, container instructions, and human review workflow. No project
  `AGENTS.md`, `CLAUDE.md`, or `aidocs/INDEX.md` was present.
- [x] Review all six changed files and the shared construction, inversion,
  deprojection, and reporting paths used by the new registration.
- [x] Check the algebra against the cited primary scientific reference.
- [x] Run the full suite, lint, strict documentation build, and focused probes.
- [x] Record findings and conditions for re-review.

This review describes current TNT behavior; it does not prescribe compatibility
or migration behavior between TNT versions. Production code was not changed.

## Findings requiring changes

### F1 — P1: preserve the requested shape coordinates when converting to angles

**Location:** `tnt/mge.py:737–739`, where the new conversion delegates to
`triaxial_viewing_angles`; the inherited clamp is at lines 655–664.

The forward conversion computes `(p, q, u)` and then clamps `u` a distance of
`4*sqrt(eps)` away from the admissible interval's boundaries. The new shape
coordinates divide by `1-p²` and `p²-q²`, so a small absolute adjustment to `u`
can produce a large adjustment to `T_maj` or `T_min`. This occurs for valid
inputs strictly inside the parameter cube, not only declared endpoint values.
The existing numerical-domain guards allow these models to build successfully.

Confirmed through `Potential.from_settings` and `raw_potential_parameters` for
both `TriaxialLightMGEPotential` and `TriaxialMassMGEPotential`, using a single
Gaussian with observed axis ratio `q' = 0.76` and anchor twist `0.2 rad`:

| Precision | Requested `(T, T_maj, T_min)` | Recovered value | Viewing-angle error |
| --- | --- | --- | --- |
| float64 | `(0.000001, 0.1, 0.2)` | `T_maj = 0.225775202831` | `phi`: −9.934695° |
| float64 | `(0.0000005, 0.1, 0.2)` | `T_maj = 0.451550365594` | `phi`: −23.784730° |
| float64 | `(0.999999, 0.2, 0.9)` | `T_min = 0.869592368579` | `psi`: −4.190773° |
| float32 | `(0.02, 0.1, 0.2)` | `T_maj = 0.262307167053` | `phi`: −12.392547° |

Angle errors compare the constructed model with the direct reference equations,
including the anchor-twist adjustment. For the first row, the algebra gives
`u = 0.9999999736000044`; the clamp instead evaluates approximately
`u = 0.9999999403953552`. The difference is small in `u` but substantial in
the requested coordinates and viewing geometry. In contrast, the ordinary
interior point `(0.43, 0.49, 0.39)` agrees with the reference to approximately
`3e-14` degrees at float64.

**Impact:** a shape/viewing-geometry search evaluates a different parameter
point from the one requested; inverse reporting exposes the changed point.
Calling the map bijective or describing only an endpoint nudge does not explain
this behavior. Agreement with the PQU conversion alone cannot catch the error,
because that is the same clamping path.

**Required outcome:** accepted points must preserve the declared coordinates
and geometry to a stated, tested accuracy. Use a numerically suitable angle
conversion, or reject points that cannot be represented accurately with an
explicit invalid-model error. Do not silently construct a substantially
different point. Add independent reference-angle and input/output checks near
both axisymmetric limits at float64 and float32, for both registered types.
Align the numerical-limit documentation with the resulting behavior.

### F2 — P2: reject the zero-thickness boundary before deprojection

**Location:** `tnt/mge.py:166–174`, especially the inclusive `q2 >= 0` check.

For observed `q' = 0.5` and `(T, T_maj, T_min) = (0.5, 0.5, 0.375)`, the
conversion denominator is exactly `0.75` and `q2` is exactly zero. The new
helper accepts this point and returns `q = 0`, despite TNT's intrinsic-axis
requirement `0 < q <= p <= 1`. This path does not pass through the PQU registry's
strictly positive raw-`q` constraint.

In the focused float64 probe, both potential types successfully construct the
model, with deprojection rounding the intrinsic `q` to
`1.0536712127723509e-08`. At float32, the same point instead raises
`MGEDeprojectionError` because the recovered `q` remains zero. Thus the
physical-domain decision depends on downstream rounding. A vanishing-thickness
input can become an accepted, extremely thin Gaussian, whose density includes
division by the intrinsic axis ratio.

**Required outcome:** enforce strictly positive `q2` in the forward conversion
before taking square roots or computing viewing angles. Add an exact-boundary
regression for this input at both precisions and for both registered types;
the configuration-driven build should reject it as an invalid model. Retain
coverage for valid positive-thickness points and the already rejected
negative-`q2` region.

## Reproduction

Run in the documented Linux development container. This minimal probe prints
the two float64 findings for the light type; the audit also ran the mass type
and both JAX precision modes.

```python
import tnt  # initialize the TNT runtime
import jax
import jax.numpy as jnp
from unxt import Quantity as Q
from tnt.mge import LightMGE
from tnt.potential import Potential, raw_potential_parameters

with jax.enable_x64(True):
    for observed_q, shape in [
        (0.76, (1e-6, 0.1, 0.2)),
        (0.5, (0.5, 0.5, 0.375)),
    ]:
        mge = LightMGE(
            I=Q(jnp.array([1.0]), "Lsun / pc2"),
            sigma=Q(jnp.array([1.0]), "kpc"),
            q=Q(jnp.array([observed_q]), ""),
            PA_twist=Q(jnp.array([0.2]), "rad"),
            major_axis_pa=Q(0.0, "deg"),
        )
        settings = {"s": {
            "type": "TriaxialLightMGEPotential",
            "parameterization": "T_maj_min",
            "mge": "m",
            "parameters": {"ml": {"unit": "Msun / Lsun"}},
        }}
        raw = {"ml": Q(1.0, "Msun / Lsun")}
        raw.update({k: Q(v, "") for k, v in zip(
            ("T", "T_maj", "T_min"), shape
        )})
        potential = Potential.from_settings(
            settings, {"s": raw}, {"m": mge}, {}
        )
        print("requested:", shape)
        print("recovered:", raw_potential_parameters(settings, potential, {}))
        print("intrinsic q:", potential.components["s"].deprojected.q)
```

## Validation and limits

| Check | Result |
| --- | --- |
| `docker compose run --rm dev pytest -q` | **457 passed**, one dependency deprecation warning, 350.01 seconds |
| `docker compose run --rm dev ruff check .` | Passed |
| Strict Sphinx build, `sphinx-build -E -b html -W docs/source /tmp/pr67-sphinx` | Passed |
| `git diff --check origin/main...HEAD` | Passed |
| Ruff format check of the four changed Python files | Three pass; `tnt/mge.py` fails on an unchanged `get_projected_mass` formatting block also present in the base; not a PR-specific blocker |
| Focused numerical probe | 40 cases: ten inputs × two potential types × two precisions; findings above reproduced |

For successfully built probe models, raw-parameter recovery, mass rescaling by
three, and insertion of recovered parameters into an `AllModels` table also
completed. Invalid cube points and exact `T = 0` / `T = 1` cases were rejected.
The probe checks construction and reporting, not an orbit integration or a
complete scientific fit. No new end-to-end YAML configuration run was performed.

The forward/inverse scalar algebra matches equations 4 and 7 of
[Quenneville, Liepold & Ma, ApJ 926:30, section 3](https://arxiv.org/pdf/2111.06904v2).
Equation 8 supplies the independent angle reference; equation 6 excludes the
zero-thickness boundary. The changed tests chiefly check algebraic round trips
and agreement with the shared PQU path; they do not cover the two findings.

No additional blocker was found in the registry adapters or their mass-parameter
handling. Re-review should verify F1 and F2 with independent regression checks,
then rerun the suite and documentation build. No commit, push, GitHub comment,
approval, or merge was performed as part of this audit.

Assisted by Codex (OpenAI).

## Response (Claude, 2026-09-17)

Reviewed independently before fixing: confirmed F1 bit-for-bit (requesting
`(1e-6, 0.1, 0.2)` at `q' = 0.76` recovers `T_maj = 0.225775202831`, matching
the table above exactly). F2's mechanism was also confirmed, though the
specific outcome differs by environment: at `q_obs = 0.5`,
`(T, T_maj, T_min) = (0.5, 0.5, 0.375)` gives `q^2` that is bit-exact `0.0` in
`_p_q_u_from_T_Tmaj_Tmin`'s own arithmetic (`0.75 / 0.75 == 1.0` exactly in
IEEE754), so whether it is later caught by `deproject_triaxial`'s general
`0 < q` check depends on libm rounding in the unrelated `acos`/`atan` round
trip through `theta`/`phi`/`psi` -- outside this repo's documented Linux dev
container it was in fact caught there. That is not a rebuttal of F2; it is
further evidence that acceptance of this input was never a deliberate
guarantee, only an accident of platform floating-point rounding -- exactly
why the fix belongs at the source rather than downstream.

**Fixed:**

* **F2** -- `_p_q_u_from_T_Tmaj_Tmin` (`tnt/mge.py`) now requires `q^2 > 0`
  strictly (previously `>= 0`), rejecting the zero-thickness boundary
  deterministically instead of relying on downstream rounding.
* **F1** -- `AbstractMGE.viewing_angles_from_T_Tmaj_Tmin` now round-trips its
  result back through `T_Tmaj_Tmin_from_p_q_u` and raises
  `MGEDeprojectionError` if the recovered `(T, T_maj, T_min)` drifts from the
  requested point by more than `_TMAJMIN_ROUNDTRIP_TOL_FACTOR * sqrt(eps)`
  (`= 100 * sqrt(eps)`) -- catching exactly the cases where
  `triaxial_viewing_angles`'s own eps-margin clamp on `u` gets amplified
  through the `(T, T_maj, T_min)` reparameterization's denominators
  (`1 - p**2`, `p**2 - q**2`).

**Verification:**

* Both of the audit's exact reproduction cases now raise cleanly
  (`InvalidPotentialParametersError` at the config-build layer).
* Reproduced the audit's own float32 finding bit-for-bit
  (`T_maj = 0.262307...`) as the drift the new round-trip check flags.
* Ordinary interior points -- including the audit's own point agreeing to
  `~3e-14` degrees -- still build normally at both float64 and float32; the
  new check does not over-reject.
* 6 new regression tests added, covering both findings at the pure-function,
  `AbstractMGE`, and full config-build layers, plus "adjacent point still
  works" / "ordinary interior point still works" controls
  (`tests/unit_tests/test_mge.py`, `tests/unit_tests/test_potential.py`).
* Full `test_mge.py` + `test_potential.py` suite: 199 passed (193 existing +
  6 new), no regressions. `ruff check` clean on all changed files.

No change was made to the registered `T`/`T_maj`/`T_min` `ParameterConstraint`
bounds, the registry adapters, or the `pqu` parameterization's own behavior.

Assisted by Claude (Anthropic).

## Re-audit — Codex, 2026-09-17

**Recommendation: not ready to merge. F2 is resolved; F1 is only partially
resolved.** The original reproductions now reject correctly, but the new
float32 tolerance still admits substantial changes to the requested geometry.

### Exact reviewed state

This re-audit reviews the **latest remote PR 67 head**,
`3feec33dfaba9c6dea27d913209564d9d996c279` (`Fix PR-67 audit findings F1/F2 in
the T_maj_min parameterization`), on `T-maj-min-parameterization`. After
`git fetch origin`, the existing local checkout's `HEAD`,
`origin/T-maj-min-parameterization`, and GitHub's PR head all matched that
commit. The checkout was clean. No separate audit branch was used.
The base remains `7429509d1fb59d45a599e17170ac78c3793e48c2` on `main`.
GitHub's head and base were checked again after the full suite and were
unchanged. The audit amendment itself is not part of that reviewed commit.

Completed: read the author's response, review the fix and added tests, rerun
the complete suite and documentation checks, reproduce both findings for both
potential types at both precisions, and test nearby accepted points against
the independent angle equations used in the original audit.

### F2 resolved: explicit zero-thickness rejection

The strict `q2 > 0` check rejects the exact input
`q' = 0.5`, `(T, T_maj, T_min) = (0.5, 0.5, 0.375)` before angle conversion.
Both registered types raise `InvalidPotentialParametersError` at float64 and
float32. The positive-thickness control `(0.5, 0.5, 0.374)` builds and recovers
the declared coordinates accurately at both precisions. This closes F2.

### F1 remains open: float32 accuracy threshold admits degree-scale errors

**Location:** `tnt/mge.py:770–773`, using
`_TMAJMIN_ROUNDTRIP_TOL_FACTOR = 100` defined at line 92.

The round-trip guard is the right kind of check, and it rejects all four
original F1 examples at the precision where their inaccurate construction was
reported. However, `100 * sqrt(eps)` is an **absolute per-coordinate** tolerance
of approximately `1.49e-6` at float64 and **`0.0345267` at float32**. The latter
allows a change of 3.45 percentage points of the entire coordinate range. It
is not a small relative-error limit on the requested coordinate.

The following float32 points all build successfully through
`Potential.from_settings` in **both** light and mass potential types, using
observed `q' = 0.76` and anchor twist `0.2 rad`:

| Requested `(T, T_maj, T_min)` | Reported `T_maj` | Absolute `T_maj` drift | `phi` error |
| --- | --- | --- | --- |
| `(0.1, 0.02, 0.2)` | `0.0535070971` | `0.0335071` | −5.302959° |
| `(0.1, 0.03, 0.2)` | `0.0534717366` | `0.0234717` | −3.432868° |
| `(0.05, 0.08, 0.2)` | `0.1055563763` | `0.0255564` | −2.541780° |

Errors were calculated against the actual stored input values, using the
paper's equation 8 and the anchor-twist adjustment, not against the clamped
PQU conversion. In the first row, `T_maj` increases by approximately 168%.
The largest coordinate drift remains below `0.0345267`, so the new guard
accepts the model. Raw reporting returns the changed coordinates, and mass
rescaling preserves them. At float64 these same three points agree with the
reference to approximately `2e-13` degrees. The remaining demonstrated defect
therefore concerns supported reduced-precision operation, not these ordinary
points under TNT's default float64 setting.

**Required before merge:** tighten or otherwise redesign the accepted-error
criterion so the three examples above either preserve the requested geometry
or explicitly reject it as numerically unreliable. Check the resulting policy
with independent coordinate/angle regressions at both precisions and for both
registered types, including accepted and rejected controls. A blanket absolute
allowance of `0.0345` does not close the original shape-preservation finding.
Document the accepted accuracy and rejection policy in the current
parameterization documentation; `aidocs/KNOWLEDGE.md` and
`docs/source/potential.md` currently do not describe the new round-trip guard.

The six added regressions run at default precision and cover the original
examples. They do not exercise these float32 points below the new threshold.

### Reproduction of the remaining F1 behavior

Run the original audit's `Potential.from_settings` reproduction with
`jax.enable_x64(False)`, observed `q = 0.76`, twist `0.2 rad`, and raw shape
`(0.1, 0.02, 0.2)`. A compact reproduction at the shared MGE conversion layer is:

```python
import tnt
import jax
import jax.numpy as jnp
from unxt import Quantity as Q
from tnt.mge import LightMGE

with jax.enable_x64(False):
    mge = LightMGE(
        I=Q(jnp.array([1.0]), "Lsun / pc2"),
        sigma=Q(jnp.array([1.0]), "kpc"),
        q=Q(jnp.array([0.76]), ""),
        PA_twist=Q(jnp.array([0.2]), "rad"),
        major_axis_pa=Q(0.0, "deg"),
    )
    requested = tuple(float(Q(v, "").ustrip("")) for v in (0.1, 0.02, 0.2))
    angles = mge.viewing_angles_from_T_Tmaj_Tmin(*requested)
    print(mge.T_Tmaj_Tmin_from_viewing_angles(*angles))
    # Approximately (0.100000583, 0.053507097, 0.197849512), without an error.
```

### Re-audit validation

| Check | Result |
| --- | --- |
| Full Linux suite: `docker compose run --rm dev pytest -q` | **463 passed**, one dependency deprecation warning, 189.60 seconds |
| `docker compose run --rm dev ruff check .` | Passed |
| Strict Sphinx: `sphinx-build -E -b html -W docs/source /tmp/pr67-reaudit-sphinx` | Passed |
| `git diff --check origin/main...HEAD` | Passed |
| Focused probe | 40 cases: ten inputs × two types × two precisions; original reproductions, ordinary/positive-thickness controls, and three tolerance-gap cases |

The probe checked full potential construction, inverse parameter reporting,
independent reference angles, and mass rescaling for successfully built models.
It did not run an orbit integration, scientific fit, or a new end-to-end YAML
configuration session. No other PR-specific blocker was found in the fix.

The existing Colima VM was initially stopped and was started for validation,
after backing up its configuration. Tests and the numerical probe ran
sequentially. Only this audit document was edited in the repository; no commit,
push, GitHub comment, approval, or merge was performed during this re-audit.

Assisted by Codex (OpenAI).

## Response 2 (Claude, 2026-09-17)

Reproduced the re-audit's three flagged points independently before fixing,
in an isolated worktree of this exact commit: `viewing_angles_from_T_Tmaj_Tmin`
followed by `T_Tmaj_Tmin_from_viewing_angles` at float32 reproduced the
reported numbers essentially bit-for-bit (e.g. `(0.1, 0.02, 0.2)` ->
`(0.10000058, 0.05350710, 0.19784951)`). The root cause is exactly as
diagnosed: `_TMAJMIN_ROUNDTRIP_TOL_FACTOR * sqrt(eps)` was an **absolute**
per-coordinate tolerance (`~0.0345` at float32), which is blind to a small
requested coordinate -- the same class of mistake already caught and fixed
differently for PR #68's `q_min` check, which this PR's original response
should have applied here too but didn't.

**Fixed:** `viewing_angles_from_T_Tmaj_Tmin` (`tnt/mge.py`) now checks each
coordinate against a combined `atol + rtol * |target|` bound
(`_TMAJMIN_ROUNDTRIP_ABS_TOL_FACTOR * eps + _TMAJMIN_ROUNDTRIP_REL_TOL_FACTOR
* sqrt(eps) * |coordinate|`), not a single absolute bound. A combined
bound rather than relative-only, because `T`/`T_maj`/`T_min` are inclusively
bounded in `[0, 1]` and a requested coordinate can legitimately be exactly
`0`, where a purely relative test is meaningless.

The two constants were calibrated against measured round-trip drift, not
guessed: swept ordinary points (including small `T_maj`/`T_min` values) plus
the flagged bad ones across both precisions. An ordinary interior point's
drift stays below `~6e-6` relative and `~6e-7` absolute at float32 (and far
tighter at float64), while every flagged bad case (including the re-audit's
new ones) measured `32%-168%` relative -- `_TMAJMIN_ROUNDTRIP_REL_TOL_FACTOR
= 50` (giving `~1.7%` relative tolerance at float32) sits comfortably
between the two, with the same numeric value already used for PR #68's
`q_min` check for consistency.

**Verification:**

* All three of the re-audit's newly flagged points, plus the original F1
  case, now raise `MGEDeprojectionError` at float32.
* The same three re-audit points still build normally at float64 -- matching
  the re-audit's own finding that they agree to `~1e-13` there and are not
  actually a problem at TNT's default precision.
* Ordinary interior points (including small-`T_maj`/`T_min` cases like
  `(0.1, 0.1, 0.2)`) still build normally at both precisions.
* 3 new regression tests added (parametrized over the re-audit's three
  points, each checked at both precisions), alongside the existing 6 from
  the first response.
* Full `test_mge.py` + `test_potential.py` suite: 202 passed (199 + 3 new),
  no regressions. `ruff check` clean.

Also done, per the re-audit's own request: documented the round-trip guard's
accepted-accuracy/rejection policy in both `aidocs/KNOWLEDGE.md` and
`docs/source/potential.md`.

Assisted by Claude (Anthropic).
