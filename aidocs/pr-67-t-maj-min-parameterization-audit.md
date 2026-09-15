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
