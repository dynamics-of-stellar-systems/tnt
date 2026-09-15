# PR 68 audit: oblate q_min parameterization

Audit date: 2026-09-15. Reviewer: Codex, at Thomas's request.

**Recommendation: hold the merge until F1 is addressed.** The algebra and
registry integration are sound in ordinary cases, but accepted inputs can
construct a materially different intrinsic thickness at reduced precision.
One P2 correctness finding requires a defined accuracy policy and regression
coverage. No other PR-specific merge blocker was identified.

## Reviewed state and scope

- PR: [#68 — Add the q_min intrinsic-shape parameterization for oblate MGEs](https://github.com/dynamics-of-stellar-systems/tnt/pull/68).
- Branch: `oblate-q-parameterization`.
- Reviewed latest remote PR commit: `0498cc34d96f70d56993284bf74c650ce647034e`.
- Base `main`: `7429509d1fb59d45a599e17170ac78c3793e48c2`.
- After fetching origin, local `HEAD`, `origin/oblate-q-parameterization`, and
  GitHub's PR head matched the reviewed commit. A GitHub recheck after the
  full suite confirmed the same head and base.
- The existing checkout was already on the PR branch and clean, with one
  worktree. No separate audit branch was created.
- GitHub reported `MERGEABLE`, review required, and no status checks listed.
  Absence of a Git conflict does not resolve the numerical finding below.

Completed audit tasks:

- [x] Read README and relevant `aidocs/KNOWLEDGE.md` architecture, numerical,
  container, parameterization, and review-workflow guidance. No project
  `AGENTS.md`, `CLAUDE.md`, or `aidocs/INDEX.md` was present.
- [x] Review all six changed files and shared construction/reporting paths.
- [x] Check the scientific relation against the primary reference.
- [x] Run the full tests, lint, strict documentation build, configuration
  checks, and focused numerical probes.
- [x] Record current behavior and merge conditions.

This document describes current TNT behavior, without compatibility or
migration history between TNT versions. Production code was not changed.

## F1 — P2: check shape accuracy before accepting the converted inclination

**Location:** `tnt/mge.py:471–479`, especially the conversion and return at
lines 478–479 in `inclination_from_q_min`.

The new conversion guards `1 - q_min²` near zero, but does not guard loss of
shape information when `q_min` is small or the observed anchor is nearly
circular. After conversion to an inclination at the active JAX precision,
`deproject_oblate` reconstructs the intrinsic ratio by subtracting
`cos(i)²` from `q_obs²`. Those nearly equal quantities can lose enough
precision to change the requested intrinsic shape substantially. The result
can still satisfy all domain checks and become an accepted potential.

Confirmed with `Potential.from_settings`, the built component's intrinsic
axes, and `raw_potential_parameters`, for **both** `OblateLightMGEPotential`
and `OblateMassMGEPotential`:

| Precision | Observed anchor q | Requested q_min | Constructed/reported q_min | Relative error |
| --- | --- | --- | --- | --- |
| float32 | `0.76` | `0.001` | `0.00106248515658` | +6.2485% |
| float32 | `0.999999` | `0.6` | `0.613572120667` | +2.2620% |
| float64 | `0.999999999999999` | `0.6` | `0.597614304667` | −0.3976% |

The float32 inputs above are ordinary decimal declarations; percentage errors
were computed against their actual stored float32 values, so this is not merely
decimal display rounding. The probe used two untwisted Gaussian components with
the same observed axial ratio, widths `[2, 1] kpc`, and light intensities
`[2, 1] Lsun/pc²`. The mass-type probe converted that template using
`1 Msun/Lsun`. Mass rescaling by three preserves the erroneous shape, and
the recovered value can be inserted into the `AllModels` table.

**Impact:** a requested shape-search point can silently become a different
intrinsic thickness. At fixed projected intensity and width, the deprojected
central density is inversely proportional to that thickness, so this also
changes the density used to construct the potential. The claim that reporting
faithfully recovers the input currently needs a numerical qualification.

The default float64 path is accurate in the ordinary and moderately thin cases
tested. The default-precision example above is extremely close to a circular
observed shape; it is included to identify the conditioning limit, not to imply
that ordinary float64 configurations exhibit percent-level errors.

**Required outcome:** define and enforce an accuracy criterion for accepted
shape inputs. Either preserve `q_min` through a numerically suitable construction,
or reject points whose converted inclination cannot reproduce it reliably at
the active precision, using an invalid-model exception. Add independent
input-versus-built-shape checks for thin and nearly circular cases at both
precisions, covering both registered types and inverse reporting. Update the
current numerical-domain documentation accordingly. There is no requirement
to support an arbitrarily thin or nearly circular configuration; explicit
rejection is preferable to silently changing its shape.

### Minimal reproduction

Run in the documented Linux development container:

```python
import tnt  # initialize TNT before selecting the probe's JAX precision
import jax
import jax.numpy as jnp
from unxt import Quantity as Q
from tnt.mge import LightMGE
from tnt.potential import Potential, raw_potential_parameters

with jax.enable_x64(False):
    mge = LightMGE(
        I=Q(jnp.array([2.0, 1.0]), "Lsun / pc2"),
        sigma=Q(jnp.array([2.0, 1.0]), "kpc"),
        q=Q(jnp.array([0.76, 0.76]), ""),
        PA_twist=Q(jnp.array([0.0, 0.0]), "rad"),
        major_axis_pa=Q(23.0, "deg"),
    )
    settings = {"s": {
        "type": "OblateLightMGEPotential",
        "parameterization": "q_min",
        "mge": "m",
        "parameters": {"ml": {"unit": "Msun / Lsun"}},
    }}
    values = {"s": {
        "ml": Q(2.0, "Msun / Lsun"),
        "q_min": Q(0.001, ""),
    }}
    potential = Potential.from_settings(settings, values, {"m": mge}, {})
    print(potential.components["s"].deprojected.q)
    print(raw_potential_parameters(settings, potential, {})["s"]["q_min"])
    # Both report approximately 0.001062485, rather than 0.001.
```

## Other observed behavior

- The conversion is the correct rearrangement of the oblate relation in
  [Cappellari (2002), section 2.2.2, equation 9](https://arxiv.org/pdf/astro-ph/0201430).
  The minimum observed axial ratio correctly selects the limiting Gaussian.
- With anchor `q' = 0.76`, `q_min = 0.6` round-trips correctly; `q_min = q'`
  gives exactly `90 deg` and recovers the stored input at both precisions.
  A two-component probe with its anchor second also confirms anchor selection.
- Circular anchors, nonpositive inputs, and values above the observed anchor
  are rejected. Nonzero position-angle twist is rejected by the existing
  oblate deprojection validation. No implicit alternative geometry is selected.
- Some very thin inputs are explicitly rejected when the reconstructed ratio
  rounds to zero. `ModelIterator` catches `MGEDeprojectionError` as an invalid
  model, as well as `InvalidPotentialParametersError` from the adapter.
- The unchanged `deproject_oblate` can also reject a circular *non-anchor*
  component when rounding puts its intrinsic ratio slightly above one. For
  example, observed ratios `[1, 0.999999]` and `q_min = 0.6` give a first
  component ratio `1.0000000000043507` at float64. This strict-boundary behavior
  belongs to the existing native inclination path and is not counted as a
  separate PR-specific finding. It is a candidate for a focused follow-up.
- Registry dimensions, constraints, mass-parameter selection, generic inverse
  dispatch, and mass rescaling are coherent. Both types passed YAML
  `Configuration.read` checks with the new parameter name.

## Validation

| Check | Result |
| --- | --- |
| `docker compose run --rm dev pytest -q` | **454 passed**, one dependency deprecation warning; 301.99 seconds |
| `docker compose run --rm dev ruff check .` | Passed |
| Strict Sphinx build: `sphinx-build -E -b html -W docs/source /tmp/pr68-sphinx` | Passed |
| `git diff --check origin/main...HEAD` | Passed |
| Ruff format check of the four changed Python files | Three pass; an unchanged `get_projected_mass` block in `tnt/mge.py` needs formatting, also present on the base commit; not a PR blocker |
| Focused probe | 60 cases: 15 inputs × two types × two precisions; construction, inverse reporting, rescaling, and table insertion checked for successful models |
| Follow-up numerical probe | 24 cases isolating thin/near-circular conversion from circular non-anchor rejection |
| YAML configuration resolution | Passed for both registered types |

The added tests cover the ordinary inverse, edge-on inclination, registration,
domain errors, equivalence to the native path, and light-type rescaling. They
do not cover F1. Agreement with a native configuration using an inclination
returned by the same converter does not independently check shape preservation.

Configuration checks resolved YAML without loading observational data or running
a model search. No new end-to-end scientific fit or orbit integration was run.
After addressing F1, rerun the focused regressions, full suite, and strict docs
build before re-review. This audit is the only repository edit; no commit,
push, GitHub comment, approval, issue creation, or merge was performed.

Assisted by Codex (OpenAI).
