# PR 59 audit: validate potential parameter domains at runtime

Pull request: #59, `Validate potential parameter domains at runtime`

Branch: `codex/issue-30-potential-domain-validation`

Issue: #30, `Add potential parameter-schema and physical-domain validation`

Base/head reviewed: `main` at `00936d7` / PR head at `32f05f0`

## Overall judgment

The design is right: domain/value validation runs at the runtime-object
boundary (`ResolvedPotentialComponent.build`), not as a new config-prep pass;
constraint metadata hangs off the existing authoritative schema
(`NativeParameter.constraint`, `ParameterizationSpec.raw_constraints`, a
component `_constraints` ClassVar) with no parallel registry; and every
profile-specific bound is traceable to `galax`'s own documentation or code
(verified below, not just asserted).

One **Medium** finding that **blocks merge**: the new domain errors are not
caught anywhere `ModelIterator` evaluates a proposed point, so a proposed
parameter set that fails a domain check crashes the whole model-search run
instead of being recorded as one invalid `Model` -- the treatment every other
build-time failure already gets. This is not hypothetical: it demonstrably
*regresses* one case that worked before this PR (oblate inclination). Plus one
substantive Low on `_validate_parameter_values`'s structure, and two minor
Lows. No correctness bug in the physics itself.

## Architecture

- `ParameterConstraint` (NamedTuple in `tnt/potential/registry.py`): numeric
  `minimum`/`maximum` with inclusive flags, an optional comparison `unit`, and
  an optional same-component relationship (`other_parameter` + `relation`).
- `_validate_constraint_metadata` runs at module import for
  `_SUPPORTED_GALAX_TYPES`, and per registration in `register_component` /
  `register_parameterization` -- a constraint naming a parameter not in the
  schema, an incomplete relationship, a self-comparison, or empty bounds fails
  fast at import/registration, never silently.
- `ResolvedPotentialComponent` gains `raw_constraints`,
  `canonical_dimensions`, `canonical_constraints`, and `path`. `build`
  validates the raw mapping against the raw schema+constraints, runs the
  converter (if any), then validates the converter's canonical output against
  the *native* schema+constraints. `convert is None` (every MGE type, and
  native types without a parameterization) skips the second pass.
- `_validate_parameter_values` (in `tnt/potential/components.py`) enforces,
  for every parameter: exact name set, `Quantity` type, declared dimension
  (via `tnt.units.validate_dimension`), scalar shape, finiteness -- then,
  per constraint, bounds and relationships, converting compatible units only
  for the comparison.

## Physics constraints -- all grounded

Checked each added bound against the installed `galax` source, not just read
against the PR's own claims:

| type / parameter | PR constraint | `galax` basis |
|---|---|---|
| `PowerLawCutoffPotential.alpha` | `0 <= alpha < 3` | docstring: "Must satisfy: ``0 <= alpha < 3``" |
| `gNFWPotential.gamma` | `0 <= gamma < 2` | docstring: "we require $\gamma \in [0, 2)$" |
| `LeeSutoTriaxialNFWPotential` a1/a2/a3 | `a1 >= a2 >= a3 > 0` | docstring + `__check_init__` (`error_if(a1 < a2 or a2 < a3)`) |
| `StoneOstriker15Potential` r_h | `r_h > r_c` | commented-out `__check_init__`; `Phi ∝ 1/(r_h − r_c)` |
| `Vogelsberger08TriaxialNFWPotential.q1` | `0 < q1 < sqrt(3)` | `q2^2 = 3 − q1^2` must stay positive |
| oblate MGE `inclination` | `0 < i <= 90 deg` | PR #48's `deproject_oblate` guard |
| mass / length / speed / frequency amplitudes | strictly `> 0` | PR's stated policy (issue #30) |
| `MonariEtAl2016BarPotential` `alpha`, `Omega` | left signed | PR's stated policy |

`LMJ09LogarithmicPotential` q1/q2/q3 get only `> 0` (no upper bound) --
conservative and correct; these are potential-flattening parameters and can
exceed 1. The new tests hit the exact edge values (`alpha=3.0`, `gamma=2.0`,
`q1=sqrt(3)`) and assert rejection.

## Findings

### Critical / High

None.

### Medium (blocks merge): a domain-invalid proposed point crashes the run instead of being recorded as an invalid model

References:

- `tnt/model_iterator.py:450-458` (`ModelIterator._evaluate`)
- `tnt/potential/components.py:114-231` (`_validate_parameter_values`)

`_evaluate` already has an established pattern for "this proposed point
cannot become a valid `Potential`": it catches `MGEDeprojectionError`
specifically around `build_potential(...)`, logs a `WARNING`, and returns
`_invalid_potential_model(parameters)` so the search records the failure and
continues. Every domain error this PR introduces is a bare `ValueError` (or,
for a wrong Python type, `TypeError`) -- neither is caught there, so it
propagates out of `_evaluate`, out of `run()`, and takes down the entire
model-search loop for what is, semantically, exactly the same kind of failure
`MGEDeprojectionError` already handles gracefully.

This is reachable today, not just once `GridSearchParameterGenerator` or a
future sampler exists: `SinglePointParameterGenerator` echoes a component's
*declared* config values verbatim, and config-prep never checked physical
domains before this PR (that was issue #30's whole premise). A single typo'd
config value -- `ml: -5.0`, or an inclination outside `(0, 90]` -- now crashes
a run that would previously have either built successfully with a silently
wrong answer (issue #30's original complaint) or, for the inclination case
specifically, been caught and handled correctly.

**Demonstrated regression, not just a gap:** before this PR, an out-of-range
oblate `inclination` failed inside `deproject_oblate` with
`MGEDeprojectionError`, which `_evaluate` catches. This PR's
`_INCLINATION_CONSTRAINT` now runs *first*, inside `_validate_parameter_values`,
and raises a bare `ValueError` -- so the exact same bad config value that
`_evaluate` used to handle gracefully now crashes the run. The PR's own new
test (`test_oblate_mge_inclination_domain_is_checked_before_deprojection`)
confirms the new exception fires first; nothing in
`tests/unit_tests/test_model_iterator.py` exercises this through `_evaluate`,
so the regression isn't caught by the suite.

**Required change:** give the new domain errors a dedicated exception type
(e.g. `InvalidPotentialParametersError(ValueError)`, defined alongside
`ParameterConstraint`) and raise it instead of bare `ValueError` throughout
`_validate_parameter_values`. Widen `_evaluate`'s except clause to
`except (MGEDeprojectionError, InvalidPotentialParametersError) as error:`,
treating both as the same class of recoverable, per-candidate failure. Keep
`TypeError` (wrong Python type reaching `build`) uncaught -- that is a real
programming bug, not a proposed point's business, and conflating the two by
widening to a bare `except ValueError` would also swallow genuine bugs that
happen to raise `ValueError` elsewhere in the call chain. Add a test that
drives this through `ModelIterator._evaluate` (or `.run()`), not just through
`resolved.build()` directly, so the integration is what's actually verified.

### Low 1: `_validate_parameter_values` conflates two different jobs into one 117-line function

Reference: `tnt/potential/components.py:114-231`

The function does three things: (a) exact-name-set check, (b) per-value
invariants (`Quantity` type, dimension, scalar, finite), (c) per-constraint
domain checks (min/max/relationship). (c) alone is ~70 lines because the
min-bound and max-bound branches are near-identical copies (pick a comparison
operator from an inclusive flag, build a "must be `{phrase}` `{bound}`"
message), two `RuntimeError` branches (lines ~202-211) are unreachable dead
code -- `_validate_constraint_metadata` already guarantees a constraint's
`other_parameter`/`relation` are complete and refer to a real parameter, at
registration time -- and a 4-entry `comparisons` dict is rebuilt every loop
iteration just to index it once.

(a) and (b) are not actually redundant with configuration-prep's schema
check, despite checking similar things: `build` receives a `ParameterSet`,
which is *generator output*, never re-validated after config-prep produced
the original declared values. `SinglePointParameterGenerator` preserves the
invariant by construction today, but `GridSearchParameterGenerator` is
unimplemented and a future sampler-style generator (see the `prior-concept`
branch's `PriorSampler`, which builds a fresh `Quantity` per draw from a
sampled array) is exactly the kind of code where a shape or unit-wrapping bug
is plausible. (a)/(b) are best understood as a **generator/pipeline contract
check**, a different job from (c)'s **physical-domain check** -- worth
keeping, but as its own small, separately named function, not interleaved
with constraint logic.

Recommended shape:

- Move `ParameterConstraint`'s bound/relationship logic onto the type itself,
  e.g. `ParameterConstraint.violation(value, siblings) -> str | None`
  returning a human-readable reason or `None`. `_validate_parameter_values`'s
  constraint loop then collapses to a few lines, and the semantics live next
  to the data that defines them (co-locate in `tnt/potential/registry.py`,
  where `ParameterConstraint` is already defined -- the function currently
  in `components.py` has no component-specific knowledge).
- Keep the name-set/type/scalar/finite check as its own small function
  (e.g. `_check_parameter_set_contract`), clearly documented as defending
  against a generator/rescaling bug rather than user config.
- Both should raise the new `InvalidPotentialParametersError` from the Medium
  finding, not bare `ValueError`/`TypeError` (except genuinely-a-bug cases).

Not a correctness bug -- purely a maintainability/readability finding -- but
substantive enough to fix alongside the Medium finding above, since both
touch the same function and the same new exception type.

### Low 2: `build` is now un-`jit`-able

`_validate_parameter_values` calls `float(...)` and `math.isfinite(...)` on
parameter values, so `ResolvedPotentialComponent.build` can no longer run
inside a `jax.jit`/`vmap` trace. Nothing calls it in a traced context today
(`ModelIterator._evaluate` builds the potential, then hands it to the
not-yet-implemented orbit integration), so there is no live problem. Worth a
one-line note in `build`'s docstring so whoever wires orbit-library
evaluation keeps `build` outside the jitted region.

### Low 3: `ruff format` would reformat 3 of the changed files

`tnt/potential/{components,registry,oblate_mge,triaxial_mge}.py` -- a few
manually wrapped `ParameterConstraint(...)` calls and boolean expressions
that fit on one line. CI runs neither `ruff format` nor pre-commit (only
`pytest`), and ~14 pre-existing files fail the same check, so this does not
gate merge; a `ruff format` pass over the touched files would be tidy, and
worth doing anyway if Low 1's refactor touches these files regardless.

## Issue #30 point 1 (the q/p/u carve-out)

The issue anticipated needing a carve-out so `triaxial_light_mge` /
`triaxial_mass_mge` components declaring staged `q`/`p`/`u` parameters
wouldn't be rejected as "unexpected". This is now moot: #52 made
`_validate_potential` enforce the exact key set for every registered type
(MGE types included), the `(p, q, u)` viewing-geometry parameterization is
still unregistered, and no shipped or test config declares those keys -- they
use `theta`/`phi`/`psi` directly. The PR correctly adds no carve-out.

## Test coverage assessment

12 new unit tests cover the constraint machinery thoroughly at the
`resolved.build()` level: nonpositive/nonfinite mass, exact-name/scalar
enforcement, every analytic bound at its exact edge value, same-component
relationships across declared units, NFW raw-before/converted-after,
registration-time rejection of bad constraint metadata, and a schema/registry
consistency check. This is solid coverage of the validation logic itself.

What's missing is coverage of the *consequence* of a validation failure one
layer up, in `ModelIterator._evaluate` -- see the Medium finding. All of the
new tests call `resolved.build(...)` or `AbstractPotentialComponent.resolve`
directly; none exercise `_evaluate`/`run()` with a domain-invalid proposed
point, so the gap (and the oblate-inclination regression specifically) has no
test that would fail today and should once the fix lands.

## Checks run

From the PR head on macOS (Apple Silicon):

- full suite: **368 passed**, one warning (the pre-existing dependency-owned
  TF-Probability / JAX deprecation);
- `ruff check .`: passed;
- `ruff format --check`: flags 3 changed files (Low 3);
- strict `sphinx-build -E -b html -W`: passed;
- `git diff --check main...HEAD`: passed.

## Recommended merge path

1. **Required:** introduce `InvalidPotentialParametersError` (or similarly
   named), raise it from the new validation code, and catch it alongside
   `MGEDeprojectionError` in `ModelIterator._evaluate`. Add a test that
   exercises this through `_evaluate`/`run()`, including the oblate
   inclination case, to close the untested regression.
2. **Recommended, same pass:** split `_validate_parameter_values` into a
   constraint check living on `ParameterConstraint` (relocated to
   `registry.py`) and a separately named generator-contract check, per Low 1
   -- the required exception-type change touches this function anyway.
3. Optional: `build`'s docstring note (Low 2); `ruff format` the touched
   files (Low 3).
4. Re-run the full suite, `ruff check`, and strict Sphinx.
5. Obtain the required GitHub review approval and squash-merge PR 59.
6. Remove this audit document in its own commit before merge.

## Decision required

Whether Low 1's refactor happens in this PR (bundled with the Medium fix,
since both touch `_validate_parameter_values`) or as an immediate fast-follow
-- either is reasonable, but the Medium fix itself is not optional.

## Resolution update

All findings were addressed on the PR branch:

- The Medium finding now has a dedicated
  `InvalidPotentialParametersError`, which `ModelIterator._evaluate()` records
  as an invalid potential candidate. An end-to-end oblate-inclination
  regression covers this path, while a non-`Quantity` programming error still
  propagates.
- Low 1 was resolved by separating the parameter-set contract check from
  physical-domain validation and moving bound/relationship evaluation onto
  `ParameterConstraint.violation()`.
- Low 2 was resolved by documenting that potential construction is an eager
  Python boundary outside `jax.jit`/`jax.vmap` and that only the constructed
  potential enters compiled numerical work.
- Low 3 was resolved by formatting the Python files changed by the PR. The
  remaining current-state wording in `NNLSWeightSolver` was also clarified.

Linux validation through the Colima development container completed with
**370 tests passed** (one dependency-owned JAX deprecation warning),
`ruff check .` passed, and strict Sphinx
(`sphinx-build -E -b html -W docs/source docs/build/html`) passed.
