# PR #61: prior-concept audit

Date: 2026-10-08

PR: https://github.com/dynamics-of-stellar-systems/tnt/pull/61

Reviewed head: `e7ad359dfe304efd70dd89fbb87f701c00dcecb3`.
Reviewed main: `3f202994fe384f39cc5639f499fd99b577f6d017`.
The head includes that main commit; the review does not require reconciling
an older base. The checkout was clean before this audit.

## Recommendation

**Request changes before merging.** The design is coherent, but repeated
sampling, silent acceptance of malformed distributions, and silently
returning a severely divergent sampling run are confirmed merge blockers.
There are also invalid-potential handling, historical resume compatibility,
and documentation issues. All 713 existing tests pass; these findings expose
gaps in their coverage rather than failures in the existing suite.

The user confirmed that this PR is for candidate sampling only. Custom
constraints affecting final chi-squared scores or model ranking are outside
this review's merge requirements.

## Design assessment

- Keeping a composed `Prior` independent of parameter generators is useful:
  other search methods can consume the same declarations later.
- Parameter distributions plus factor-only plugins express both independent
  ranges and relationships between components without widening the potential
  parameterization interface.
- Samples are restored to quantities in their declared units. The potential
  builder reuses the existing traceable construction boundary.
- Keeping prior/search changes outside resume-critical model structure is
  consistent with the existing model-search compatibility policy.
- The PR intentionally does not reject invalid geometry unless a plugin calls
  `build_potential()`. That stated limitation is not, by itself, an audit
  finding. Safe handling when a plugin does construct a potential still needs
  to respect the construction validity contract.

## Merge findings

### F1 — P1: every iteration repeats the same sampled candidates

Location: `tnt/parameter_generator.py:268-274`.

`PriorSampler._propose_free_parameters()` discards `all_models` and constructs
`jax.random.PRNGKey(self.seed)` on every call. Sampling is deterministic for a
given key, so later iterations and resumed runs repeat the first batch, on
both the independent-sampling and factor-sampling paths. The iterator has no
duplicate filter: it evaluates and appends these points again. With the
normal improvement threshold, the repeated batch can stop the search; with
improvement stopping disabled, it wastes the remaining search allowance.

**Reproduction:** a four-candidate `Uniform(1, 9)` prior with seed 0 returned
the following on two consecutive calls:

```text
batch 0: [6.36954389455336, 8.181303812093011, 5.594834985545665, 8.49433948698086]
batch 1: [6.36954389455336, 8.181303812093011, 5.594834985545665, 8.49433948698086]
```

**Required change:** derive a distinct deterministic key per cumulative
iteration, for example by folding `all_models.n_iterations()` into the
configured seed, or persist and restore an explicit random-stream state.
Conditioning on fitted scores is not needed to use the iteration counter.
Add a regression that advances model history and checks that the next batch
differs, remains reproducible, and matches an uninterrupted run after resume.
The reproduction also confirmed the repeated batch after advancing a real
`AllModels` history to one completed iteration.

### F2 — P1: invalid distribution arguments silently produce bogus draws

Location: `tnt/priors.py:253-257`; configuration structure is checked in
`tnt/configuration/validation.py:527-544`.

Structural validation deliberately defers distribution semantics, but the
runtime constructor does not enable NumPyro's argument validation either.
With the locked dependency versions, malformed priors can successfully return
candidates rather than raise an error. This is a regression from the previous
explicit lower/upper-bound checks, and violates TNT's preference for hard
errors over silent degradation.

**Reproduction:** all these declarations pass `_validate_prior()` and
`Prior.sample(..., num_samples=4)`:

| Declaration | Actual result |
| --- | --- |
| `Uniform(9, 1)` | Four identical values of `9.0` |
| `Uniform(1, 1)` | Four identical values of `1.0` |
| `LogUniform(-1, 9)` | Four NaNs |
| `Normal(5, -1)` | Four finite draws, despite the invalid negative scale |

**Required change:** resolve and validate parameter distributions at the
runtime setup boundary, before sampling or publishing a run. Check the
resolved object is a supported distribution, use argument validation, and
reject invalid or unsupported scalar distributions with the configuration
path in the error. NumPyro marks Uniform's bounds as dependent constraints,
so `validate_args=True` alone is insufficient: explicitly require ordered,
distinct bounds, and positive bounds for LogUniform. Add regression coverage for reversed/equal Uniform bounds,
non-positive LogUniform bounds, and invalid scales. Preparation can remain
structural; it need not begin scientific execution.

### F3 — P1: a severely divergent chain is returned as successful sampling

Location: `tnt/priors.py:367-374`; the narrow-region acceptance test is
`tests/unit_tests/test_priors.py:371-411`.

Markov chain Monte Carlo (MCMC) draws candidates through a simulated chain.
The No-U-Turn Sampler (NUTS) reports divergences when its numerical trajectory
integration becomes unreliable. This is a sampling diagnostic, not a count of
invalid final geometries, and a high rate makes the claimed target-distribution
sampling unreliable without further investigation.

`Prior.sample()` discards those diagnostics and returns `get_samples()`
unconditionally. The PR's narrow `(p, q, u)` example passes its validity and
mean checks even though reproducing that same model and seed with diagnostics
exposed yielded:

```text
num_warmup = 1000; num_samples = 1000; seed = 0
divergences = 985 / 1000
mean_accept_prob = 0.7835490732205694
mean_num_steps = 14.319
unique_p_samples = 923
```

The chain moves and can return physically valid points, but those properties
do not establish correct sampling. The issue is not hypothetical: 98.5% of
the retained transitions in the showcased regression are divergent.

**Required change:** expose and check sampler diagnostics with an explicit
failure/reporting policy before accepting a batch as usable. Investigate this
example's hard geometry boundary and invalid derived factors (F4); revise the
sampling method or supported model contract explicitly as needed. Do not add
an implicit fallback sampler. Strengthen the regression beyond the current
mean/validity checks, and qualify the documentation's robustness claim until
the sampling behavior is verified.

### F4 — P2: the validity factor does not guard derived potential evaluation

Location: `tnt/priors.py:262-269`; the documented consumer is
`docs/source/priors.md:101-110`.

The builder registers a `-inf` factor when construction is invalid, then
returns the invalid placeholder potential and hides its validity flag. The
plugin proceeds directly through `to_galax()` and enclosed-mass evaluation.
That violates the existing `Potential.build_with_validity()` contract:
numerical use must be conditional on a true flag, including use through
`to_galax()`.

**Reproduction:** using the PR's own enclosed-mass plugin and PQU fixture at
`p=0.1, q=0.9, u=0.2, dh.m=1e12`, NumPyro's `log_density()` reports:

```text
_valid_potential_geometry factor: -inf
m_within_10kpc factor: nan
complete model log density: nan
```

The rejection factor does not make the complete log density `-inf` when
another factor is NaN. The tested gradient with respect to `p` was zero;
this reproduction demonstrates an undefined density, not an observed
non-finite gradient. NUTS may reject that trajectory, but this is the old
NaN-based rejection behavior the PR says it has replaced.

**Required change:** provide a supported way to guard derived calculations
with the construction flag. For example, expose validity and evaluate derived
quantities with JAX conditional execution, using a finite neutral soft factor
on the invalid branch while retaining the separate `-inf` support factor.
Add a regression checking the complete invalid-point density is `-inf` and
that invalid placeholders are not numerically evaluated. Unconditional
geometry rejection without a plugin can remain deferred.

### F5 — P2: historical search settings become falsely resume-critical

Location: `tnt/configuration/compatibility.py:19-25`.

The PR replaces `generator_settings` with `prior` in the set of parameter
fields excluded from the scientific schema. Historical archived configurations
still contain the old field. `ensure_resume_compatible()` reads those archived
files and applies the current projection without preparing or migrating them,
so their search controls now appear to be model-schema fields. A user who
updates their profile to the new syntax cannot resume an otherwise unchanged
model set.

**Reproduction:** applying the actual schema projector and comparator to the
integration fixture on reviewed `main` versus the PR fixture reports seven
differences, all ending in `.generator_settings`: the black-hole mass, both
halo parameters, and stellar `ml`, `theta`, `phi`, and `psi`. These are exactly
the search metadata removed by this PR, not scientific schema changes.

**Required change:** keep both `generator_settings` and `prior` excluded when
projecting historical configurations. New user-profile preparation can still
reject the deprecated field. Add a regression resuming from an archived
pre-PR configuration after converting its search controls to priors.

### F6 — P2: existing documentation examples no longer pass preparation

Locations: `docs/source/configuration.md:157-163` and
`docs/source/potential.md:247-289`.

The packaged parameter default remains `fixed: false`, and this PR makes a
prior mandatory for that state. The named-registry example and all five MGE
potential examples still declare bare values with neither `fixed: true` nor a
prior. Their parameter blocks are rejected by the new validation. A successful
Sphinx build only verifies documentation rendering, not these examples.
`configuration.md` also still says preparation checks generic search-parameter
bounds, although the corresponding bound validator was removed.

**Required change:** make each example's intended fixed/search behavior
explicit and update the validation description. Give the Priors page a
complete `PriorSampler` settings example, including required `seed` and
`num_warmup`, so users can enable the new generator without reading its code.
The direct configuration-read reproduction rejects bare parameter values with
`potential.stars.parameters.ml is not fixed and so must declare a prior.`

## Additional review notes

- The factor-only/read-only plugin contract is an instruction to plugin
  authors, not an enforced boundary. The loader checks callability and arity;
  `_build_model()` invokes the callable directly and exposes mutable nested
  dictionaries. A plugin that only calls `factor` can nevertheless overwrite
  a candidate value. A reproduction changed a candidate mass to 99 for its
  factor while `Prior.sample()` still returned the original stochastic sample
  values. Clarify the guarantee in the docs or enforce the contract if safe
  composition is meant to be guaranteed by the framework. This is a plugin
  correctness concern, not a request for a security sandbox.
- Factor names share the model's global site namespace. Independent plugins
  must coordinate their names, and the internal `_valid_potential_geometry`
  name is also present. Automatic scoping by configured plugin name would make
  the advertised composition less fragile.
- The public PR description's claim to replace `ExternalChi2` needs the
  candidate-only qualification confirmed by the user: these factors do not
  modify chi-squared scores or the ranking of already evaluated models.
- A saved run currently records prior declarations and plugin paths, but not
  plugin source or a content hash; the dependency manifest also omits NumPyro.
  These are gaps in recording how candidates were produced. Consider capturing
  that information alongside a future explicit sampler-state policy.
- Full orbit integration and weight solving remain existing scaffolds. This
  audit cannot certify an end-to-end physical fit that the repository does
  not yet implement.

## Validation

Environment: Linux `x86_64`, Python 3.12.13, JAX 0.10.2, NumPyro 0.21.0,
using the rebuilt Colima development image and frozen project lockfile.

- `docker --context colima compose build`: passed.
- `docker --context colima compose run --rm dev ruff check .`: passed.
- Strict Sphinx build (`-E -b html -W`): passed.
- Direct behavioral reproductions of F1 and F2: confirmed.
- The final probes confirmed F3-F6 and F1 with advanced iteration history.
- All 713 existing tests passed across 27 modules, executed sequentially with
  each module in a separate Python process to limit accumulated JAX compilation
  memory. This includes all unit and integration tests in the repository.
- Warnings included a TensorFlow Probability deprecated-JAX-API warning and
  float64-to-float32 truncation warnings in 32-bit numerical test cases.
- Verification used Python 3.12 only; Python 3.13 was not tested locally.
- GitHub still reports the reviewed head, no merge conflict, review required,
  and no PR status checks. The existing CI workflow has no pull-request trigger;
  local validation supplies the execution evidence for this audit.

## Completion checklist

- [x] Read project context and review every changed production module.
- [x] Run all existing tests, lint, and the strict documentation build.
- [x] Reproduce and document the findings and required follow-up changes.
- [x] Recheck that the PR head remained unchanged during the audit.

No production code was changed.
Following the project's review workflow, remove this temporary audit document
once its findings have been addressed, before merging.

Assisted by Codex (OpenAI).
