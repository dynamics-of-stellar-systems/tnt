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

**Response -- fixed.** `PriorSampler._propose_free_parameters` now derives
its key as `jax.random.fold_in(jax.random.PRNGKey(self.seed),
all_models.n_iterations())`, exactly the suggested fix: distinct,
deterministic draws per cumulative iteration, with no conditioning on
fitted scores. `PriorSampler` still doesn't read chi2/scores -- a generator
that refines its proposals from fitted scores remains a distinct, future
generator -- but repeating a batch across iterations or resumes is no
longer a consequence of this one. New regression
`test_prior_sampler_draws_a_fresh_batch_each_iteration` in
`tests/unit_tests/test_parameter_generator.py`: advances a real `AllModels`
history (via lightweight `Model` fixtures) through one and then two
completed iterations, and checks the next batch differs from the previous
one, is reproducible when resumed at the same `n_iterations()`, and matches
an uninterrupted run that reaches the same point directly.

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

**Response -- fixed.** Added `tnt.priors._resolve_prior_distribution()`,
called once per site when `Prior` is constructed (not re-run on every model
trace): checks `distribution` names a real `numpyro.distributions.Distribution`
subclass (`TypeError` otherwise, with the config path), constructs it with
`validate_args=True` (confirmed this alone already catches `Normal(5, -1)`
and `LogUniform(-1, 9)`, since numpyro declares `Normal.scale`/`LogUniform.low`/
`high` as precise `arg_constraints`, just never checked because nothing
upstream passed `validate_args` at all), and separately requires `low <
high` whenever the constructed distribution exposes both (`Uniform`,
`LogUniform`) -- confirmed by reading `Uniform.__init__`'s actual source
that this ordering is never checked by numpyro itself, `validate_args` or
not, since `low`/`high` are declared `constraints.dependent` (numpyro's own
"the distribution must check this itself" marker). All four reproductions
now raise `ValueError`/`TypeError` at `Prior.__init__` instead of silently
sampling. 6 new tests in `tests/unit_tests/test_priors.py` (the 4
reproductions, a bogus `distribution` name, and a well-formed case).
Preparation (`tnt/configuration/validation.py`) is unchanged, still
structural-only, per the suggested fix.

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

**Investigation -- not fixed by F4, still open.** Fixed F4 first on the
hypothesis that the `NaN`-poisoned density it produces was the proximate
cause of this divergence rate. Re-ran this exact reproduction (same model,
same seed) after F4's fix: **identical 985/1000 divergences, bit-for-bit**.
NUTS evidently treats a `NaN` potential energy and a legitimate `-inf` one
the same way for its divergence check (both non-finite), so the chain's
actual trajectory decisions were unchanged either way. The real cause looks
more fundamental than F4: a hard `-inf` step-function cliff at the
valid/invalid geometry boundary is a known-bad shape for HMC/NUTS's
leapfrog integrator regardless of what produces it -- not a `NaN`-specific
bug. F4 was still a real, independently necessary fix (an undefined density
is not the same bug as a well-defined but pathologically-shaped one), just
not a fix for this finding. Still needed: diagnostics exposure + an
explicit enforcement policy in `Prior.sample()`, and likely reconsidering
the showcase test's prior shape -- a model with a ~98% divergence rate
isn't something any threshold should let through.

**Response -- fixed, but not with an automated threshold.** No fixed
divergence/ESS/split-R-hat cutoff can tell a genuinely broken chain apart
from a prior that's simply as wide as the actual science requires, so this
isn't solved by picking a number. Instead, `Prior` gained
`sample_with_diagnostics()` (returns the same samples as `sample()` plus
divergence/ESS/split-R-hat for every draw, MCMC or `Predictive` alike) and
`tnt.priors_diagnostic_plot.plot_prior_diagnostics`, which draws once -- at
exactly the `num_samples`/`num_warmup` the real run will use -- and saves a
diagnostic figure (pairwise scatter colored by divergence, each
parameter's marginal histogram, an ESS/split-R-hat table, and
per-parameter trace plots) to
`io_settings.output_directory/prior_diagnostics.png`.
`ModelIterator.from_configuration` generates this once whenever
`generator_type` is `PriorSampler`. It is purely advisory: nothing raises,
nothing blocks, the run proceeds regardless -- the user looks at the
figure themselves and decides whether to let `PriorSampler` start
proposing. `matplotlib` is now a regular (not optional) dependency, since
TNT needs it for visualization regardless of this feature.

**Additional response -- the showcased reproduction's own configuration is
no longer expressible.** Separately from the diagnostic plot, `prior-concept`
also added *default priors* for the MGE shape parameters that produced this
finding's reproduction: `pqu`'s `p`/`q`/`u` (`TriaxialLight/MassMGEPotential`)
and `q_min` (`OblateLight/MassMGEPotential`) are no longer sampled from a
user-declared, independent `prior` at all -- they're always sampled from a
built-in generative prior instead (`tnt.potential.registry.DefaultPriorSpec`,
`tnt.priors._build_model`), which by construction keeps the vast majority of
draws inside the valid geometry (`p ~ Uniform(0, 1)`; `q ~ Uniform(0, min(p,
q'))`, `q'` the MGE's own anchor axis ratio -- bounding `q` by `q'` as well
as by `p` is what rules out the specific empty-domain case an unconstrained
`q <= p` alone does not; `u` sampled within the MGE-dependent domain
`potential.md` already describes). Measured against the real MGE fixture:
~95-99.95% of draws are valid by construction (float32 vs. float64), versus
the small fraction this finding's own reproduction got from three
independent `Uniform(0, 1)` priors on `p`, `q`, `u`.

A declared `prior` on `p`/`q`/`u` is still accepted, but it no longer drives
the sample the way it did in this finding's reproduction -- it's layered on
as an *additional* factor scoring the value the default prior already
sampled, not a competing sample site. So the exact configuration shape this
finding reproduced (three independent `Uniform(0, 1)` sample sites directly
producing the proposed `(p, q, u)`, hence the hard `-inf` cliff NUTS had to
navigate almost everywhere) can no longer be constructed for these
parameterizations at all, regardless of what any plugin or declared prior
does on top.

This does not replace the diagnostic-plot response above -- it's still the
general, advisory answer for any factor-bearing model, `pqu`/`q_min` or
otherwise, and a user can still build some other pathologically-shaped
prior+plugin combination this finding's general concern would still apply
to. It specifically neutralizes the one configuration this finding's own
reproduction was built on. Renamed/updated regression:
`test_default_prior_keeps_pqu_draws_valid_without_a_declared_prior` (was
`test_plugin_factor_on_enclosed_mass_rejects_invalid_pqu_shape_draws`) in
`tests/unit_tests/test_priors.py`, plus
`test_default_prior_keeps_qmin_draws_valid_without_a_declared_prior` for the
oblate counterpart.

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

**Response -- fixed.** `PriorContext` gained `build_potential_with_validity()`
(the `(Potential, valid)` pair, built and cached together -- `build_potential()`
now just returns its first element, unchanged for every existing caller) and
`factor_if_valid(name, compute)`: evaluates `compute` only when `valid` is
true, via `jax.lax.cond` -- which evaluates and differentiates only the
taken branch, unlike `jnp.where`, which would still evaluate and
differentiate both and leak the `NaN` back into the gradient even though
its *selected* forward value looks fine. The worked `m_within_10kpc`
example (`docs/source/priors.md`, `tests/unit_tests/test_priors.py`,
`aidocs/prior_concept_illustration.py`) now registers its factor through
`factor_if_valid` instead of calling `numpyro.factor` directly. New
regression `test_factor_if_valid_keeps_an_invalid_draws_density_exactly_minus_inf`
reproduces the audit's own method exactly (`numpyro.infer.util.log_density`
at `p=0.1, q=0.9, u=0.2, m=1e12`): now `-inf` with a finite gradient, not
`NaN`. This did not, however, turn out to fix F3 -- see that finding's own
investigation note above.

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

**Response -- fixed.** `_NON_SCHEMA_PARAMETER_KEYS` in
`tnt/configuration/compatibility.py` now excludes both `generator_settings`
and `prior`, exactly the suggested fix -- confirmed by reverting just this
one-line change and re-running the new regression, which fails with
precisely the reproduction this finding describes (`changed fields:
critical_configuration.potential.stars.parameters.q.generator_settings`),
then passes once the key is restored. New regression
`test_resume_allows_a_legacy_generator_settings_archive_against_current_prior`
in `tests/unit_tests/test_configuration_compatibility.py`: an archived
baseline run whose `q` parameter still declares the pre-`prior:`
`generator_settings` sub-dict (`lower_bound`/`upper_bound`/`step`/
`minimum_step`), resumed against a current config declaring `prior` for the
same parameter with identical `value`/`unit`/`fixed` -- `ensure_resume_compatible`
no longer raises. New user-profile preparation (`tnt/configuration/validation.py`)
is unchanged -- it already rejects `generator_settings` as an unknown
per-parameter key for a *new* config, per the suggested fix; this response
only touches historical-archive comparison.

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

**Response -- fixed.** `configuration.md`'s registries example (`ml`,
`theta`, `phi`, `psi`) and all five `potential.md` MGE examples now declare
`fixed: true` explicitly, instead of silently inheriting the packaged
`dynamic_object_defaults.parameter.fixed: false` and failing preparation.
The two `potential.md` examples covered by this PR's own later default-prior
work (`stars_pqu`'s `p`/`q`/`u`, `bulge_q_min`'s `q_min` -- see the F3
response above) are left without `fixed`/`prior` on purpose: they no longer
need either, and a note right after that example explains why only `ml`
needed `fixed: true` there. `configuration.md`'s stale "generic
search-parameter bounds" sentence (the removed bound validator this finding
also flagged) is reworded to describe the current `fixed`/`prior`/
default-prior contract. `priors.md` gained a new "Enabling `PriorSampler`"
section with a complete, verified `parameter_space_settings` block
(`generator_type`, `generator_settings.seed`/`num_warmup`,
`stopping_criteria`, `potential_rescalings`, `priors`) -- while building it,
also caught and fixed a related inaccuracy one section down: the diagnostic
plot's `num_samples` comes from `stopping_criteria.max_new_mods_per_iter`,
not a nonexistent `generator_settings.num_samples` the text previously named.
All three edited examples (the registries example, the five-component
example, and the new `PriorSampler` block) were each verified end to end
through the real `Configuration().read()` path, not just a Sphinx build
(which the finding correctly noted only checks rendering). Strict Sphinx
build (`-E -b html -W`) still passes.

## Follow-up work

The default-prior mechanism added in response to F3 (see above) deliberately
covers only `pqu` and `q_min` -- two further cases were scoped out during
that work and tracked as
[issue #82](https://github.com/dynamics-of-stellar-systems/tnt/issues/82):

- Extending default priors to `pqu`'s sibling parameterizations
  (`T_maj_min`, and the native `theta`/`phi`/`psi` case with no declared
  `parameterization`), blocked on making the existing parameterization
  `invert` conversions JAX-traceable -- PRs #77/#78 traced the forward
  direction only; `triaxial_intrinsic_shape`,
  `T_Tmaj_Tmin_from_viewing_angles`, and `q_min_from_inclination` are
  still eager.
- Extending default priors to curated `galax` potential types with their
  own joint, but data-independent, ordering constraints among native
  parameters (`LeeSutoTriaxialNFWPotential`'s `a1 >= a2 >= a3`,
  `StoneOstriker15Potential`'s `r_h > r_c`) -- already enforced via the
  existing `ParameterConstraint(other_parameter=..., relation=...)`
  mechanism, but not yet sampled efficiently the way `pqu`/`q_min` are.

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

---

## Re-review — 2026-10-14

Reviewed PR head: `5ec8b09d6db3e8991c7549ff1b97ac4267253534`.
Current main: `4bb8711150a3a6511b8edc1d1b0d9c6fccf226f0`.
The material above, including the author's responses, is preserved unchanged.

### Judgment

**Not ready to merge yet.** Most original fixes are implemented correctly.
The remaining concerns are the default prior's handling of fixed shape
coordinates, reporting diagnostics for actual search batches, and edge cases
and import side effects in the new plotting module. These arise in the new
default-prior/diagnostic work; they are not requests to implement the separate
features tracked in issue #82.

Candidate sampling remains the agreed scope. No fit-score or posterior
integration change is required. An advisory diagnostic policy is reasonable
for this exploratory generator; an arbitrary universal rejection threshold
is not required by this re-review. The advisory information must nevertheless
describe the samples actually used, and plotting must represent them correctly.

### Original findings and individual commits

| Finding/change | Re-review outcome |
| --- | --- |
| F1: repeated batches; `153d269`, then `01f0f7a` | Fixed. The final code folds in cumulative iteration history. The initial deferral was superseded by the implementation and regression. |
| F2: distribution validation; `1b63cda` | Fixed for the reproduced malformed distributions. Construction validates distribution classes/arguments and explicitly checks bound ordering. |
| F4: invalid derived factors; `34299b8` | Fixed through the supported `factor_if_valid` interface. The invalid-density regression passes. |
| F3: diagnostics; `2d8b209` | Partially addressed. Diagnostics are exposed, but the generator still discards them and the automatic figure describes a separate pilot draw. See R2. |
| F3: generative defaults; `b16238a` | A substantial improvement for fully free shapes. Partial fixing leaves large structurally invalid regions. See R1. |
| F5: historical resume; `dea5de3` | Fixed. Both historical and current search declarations are excluded; the new archive-compatibility regression passes. |
| F6: examples; `bb7c95c`, `2403e1c` | The original fixed/prior omissions were corrected; the preparation changes and strict documentation build pass. Additional wording issues are noted below. |
| Sampling batch size; `6190375` | The separation from the model-count cap is sound. Configuration validates the new settings and small output caps work in the generator tests. |
| Follow-up scope; `5ec8b09` | Issue #82 explicitly tracks other coordinate systems/native joint constraints. Those extensions can remain separate. |

### R1 — P1: default PQU sampling does not respect fixed downstream coordinates

Location: `tnt/potential/triaxial_mge.py:376-387`.

The sampler draws `p` before considering a fixed `q` or `u`. With `q=0.6`
fixed, it still draws `p ~ Uniform(0, 1)`, so about 60% of proposals violate
`q < p` before deprojection. A fixed `u` likewise does not restrict `p` or
the sampled `q`. Passing a fixed value through unchanged is insufficient to
sample the remaining coordinates within their joint domain.

Using the PR's `_POTENTIAL_PQU` and `_TRIAXIAL_MGE` fixtures, seed 1, 1,000
draws, no plugins, and the actual `Potential.build_with_validity` path:

| Fixed shape fields | Fraction of valid proposals |
| --- | --- |
| None | 98.4% |
| `q=0.6` | 39.0% |
| `u=0.93` | 19.2% |
| `q=0.6, u=0.93` | 20.9% |
| `p=0.85` | 99.9% |

These are supported fixed-field declarations, with valid nominal coordinates.
The missing domain restrictions are structural, not the documented residual
floating-point margin case. With a potential-building plugin, the hard
rejection boundaries that the new prior was intended to avoid return; without
one, most such draws become invalid models. The iterator stops a fresh search
after an all-failed first round, so wasting most of a small proposal batch can
also terminate useful exploration.

**Required change:** condition the remaining coordinates' sampling bounds on
all fixed members of the group, or explicitly reject unsupported fixed/free
combinations with a named setup error. Add coverage for the supported subsets,
especially fixed `q`, fixed `u`, and fixed `q/u` together. Preserve fixed
values; do not silently substitute a different shape.

### R2 — P2: actual search batches still lose their diagnostics

Locations: `tnt/model_iterator.py:187-197`,
`tnt/parameter_generator.py:293-302`, and `tnt/priors.py:564-571`.

The constructor generates `prior_diagnostics.png` with `PRNGKey(seed)`, while
the generator uses `fold_in(PRNGKey(seed), all_models.n_iterations())` and
calls `Prior.sample`, which discards the returned diagnostics. Even the first
batch therefore differs from the plotted draw; later iterations/resumes have
no retained or reported batch diagnostics. Matching warmup/sample counts and
target acceptance settings does not make these the same sampling run.

**Reproduction:** for `Uniform(1, 9)` and seed 0, the pilot's first value is
`6.36954389455336`, while the first search batch starts with
`3.813330758938811`. For the updated PQU/enclosed-mass fixture with 1,000
warmup steps and 1,000 samples, the pilot reported **51 divergences**, while
the actual first-batch key reported **24**. The first batch's diagnostics are
then thrown away by the normal generator path. These counts are diagnostic
observations, not counts of invalid final models or proof of a specific bias.

The new default substantially improves the old 985/1,000 benchmark, but a
separate pilot cannot establish the quality of every later chain. The plot
also skips its entire ESS/split-R-hat/divergence-summary table when there is
only one free parameter (`table_rows = n // 2`), a common supported case.

**Required change:** retain and report diagnostics from each actual sampling
batch, associated with its iteration/run. Rendering can consume those samples
and diagnostics rather than draw another batch. A pilot may remain an
additional, clearly labelled check. Include numerical summaries for
one-parameter models. This can stay advisory; no automatic cutoff or user
approval gate is required.

### R3 — P2: plotting fails or misrepresents supported parameter configurations

Locations: `tnt/priors_diagnostic_plot.py:36-41, 107-122, 166-174`.

Two directly reproduced cases:

- **All parameters fixed:** the realistic integration configuration passes
  `Configuration.read` with `generator_type: PriorSampler` and every
  parameter fixed, then `ModelIterator.from_configuration` raises
  `ValueError: Number of rows must be a positive integer, not 0` in the
  automatic plot. `Prior` itself correctly returns no stochastic sites;
  plotting assumes at least one.
- **Signed values:** a valid bar-orientation prior,
  `LongMuraliBarPotential.alpha ~ Uniform(-1, 1)` radians, is selected for
  logarithmic plotting because `_use_log_scale` checks only its positive
  subset. The histogram then computes `log10` of its negative minimum,
  emits a runtime warning, and uses NaN bin edges. Inspection of the generated
  figure showed **zero histogram counts for 1,000 draws**, and its
  logarithmic trace axis excludes the **513 negative draws**. The function
  returns successfully, so the advisory picture silently loses valid data.

**Required change:** handle zero stochastic parameters explicitly, or reject
that unsupported sampler setup during validation with a clear message. Only
select a logarithmic axis when the complete data to be displayed are strictly
positive and finite; render signed ranges without losing values. Add tests
for empty, one-parameter, and signed-value plots that check the represented
data, not only that a PNG exists.

### R4 — P2: importing ModelIterator changes the application's plotting backend

Location: `tnt/priors_diagnostic_plot.py:13-17`, imported unconditionally by
`tnt.model_iterator`.

`matplotlib.use("Agg")` runs at module import, even when the caller does not
select `PriorSampler` or request a diagnostic figure. In an isolated
reproduction, selecting the SVG backend and creating a figure, then importing
`tnt.model_iterator`, changed the global backend from **SVG to Agg**. The
existing figure remained present in this Matplotlib version; this audit does
not claim it was deleted. The application nevertheless loses its chosen
backend for subsequent plotting, including interactive workflows.

**Required change:** use a figure-local canvas for file rendering, or leave
backend selection to the application/standalone entry point. Importing TNT's
model iterator should not change unrelated plotting configuration. Add a
regression with a preselected backend.

### Documentation and integration notes

- Keep the advisory policy, but correct the claim that a wide scientific prior
  makes divergences/low effective sample size innocuous. These can indicate
  sampling trouble even when the model/prior is scientifically appropriate;
  they do not, by themselves, identify whether the cause is code or geometry.
- `KNOWLEDGE.md` still says every non-fixed parameter must declare a prior,
  without the new default-prior exception. The plugin section also says there
  are no built-in priors, although the default priors are now built in;
  distinguish default priors from user-authored factor plugins.
- Refresh the PR description: it still discusses the old NaN behavior,
  independent PQU sampling, and default priors as entirely deferred.
- The PR is behind current main only by the orbit-start-space merge (#80).
  GitHub reports no merge conflict. Merging main cannot fix R1-R4; the combined
  state should be checked when the code fixes are ready.
- Per the existing workflow, remove the temporary audit document after the
  review findings are resolved, before merging.

### Re-review validation

- Rebuilt the frozen Linux `x86_64` development image, including Matplotlib.
  Python 3.12.13, JAX 0.10.2, NumPyro 0.21.0, Matplotlib 3.11.2.
- **231 tests passed** across 10 focused modules: priors, parameter generator,
  configuration, configuration compatibility, diagnostic plotting,
  model-search integration, model iterator, shared shape conversions,
  parameterized MGE construction, and packaged defaults. Modules ran in
  separate processes, sequentially, within the documented Colima memory limit.
- Ruff and the strict Sphinx build passed; whitespace check passed.
- The additional probes above reproduced R1-R4. The actual first-batch probe
  explicitly confirmed JAX x64 was enabled.
- The full repository suite and Python 3.13 were not rerun. The commit message
  reporting one pre-existing failure does not identify it; its origin was not
  established by this focused run. None of the selected modules failed.
- PR head was unchanged on recheck. GitHub reports the old requested-changes
  review and no PR status checks; no merge conflict is reported.

Assisted by Codex (OpenAI).
