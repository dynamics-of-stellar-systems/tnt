# Pull Request 16 Model-Search Audit

Status: temporary working document for reviewing and fixing pull request
[#16](https://github.com/dynamics-of-stellar-systems/tnt/pull/16). Update this
document as findings are resolved and delete it before the changes are merged
into `main`.

Audit date: 2026-08-05

Audited branch: `model-search-loop`

Pull request #16 establishes a useful model-search architecture, but the audit
recommends requesting changes before merging. Several execution-loop promises
are either incomplete or contradicted by the implementation.

## Main findings

### 1. Resolved: multi-round chi-squared stopping was unimplemented

Resolved on 2026-08-06. `ModelIterator._chi2_stopped_improving()` now
implements both configured modes. Improvement is the cumulative previous best
chi-squared value minus the cumulative new best value, since smaller values are
better. Absolute mode compares that difference directly with the threshold;
relative mode divides it by the previous best first. Equality continues the
search because stopping requires improvement to be strictly less than the
threshold.

`minimum_delta_chi2.enabled: false` explicitly disables
chi-squared-improvement stopping, allowing exploration to continue until the
model/iteration limits or parameter generator stop it. Mode and value remain
present, validated, and nonnegative while disabled; the generator's separate
`delta_chi2_threshold` is also nonnegative. Relative mode handles a previous
best of zero without division by zero.

Focused unit tests cover absolute, relative, equality-boundary, disabled, and
zero-denominator behavior.

### 2. Resolved: failure behavior now distinguishes a missing search base

Resolved on 2026-08-07. A fresh run records and then terminates after a first
iteration in which every model fails, because the parameter generator has no
successful model and therefore no valid configured chi-squared base from which
to continue. Resuming an `AllModels` table containing only failed models
terminates before requesting another proposal for the same reason. Both paths
now stop cleanly instead of leaking `AllModels.best()`'s `ValueError`.

Once at least one successful model exists, a later iteration containing only
failed models is not terminal. Those failures remain recorded, the previous
best model is retained, and the delta-chi-squared improvement check is skipped
for that iteration so the generator can try another proposal. The normal
generator, iteration-count, and model-count limits continue to apply.

Focused tests cover the initial all-failed iteration, an all-failed resumed
table, and a successful run that continues through a later failed-only
iteration to a subsequent successful model.

### 3. Resolved: model count is explicitly a soft target

Resolved on 2026-08-07. The former `n_max_mods` setting is now named
`target_model_count`, making its non-strict semantics explicit. TNT checks the
cumulative count before starting a new iteration, then evaluates every model
and potential rescaling in an iteration already underway. It does not triage
or defer part of the parameter generator's proposal merely to hit the target
exactly, so the final count may exceed the target. Other stopping conditions
may end the search below it.

The runtime variable, validation schema, packaged defaults, integration
configuration, tests, and documentation now use the new name. The integration
test deliberately retains the representative behavior in which a target of 3
produces 11 models by completing one potential-rescaling batch.

As a related stopping-contract update, the former cumulative `n_max_iter` is
now `n_new_iter`: the maximum number of additional iterations performed by the
current `ModelIterator.run()` call. A resumed run receives a fresh allowance,
while model and `IterationConfigLog` iteration numbers remain cumulative. The
runtime, validation schema, defaults, fixtures, tests, and documentation use
the new name; the legacy key is rejected rather than silently translated.

### 4. Resolved: unsupported execution settings are explicit

Resolved on 2026-08-06 for the requested scheduling scope:

- `model_processing_order: stage_by_stage` now raises a clear
  `NotImplementedError` during runtime construction and at `run()`, before
  model-search work begins; `model_by_model` remains supported.
- The remaining `orbit_workers` and `weight_workers` settings are explicitly
  documented as validated and retained but currently without execution effect
  because no scheduler uses them.
- The unused `external_chi2_workers` setting has been removed from
  configuration and test fixtures. TNT has no external chi-squared execution
  path; its internal chi-squared metrics remain unchanged. Configurations that
  supply the former key are rejected as unknown.
- The unused `orbit_family_integration_in_parallel` setting has subsequently
  been removed completely from configuration, runtime calls, and
  `Potential.generate_orbit_library()`'s signature. Configurations that still
  supply the former key are rejected as unknown.

Focused unit and integration tests cover the stage-by-stage failure, and
configuration tests cover rejection of the removed orbit-family and external
chi-squared worker keys plus the unsupported retry-enabled value.

`weight_solver_settings.reattempt_failures` remains available for the future,
but currently must be `false`; configuration rejects `true` with a clear error,
and every solve gets exactly one attempt. Implementing retry-enabled behavior
is intentionally left for a later change.

**Future discussion point:** Before allowing `reattempt_failures: true`, define
which failure types are retryable, whether only the weight solve or a larger
evaluation stage is repeated, how many attempts are allowed, what changes
between attempts, and how attempts and the final outcome are logged and
recorded. Decide whether a boolean remains sufficient once that policy is
specified.

### 5. Medium: schema changes during resume can discard parameter units

`AllModels._add_row()` creates every newly encountered column as an ordinary
unitless NumPy column. That is correct for new chi-squared metrics, but not for a
newly encountered potential parameter.

The audit tested appending a second model with a new parameter measured in
seconds: the stored column contained the numeric value but its unit was `None`.

Relevant location: `tnt/all_models.py`, around line 50.

Since the pull request specifically supports resuming under an edited
configuration, it should either:

- preserve units for new parameter columns; or
- reject parameter-schema changes when resuming.

**Discussion point:** Define the configuration-compatibility contract for
separate runs that append to the same `AllModels` set. Distinguish changes that
preserve a comparable model search—such as execution controls, stopping limits,
or parameter values and ranges—from changes to the scientific or table schema,
such as potential components and units, MGEs, observational data, spatial
binnings, or chi-squared definitions. Decide which changes are allowed, how
compatibility is checked against persisted run information, and whether an
incompatible change must start a new `AllModels` set or use an explicit
migration.

**Disposition:** Deferred from pull request #16. The configuration-compatibility
contract and its implementation will be addressed through issue #19 and a
separate pull request.

### 6. Low: the Galax dependency stubs require cross-platform review

The three new test modules claim the current environment cannot import the real
dependencies and consequently inject fake `galax` modules. The initial audit
recorded a successful Intel macOS import, but fresh isolated checks contradict
that result and show dependency failures on both the Intel-specific and modern
platform stacks. One relevant stub is in
`tests/unit_tests/test_model_iterator.py`, around line 9.

**Discussion point:** Do not treat this as a simple Intel macOS versus Apple
Silicon compatibility issue or remove the stubs before the dependency stack is
resolved. A fresh Intel macOS import fails in the Coordinax 0.20.0 and
Dataclassish 0.9.0 chain. A Linux x86_64 smoke test using the exact modern
versions selected by `uv.lock`—including Galax 0.0.2, Equinox 0.13.8,
Coordinax 0.23.3, and JAX 0.11.0—fails because Galax imports the removed
private Equinox symbol `_has_dataclass_init`. Apple Silicon resolves to this
same modern package set, so the same pure-Python import failure is expected
there as well. The interaction among Galax, Equinox, Coordinax, Dataclassish,
JAX, and their platform-specific version constraints must therefore be fixed
and tested across Intel macOS, Apple Silicon, and Linux before deciding that
the test stubs are obsolete.

**Disposition:** Deferred from pull request #16. Dependency-stack and
cross-platform compatibility will be addressed in a separate issue and pull
request.

## What the pull request implements

Conceptually, the new execution path is:

```text
resolved configuration
        |
        v
build MGEs, binnings, kinematics, populations
        |
        v
construct generator, orbit services, weight solver
        |
        v
generate potential parameters
        |
        v
build potential -> integrate orbit library -> solve weights
        |
        v
optionally solve mass-rescaled variants
        |
        v
record models and iteration/configuration provenance
        |
        v
apply stopping criteria
```

The main pieces are:

- `SinglePointParameterGenerator`: returns configured potential values.
- `ModelIterator.from_configuration()`: constructs the runtime object graph.
- `ModelIterator.run()`: owns iteration, evaluation, recording, resume
  counting, and stopping.
- `Model`: records potential, completion flags, weights, chi-squared values,
  and iteration.
- `AllModels`: flattens results into an Astropy `QTable` with ECSV persistence.
- `IterationConfigLog`: associates iterations with archived resolved
  configurations.
- Potential rescaling: reuses one orbit library across nearby overall mass
  scales.

This is not yet an end-to-end scientific execution implementation.
`GridSearchParameterGenerator`, potential construction, orbit integration
helpers, orbit rescaling, and the weight solver remain scaffolds. The
integration test uses real configuration, MGE, kinematics, and population
construction but replaces all numerical execution components with test
doubles.

## Suggested review order

Do not review the approximately 1,335 added lines sequentially. A more
productive sequence is:

1. Review the contracts in `Model`, `AllModels`, and `IterationConfigLog`.
2. Decide exact semantics for model limits, iterations, failures, and resume.
3. Review `ModelIterator.run()` against those decisions.
4. Review `_evaluate()` and mass-rescaling reuse.
5. Verify every accepted configuration option is either implemented or
   explicitly rejected.
6. Review the tests for missing negative cases: all models failing, resume
   after failures, strict model limits, real dependency imports, and
   persistence round-trips.
7. Review documentation only after runtime semantics are settled.

## Validation results

- `pytest -q`: 206 passed after the finding 1 through 4 changes
- `ruff check .`: passed
- Sphinx with warnings treated as errors: passed
- `git diff --check`: passed
- Real Galax import on Intel macOS: failed on a fresh recheck in the
  Coordinax/Dataclassish import chain; the earlier pass did not establish that
  the installed dependency stack was importable
- Real Galax import on Linux x86_64 with the modern `uv.lock` versions: failed
  because Galax 0.0.2 imports Equinox's removed private
  `_has_dataclass_init` symbol
- `ruff format --check .`: failed on six files; among pull-request-touched
  files, `tests/unit_tests/test_configuration.py` would be reformatted
- GitHub showed no automated status checks for the pull request at audit time

At audit time, the branch was clean, matched `origin/model-search-loop`, and
was one commit ahead of `main`. The initial audit itself made no code changes;
remediation commits are tracked by the resolved status under each finding.
