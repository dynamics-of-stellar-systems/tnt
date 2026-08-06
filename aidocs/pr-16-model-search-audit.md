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

### 1. High: any successful multi-round search eventually raises `NotImplementedError`

After the second successful round, `ModelIterator.run()` calls
`_chi2_stopped_improving()`, which is unimplemented. The tests avoid this by
monkeypatching the method.

Relevant locations:

- `tnt/model_iterator.py`, around line 223
- `tnt/model_iterator.py`, around line 243

This means the loop is currently usable only for a single round. Either
implement both configured threshold modes or explicitly reject multi-round
execution until they are implemented.

### 2. High: recorded evaluation failures still abort the search

`_evaluate()` catches integration and solver exceptions and creates failed
`Model` rows, as intended. Immediately afterward, however, `run()` calls
`models.best()`. If no model has a valid configured chi-squared value, that
raises `ValueError`.

This was reproduced locally with a deliberately failing weight solver:

```text
ValueError: AllModels.best: no model has a computed 'chi2'.
```

The same problem occurs when resuming an `AllModels` table containing only
failed models, at approximately line 200 of `tnt/model_iterator.py`.

This contradicts the documented promise that failures remain recorded and the
search continues.

### 3. High: `n_max_mods` is not actually a maximum

The limit is checked only before an entire proposed round. Every candidate and
every mass rescaling is then evaluated without considering the remaining
budget.

The integration test explicitly accepts `n_max_mods: 3` producing 11 models in
`tests/integration_tests/test_model_search.py`, around line 133.

Because real orbit models may be expensive, exceeding a configured maximum by
an arbitrary batch size is potentially significant. The project should decide
whether this is:

- a strict model limit, in which case candidate scheduling must respect the
  remaining budget; or
- a soft "stop after completing the current round" threshold, which should be
  renamed and documented accordingly.

### 4. Medium: validated runtime settings are silently ignored

The iterator stores `execution_settings`, but never reads it. Therefore
`model_processing_order: stage_by_stage` still executes model by model. Worker
counts and parallel-integration settings are also inactive.

Similarly, `weight_solver_settings.reattempt_failures: true` is not passed into
the iterator or honored; every solve gets exactly one attempt.

This is particularly visible because the integration configuration deliberately
selects `stage_by_stage` and enables retries, yet its test succeeds without
exercising either behavior. Relevant locations include:

- `tests/integration_tests/configuration.yaml`, around line 157
- `tnt/model_iterator.py`, around line 135

It is acceptable to defer these features, but unsupported choices should fail
explicitly instead of silently behaving differently.

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

### 6. Low: the Galax/Equinox incompatibility comments are stale

The three new test modules claim the current environment cannot import the real
dependencies and consequently inject fake `galax` modules. On the Intel Mac
used for this audit, the real import now succeeds with Equinox 0.11.10 and Galax
0.0.2. The 13 affected tests also pass after importing real
`galax.potential`.

The stubs should now be removed so tests can detect future dependency
regressions. One relevant location is
`tests/unit_tests/test_model_iterator.py`, around line 9.

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

- `pytest -q`: 179 passed
- `ruff check .`: passed
- Sphinx with warnings treated as errors: passed
- `git diff --check`: passed
- Real Galax import on Intel macOS: passed
- `ruff format --check .`: failed on six files; among pull-request-touched
  files, `tests/unit_tests/test_configuration.py` would be reformatted
- GitHub showed no automated status checks for the pull request at audit time

At audit time, the branch was clean, matched `origin/model-search-loop`, and
was one commit ahead of `main`. The audit itself made no code changes.
