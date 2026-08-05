# Model search

Once a configuration is resolved (see [Configuration preparation](configuration.md)),
TNT's execution phase searches parameter space for the potential whose
orbit-based dynamics best match the observed kinematics. Four types make up
that search, each scoped to one job:

- `AbstractParameterGenerator` proposes candidate points in parameter space.
- `ModelIterator` evaluates each candidate: it builds a potential, integrates
  its orbits, and solves for orbit weights and chi2 against the data.
- `Model` records the outcome of evaluating one candidate.
- `AllModels` accumulates every `Model` evaluated so far into one table.

This part of TNT is under active development. Some of what's described below
is signature-only scaffolding rather than a working implementation -- each
module's own docstrings say exactly what's implemented and what still raises
`NotImplementedError`.

## The loop

`ModelIterator.run()` repeats a generate -> evaluate -> record -> stop cycle:
ask the parameter generator for the next round of candidates, evaluate each
one into a `Model`, append every result to the running `AllModels`, and check
whether a stopping criterion has been reached. It stops when the generator
has nothing left to propose, or once `parameter_space_settings.stopping_criteria`
is satisfied -- a maximum number of rounds or models evaluated, or the best
chi2 no longer improving between rounds.

Because `run()` accepts a previously written `AllModels` to resume from, that
budget is tracked cumulatively: resuming continues counting rounds and models
from where the earlier run left off, rather than granting a fresh allowance.

`run()` also accepts and returns an `IterationConfigLog`: one row per round,
recording which `resolved_config_path` was in effect for it. A search can be
paused and resumed under an edited configuration, so `AllModels` and
`Model.iteration` alone can't show which config file produced a given round
-- `IterationConfigLog` is the record of that, meant to be written alongside
`AllModels` into a run's config archive.

## ParameterGenerator

A `ParameterSet` is one proposed point in parameter space: a mapping from
potential-component name to a mapping of parameter name to value (e.g.
`{"bh": {"m": 5.0, "a": 0.001}, "stars": {"ml": 5.2, ...}}`). Every
`AbstractParameterGenerator` implements one method,
`generate_parameters(all_models)`, returning the next round of `ParameterSet`s
to evaluate given every model evaluated so far. Which implementation runs is
chosen by `parameter_space_settings.generator_type`:

- `GridSearchParameterGenerator` ("GridSearch") proposes parameters on a grid,
  per each parameter's `generator_settings` (`lower_bound`, `upper_bound`,
  `step`, `minimum_step`) in the configuration's `potential` section.
- `SinglePointParameterGenerator` ("SinglePoint") always proposes the same
  single point, taken directly from each parameter's configured `value`. It
  ignores `all_models` entirely, so it's meant for evaluating one nominal
  potential rather than searching -- pair it with
  `stopping_criteria.n_max_iter: 1` to stop after that one round.

## Model

A `Model` is one evaluated point: the `Potential` that was proposed, plus
whatever its evaluation produced. `potential` is always set -- it's known
before evaluation even starts -- but the rest reflects what actually
happened, since evaluating a point can fail at more than one stage:

- `orblib_done` is `False` if integrating the potential's orbit library
  itself failed.
- `weights_done` is `False` if orbit integration succeeded but solving for
  orbit weights against the kinematic data failed.
- `weights`/`chi2` are only set once `weights_done` is `True`; otherwise
  they're `None`.
- `iteration` is the 0-based search round (`ModelIterator.run()` call) that
  produced this model.

A single evaluated `ParameterSet` can produce more than one `Model`: if
`parameter_space_settings.potential_rescalings.enabled`, the same orbit
library is cheaply reused at several nearby mass scales (via `Potential.rescale`)
without re-integrating, each producing its own `Model`.

## AllModels

`AllModels` is the growing table of every `Model` evaluated so far, backed by
an astropy `QTable` that round-trips through `.ecsv` (preserving each
column's unit). Its columns are:

- one column per potential-component parameter, with unit where applicable
  (e.g. `bh.m`, `stars.ml`) -- always present, since a proposed point's
  parameters are known before evaluation;
- `orblib_done`/`weights_done`, mirroring `Model`'s own flags;
- one column per chi2 metric (e.g. `chi2`, `kinchi2`), once at least one
  appended model has `weights_done`.

Because some models fail, chi2 columns can contain `nan` for rows that never
got that far. `AllModels.best(which_chi2)` returns the row with the lowest
value in that chi2 column, ignoring `nan`s, and raises a clear error if no
model has a computed value for it yet. `len(all_models)` and
`all_models.n_iterations()` (models evaluated / search rounds completed) are
what let `ModelIterator.run()` resume its stopping-criteria counts across a
previously written `AllModels`.

## ModelIterator

`ModelIterator` owns the search loop described above and the services it
needs to run: the fixed potential-component structure and MGEs
(`potential_settings`/`mges`), the kinematic and population data being fit
against, the weight solver, the parameter generator, and the orbit-library
and stopping-criteria settings.

Its `_evaluate()` step, for one proposed `ParameterSet`, builds a `Potential`,
integrates its `OrbitLibrary`, and solves orbit weights against the
kinematic data (plus any `potential_rescalings` variants, reusing that same
orbit library). A failed orbit integration or weight solve doesn't raise --
it produces a `Model` with the corresponding flag `False` instead, so the
search can continue and the failure stays visible in `AllModels`.

One pattern recurs through `ModelIterator`'s implementation, worth knowing
before reading the code: the method doing the actual work stays ignorant of
context it doesn't need, and its caller attaches that context afterward.
Solving orbit weights doesn't need to know which `Potential` (or which
rescaling) they came from; assembling the resulting `Model` is what attaches
it. Evaluating a `ParameterSet` doesn't need to know which search round it's
in; `run()` is what stamps `Model.iteration` on once results come back.
