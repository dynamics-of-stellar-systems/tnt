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
is satisfied -- the maximum number of rounds, the soft model-count target, or
the best chi2 no longer improving between successful rounds. A fresh run also
stops after recording its first round if that round produces no successful
model: without any computed value for `which_chi2`, the parameter generator
has no valid base from which to continue.

The chi2-improvement criterion compares the cumulative best value before and
after a round. Because smaller chi2 values are better, improvement is
`previous_best_chi2 - best_chi2`. In `absolute` mode that difference is
compared directly with `minimum_delta_chi2.value`; in `relative` mode it is
divided by `previous_best_chi2` first. The search stops when the improvement
is strictly less than the configured value, so an improvement exactly equal
to the threshold continues the search.

Setting `minimum_delta_chi2.enabled: false` disables chi2-improvement stopping.
This allows the generator to keep exploring, including proposing and recording
models whose chi2 is worse than the best already found, until `n_new_iter`,
`target_model_count`, or the generator itself stops the search. `mode` and
`value` remain present and validated while the criterion is disabled;
threshold values must be nonnegative. This does not alter the generator's
separate `delta_chi2_threshold`, which is also nonnegative.

`target_model_count` is deliberately a soft target rather than a strict
maximum. TNT starts a new iteration only while the cumulative model count is
below it. Once an iteration begins, every proposed model, including potential
rescalings, is evaluated. The final model count may therefore exceed the
target; TNT does not triage or defer part of a proposed iteration merely to
match it exactly. Other stopping conditions may also end the search below the
target.

`n_new_iter` limits only the number of additional iterations performed by the
current TNT run. If a persisted search already contains five
iterations and `n_new_iter` is 3, a resumed call may perform iterations 5, 6,
and 7. The model and configuration-log iteration labels therefore remain
cumulative even though the allowance is renewed for each resumed call.

Once at least one successful model exists, a later round containing only
failed models does not trigger the delta-chi2 check. TNT retains the previous
best model and lets the parameter generator propose another round, subject to
the ordinary generator, iteration-count, and model-count limits.

Because `run()` accepts a previously written `AllModels` to resume from, the
model-count target is tracked cumulatively, while `n_new_iter` grants the
current run its configured number of additional iterations. If the resumed
table contains models but none completed successfully, TNT terminates without
asking the generator for another round because no valid chi2 base was
established by the earlier run.

`run()` also accepts and returns a `RunConfigLog`: one row per round, mapping
the cumulative iteration number to the ID of the TNT run that produced it.
Multiple rounds may reference the same run. The run's immutable manifest is
the authoritative link to its archived resolved configuration and execution
provenance. `ModelIterator.run()` allocates the run ID after runtime-object
construction and preflight validation succeed; see {ref}`run-identity`. A
search can be paused and resumed under an edited configuration, so `AllModels`
and `Model.iteration` alone cannot show which run produced a given round.

The log has a fixed persisted location at
`config_repository/run_config_log.ecsv`. Its `read()` and `write()` methods
validate contiguous iteration IDs, nonnegative run IDs, and every referenced
run manifest; `write()` uses atomic replacement so an interrupted write cannot
destroy the previous log. Model search deliberately returns both state objects
instead of writing either one internally. The calling execution layer must
load and save `AllModels` and `RunConfigLog` together; `run()` rejects a pair
that describes different numbers of iterations:

```python
from pathlib import Path

from tnt.model_iterator import ModelIterator
from tnt.model_search_state import ModelSearchState
from tnt.run_config_log import RunConfigLog

output = Path(config.data["io_settings"]["output_directory"])
models_path = output / config.data["io_settings"]["all_models_file"]
iterator = ModelIterator.from_configuration(config)
log_path = RunConfigLog.path_for_repository(iterator.configuration_repository)

state = ModelSearchState.read(
    models_path,
    log_path,
    repair_log_ahead=True,
)
models, run_config_log = iterator.run(state.all_models, state.run_config_log)
ModelSearchState(models, run_config_log).write(models_path, log_path)
```

The ECSV metadata contains `total_runs` and
`run_ids_without_iterations`. Both are derived from the immutable manifests
and the iteration rows whenever the log is read or written. The execution
layer must therefore write the returned log even when the current run produces
no iteration; its run ID will then appear in `run_ids_without_iterations`
without adding a nullable iteration row.

This applies to both initial and resumed runs. A resumed run may add no
iteration because its stopping criteria are already satisfied, its parameter
generator proposes nothing, or it cannot continue from the existing models.
The `run()` invocation creates that run's manifest and run ID, and
persisting the unchanged search state records the ID in
`run_ids_without_iterations`. The initial zero-model case is special only
because no `AllModels` table schema exists yet.

`ModelSearchState.write()` validates both temporary ECSV files before
publishing them. It replaces the run log first and `AllModels` second. A crash
between those replacements can therefore leave only safely discardable,
trailing run-log rows; the explicit `repair_log_ahead=True` option truncates
them when resuming. TNT rejects the opposite mismatch because models whose run
provenance is absent cannot be reconstructed safely. An initial zero-model
checkpoint writes only the run log because `AllModels` has no table schema yet.

The example checkpoints once after `ModelIterator.run()` returns. The iterator
writes only its immutable run bundle; it does not write model-search
checkpoints. An execution layer that checkpoints after every completed
iteration must call the same paired writer at each checkpoint.

This log records provenance and verifies that referenced per-run manifests and
resolved configurations exist and remain readable. It does not decide whether
different configurations are compatible; that compatibility contract is a
separate model-resume policy.

## Configuration compatibility on resume

TNT performs the compatibility check once at the beginning of each `run()`
invocation, before allocating that call's new run identity or entering the
internal iteration loop. The check uses `RunConfigLog` to identify the
earliest run that contributed an iteration, then loads that run's archived
`resolved_config.yaml` and compares its compatibility-critical fields
directly with the current resolved configuration. An incompatible change raises
`ConfigurationCompatibilityError` before models or the log are modified. The
error lists the differing field paths. Runs that produced no iteration do not
define the model set's compatibility baseline.

The one-call-one-run contract means repeated calls on one iterator represent
distinct runs and intentionally repeat the check. Moving it into
`Configuration.read()` would be too early because the selected chi-square and
model-table schema checks require the previous search state, and runtime-object
construction must succeed before TNT can archive the configuration as a run.

The current compatibility contract allows changes that control future search,
execution, presentation, or post-processing without changing existing model
meaning:

- worker and processing-order settings;
- stopping criteria, generator settings, potential-rescaling ranges, and
  `parameter_space_settings.priors` (prior plugins -- these govern how a run
  searches, not what the model is);
- values, declared units, declared `prior`s, fixed flags, and labels of
  existing potential parameters;
- display units, logging, and analysis settings; and
- input/output directory paths, provided the configured scientific file
  references themselves do not change.

It rejects changes to:

- internal units, `cosmological_parameters`, and physical system attributes
  such as distance (`system_attributes.name` remains metadata);
- declared potential components, types, MGE references, parameter names, or
  remaining parameter-schema fields;
- MGE settings, spatial binnings, kinematics, population data, or their
  configured scientific file references;
- all `numerics_settings` in the current contract;
- orbit-library settings; and
- weight-solver settings.

This comparison distinguishes declaration identity from physical identity.
Per-run resolved configurations preserve declared units, while compatibility
comparison recognizes atomic `{value, unit}` mappings and converts only the
two values being compared. It therefore treats declarations such as `1 kpc`
and `1000 pc` as physically equivalent without normalizing the complete
configuration. Incompatible dimensions compare unequal, while malformed
quantity declarations raise `ConfigurationCompatibilityError`. Numerical
values must match exactly after unit conversion; compatibility does not add a
tolerance that could hide a small scientific change. Changing the internal
unit system itself remains incompatible under the current contract. This
comparison is intentionally host-side and independent of
`numerics_settings.jax_enable_x64`; choosing 32-bit runtime calculations must
not reduce the precision used to decide whether a declaration changed.

Changing `parameter_space_settings.which_chi2` is allowed only when the chosen
metric exists and is finite for every successful historical model. A nonempty
`AllModels` table must also contain every parameter column required by the
declared potential components.

TNT does not hash scientific input files. Users must not modify an MGE,
spatial-binning, kinematics, or population file in place while an existing
model set may still be resumed. Changing a configured scientific filename is
rejected, but changing bytes at the same path cannot be detected. This is an
explicit user responsibility in exchange for a lean configuration repository.

The same output directory must have only one coordinating TNT writer. Parallel
workers may calculate different models within an iteration, but they must
return results to the coordinator; only that process updates `AllModels`,
`run_config_log.ecsv`, or other shared bookkeeping files.

A negative orbit-library random seed is valid for both fresh and continued
runs. Like the other orbit-library settings, changing its configured value
between runs remains incompatible under the current contract.

## ParameterGenerator

A `ParameterSet` is one proposed point in parameter space: a mapping from
potential-component name to a mapping of parameter name to value (e.g.
`{"bh": {"m": 5.0, "a": 0.001}, "stars": {"ml": 5.2, ...}}`). Every
`AbstractParameterGenerator` implements `_propose_free_parameters(all_models)`;
the base class's concrete `generate_parameters(all_models)` calls it and caps
the result at `stopping_criteria.max_new_mods_per_iter`, uniformly across every
generator regardless of how many candidates its own proposal logic happens to
produce. Which implementation runs is chosen by
`parameter_space_settings.generator_type`. Concrete generators explicitly
register their `_type`; an undecorated subclass is not a
configuration-selectable generator:

- `GridSearchParameterGenerator` ("GridSearch") is the registered scaffold for
  proposing parameters on a grid from each parameter's declared `prior`; its
  proposal algorithm is not implemented yet.
- `SinglePointParameterGenerator` ("SinglePoint") always proposes the same
  single point, taken directly from each parameter's configured `value`. It
  ignores `all_models` entirely and never consults `parameter_space_settings.priors`,
  so it's meant for evaluating one nominal potential rather than searching --
  pair it with `stopping_criteria.n_new_iter: 1` to stop after that one round.
- `PriorSampler` proposes parameters by sampling from a `tnt.priors.Prior` --
  see [Priors](#priors) below. Also ignores `all_models`.

## Priors

A `prior` on a potential parameter (`{distribution: "<numpyro.distributions
class>", args: [...]}`, sibling to `value`/`unit`/`fixed`) declares its
search-space distribution and compiles to one `numpyro.sample` site, keyed by
`"<component>.<parameter>"`. Every non-fixed parameter must declare one;
configuration preparation rejects a `fixed: false` parameter with no `prior`.
`PriorSampler` composes every non-fixed
parameter across the whole `potential` section, plus every
configured `parameter_space_settings.priors` plugin, into one numpyro model
per run (`tnt.priors.Prior`) and draws candidates from it.

A prior **plugin** is a user's own Python function -- not a class, not
shipped by TNT itself (there are no built-in priors; even the canonical
mass-fraction example below is documentation, not a package feature) --
referenced from config as `plugin: "<path>:<function_name>"`, with `<path>`
resolved relative to `io_settings.input_directory`, the same convention MGE,
kinematics, and population files already use:

```yaml
parameter_space_settings:
  priors:
    dh_mass_fraction:
      plugin: "priors/mass_fraction.py:mass_fraction"
```

A plugin function may only call `numpyro.factor` -- never `numpyro.sample` or
`numpyro.deterministic`. This is what lets ordinary per-parameter priors and
plugins compose safely with no coordination between them: a plugin can add a
soft log-density preference over values already established elsewhere, but
it can never independently assign or overwrite a parameter's value, so it can
never collide with that parameter's own declared `prior`.

**The signature is fixed:** a plugin is callable with one positional argument,
a `tnt.priors.PriorContext`, and returns nothing. (A trailing parameter with a
default, or `*args`, is tolerated but never populated; any other arity is
rejected when the plugin is loaded.) `PriorContext` carries the run's
state a plugin is allowed to read -- currently `context.candidate` (the
parameter values assembled so far this draw, `{component: {parameter: value}}`,
bare unit-stripped numbers) and `context.mges` (the run's named MGEs). New
fields are added there over time; because a plugin only ever names this one
argument, that stays backward-compatible.

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from tnt.priors import PriorContext


def mass_fraction(context: PriorContext) -> None:
    stars = context.mges["stars"]
    candidate = context.candidate
    # `candidate`'s values are bare (unit-stripped) numbers, so `total_light`
    # is stripped to match -- here to Lsun, paired with `ml`'s Msun / Lsun.
    # Stripping the per-component term before summing (rather than summing
    # the `Quantity` and stripping after) also sidesteps a `jax.numpy.sum`
    # incompatibility with `unxt.Quantity`'s multi-alias physical types
    # (Lsun's `PhysicalType` has two names, `{'power', 'radiant flux'}`).
    total_light = jnp.sum(
        (2 * jnp.pi * stars.I * stars.sigma**2 * stars.q).ustrip("Lsun")
    )
    f = candidate["dh"]["M_200"] / (total_light * candidate["stars"]["ml"])
    # A smooth distribution here (Normal, TruncatedNormal, ...), not a hard
    # Uniform: MCMC/NUTS needs real gradient signal to respect a factor.
    # Uniform.log_prob is flat inside its support and discontinuous at the
    # edges, so NUTS gets no useful gradient and the constraint is not
    # actually enforced in practice -- verified empirically, not just
    # theoretically, during this feature's implementation.
    numpyro.factor("dh_mass_fraction", dist.Normal(0.2, 0.1).log_prob(f))
```

`M_200` here is `dh`'s own ordinary `prior` (e.g. `LogUniform`), not declared
or derived by the plugin -- NFW's existing `concentration_m200`
parameterization (see [Potential](potential.md)) is unmodified; the plugin
only adds a preference on the ratio, never assigns `M_200` directly.

`Prior.sample` (called by `PriorSampler`) traces the composed model once and
picks how to draw from it automatically: `numpyro.infer.Predictive`
(unconditioned prior-predictive sampling) if the model has no factor sites,
otherwise `numpyro.infer.MCMC`/`NUTS` over the prior+factor joint. Both paths
never touch chi2 or orbit integration -- `PriorSampler` doesn't read
`all_models`. Genuine posterior sampling (conditioning on a `Model`'s actual
chi2) needs a bridge turning that into a `numpyro.factor`, which doesn't
exist yet; that's real, separate future work reusing the same composed-model
machinery, not something `PriorSampler` does today.

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
orbit library). A proposed point outside its potential parameters' physical
domain, or an MGE viewing geometry that cannot be deprojected, produces a
`Model` with `valid_potential=False`; a failed orbit integration or weight solve
sets the corresponding later-stage flag to `False`. These candidate-specific
failures stay visible in `AllModels` instead of terminating the run. Programming
contract errors, such as a generator returning a non-`Quantity` parameter, are
not converted into invalid models. The search stops if every model in its first
round fails. After a successful model has been established, failed later rounds
remain recorded but do not by themselves terminate the search.

One pattern recurs through `ModelIterator`'s implementation, worth knowing
before reading the code: the method doing the actual work stays ignorant of
context it doesn't need, and its caller attaches that context afterward.
Solving orbit weights doesn't need to know which `Potential` (or which
rescaling) they came from; assembling the resulting `Model` is what attaches
it. Evaluating a `ParameterSet` doesn't need to know which search round it's
in; `run()` is what stamps `Model.iteration` on once results come back.

## Execution scheduling status

`execution_settings.model_processing_order: model_by_model` is the only
supported processing order. `stage_by_stage` remains a valid prepared setting
so the intended schema is preserved, but `ModelIterator.from_configuration()`
and `run()` raise a clear `NotImplementedError` before doing model-search work.

`orbit_workers` and `weight_workers` are validated, stored in the resolved
configuration, and retained on `ModelIterator`, but no scheduler consumes them
yet, so changing their values currently has no effect. Chi-squared metrics are
calculated as part of the normal model-evaluation path.

Weight-solver retry behavior is separate from execution scheduling and remains
deferred. Configuration currently requires
`weight_solver_settings.reattempt_failures: false`; setting it to `true` is
rejected until retry semantics and execution are implemented.
