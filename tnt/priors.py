"""TNT's user-extensibility mechanism: a run's complete search-space prior.

Two tiers, matching numpyro's own `sample`/`factor` distinction:

- Every non-fixed potential parameter's declared `prior` (a
  `numpyro.distributions` class name + constructor args, see
  `tnt.configuration.validation._validate_prior`) becomes one
  `numpyro.sample` site.
- Each configured prior plugin (a user's own `.py` file, resolved relative
  to `input_directory`, loaded via `load_prior_plugins`) is a plain
  function `def fn(context: PriorContext) -> None` that adds `numpyro.factor`
  terms only -- never a `sample` or `deterministic` call, so a plugin can
  never independently assign or overwrite a parameter's value; it can only
  add a soft log-density preference over values already established by
  ordinary `sample` sites. This is what lets the two tiers compose safely
  with no coordination between them: TNT never has to know which
  `(component, parameter)` a plugin targets. `PriorContext` (see its own
  docstring) is the single argument every plugin takes.

`Prior` composes both tiers into one numpyro model per run and knows how to
draw from it: `numpyro.infer.Predictive` (cheap, unconditioned
prior-predictive sampling) if the model has no factor sites, otherwise
`numpyro.infer.MCMC`/`NUTS` over the prior+factor joint -- still never
touching chi2 or orbit integration, since nothing here reads `AllModels`.

Deliberately independent of `tnt.parameter_generator`: this module only
knows about numpyro and raw declared configuration, not `ParameterSet` or
any `AbstractParameterGenerator` -- `tnt.parameter_generator.PriorSampler`
is a thin consumer that converts what `Prior.sample` returns into TNT's own
types.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import numpyro
import numpyro.distributions
import numpyro.infer
from jax import Array

from tnt.mge import LightMGE, MassMGE


class PriorContext(eqx.Module):
    """Everything a prior plugin may read, passed as its single argument.

    A prior plugin is always

        def <function_name>(context: PriorContext) -> None: ...

    -- callable with one positional argument (a trailing parameter with a
    default, or `*args`, is allowed but never populated). The function may
    only call `numpyro.factor`; it must not `sample`/`deterministic`, and its
    return value is ignored. `load_prior_plugin` rejects any other arity at
    load time.

    Attributes:
        candidate: The parameter values assembled so far for this draw, as
            `{component_name: {parameter_name: value}}`. Values are bare
            (unit-stripped) numbers -- JAX tracers while sampling -- matching
            `Prior.sample`'s output: fixed parameters carry their declared
            `value`, non-fixed ones their sampled value. Plugins run after
            every `numpyro.sample` site, so every non-fixed, prior-bearing
            parameter is present.
        mges: The run's named MGEs (`{name: LightMGE | MassMGE}`), exactly as
            loaded from configuration.

    New fields are added here as plugins gain access to more of a run's
    state; because a plugin only ever names this one argument, that stays a
    backward-compatible change.
    """

    candidate: dict[str, dict[str, Any]]
    mges: Mapping[str, LightMGE | MassMGE]


PriorPlugin = Callable[[PriorContext], None]


def load_prior_plugin(plugin: str, input_directory: str | Path) -> PriorPlugin:
    """Load one configured prior plugin's function from its own `.py` file.

    `plugin` has the shape `<path>:<function_name>` (structurally validated
    by `tnt.configuration.validation._validate_priors`); `path` is resolved
    relative to `input_directory`, the same convention every other external
    file TNT references (MGEs, kinematics, population data) already uses.
    """
    file_part, _, function_name = plugin.partition(":")
    path = (Path(input_directory) / file_part).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load prior plugin file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"{plugin!r} does not name a callable function in {path}.")
    try:
        inspect.signature(function).bind(None)
    except TypeError as error:
        raise TypeError(
            f"prior plugin {plugin!r} must be callable as `{function_name}(context)` "
            f"-- one positional argument, a `tnt.priors.PriorContext`. Its signature "
            f"{inspect.signature(function)} does not accept a single positional "
            f"argument."
        ) from error
    return function


def load_prior_plugins(
    priors_settings: Mapping[str, Mapping[str, str]], input_directory: str | Path
) -> dict[str, PriorPlugin]:
    """Load every configured prior plugin's function, keyed by its config name."""
    return {
        name: load_prior_plugin(settings["plugin"], input_directory)
        for name, settings in priors_settings.items()
    }


def _build_model(
    potential_settings: Mapping[str, Mapping[str, Any]],
    prior_plugins: Mapping[str, PriorPlugin],
    mges: Mapping[str, LightMGE | MassMGE],
) -> Callable[[], dict[str, dict[str, Any]]]:
    """Compose one numpyro model from declared parameter priors and plugins.

    Every non-fixed parameter with a declared `prior` becomes a
    `numpyro.sample` site named `"<component>.<parameter>"`; every fixed
    parameter contributes its bare declared `value` directly (not a sample
    site -- it's the same value on every draw). Plugins run last, each
    receiving a `PriorContext` over the assembled `candidate` and adding
    factor terms only.
    """

    def _model() -> dict[str, dict[str, Any]]:
        candidate: dict[str, dict[str, Any]] = {}
        for component_name, component in potential_settings.items():
            component_values: dict[str, Any] = {}
            for parameter_name, parameter in component.get("parameters", {}).items():
                if parameter.get("fixed", True):
                    component_values[parameter_name] = parameter["value"]
                    continue
                prior = parameter.get("prior")
                if prior is None:
                    continue
                site = f"{component_name}.{parameter_name}"
                distribution_cls = getattr(numpyro.distributions, prior["distribution"])
                component_values[parameter_name] = numpyro.sample(
                    site, distribution_cls(*prior["args"])
                )
            candidate[component_name] = component_values
        context = PriorContext(candidate=candidate, mges=mges)
        for prior_fn in prior_plugins.values():
            prior_fn(context)
        return candidate

    return _model


class Prior:
    """The complete, composed prior for one run's search space.

    Built once (see `tnt.model_iterator.ModelIterator.from_configuration`)
    from a resolved configuration's `potential` section, its configured
    prior plugins, and the run's named MGEs -- independent of which
    `AbstractParameterGenerator` ultimately draws from it.
    """

    def __init__(
        self,
        potential_settings: Mapping[str, Mapping[str, Any]],
        prior_plugins: Mapping[str, PriorPlugin],
        mges: Mapping[str, LightMGE | MassMGE],
    ) -> None:
        self._model = _build_model(potential_settings, prior_plugins, mges)

    def has_factors(self) -> bool:
        """Whether the composed model includes any factor (soft-constraint) term.

        A `numpyro.factor` site is implemented internally as an observed
        `sample` site -- tracing the model once and checking for any
        observed site distinguishes a pure-prior model (safe for
        `Predictive`) from one needing real inference (`MCMC`).
        """
        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(self._model, 0)
        ).get_trace()
        return any(
            site["type"] == "sample" and site.get("is_observed", False)
            for site in traced.values()
        )

    def sample(
        self, key: Array, num_samples: int, num_warmup: int | None = None
    ) -> dict[str, Array]:
        """Draw `num_samples` candidates from the composed model.

        Returns dotted `"component.parameter"` site names -> sampled arrays
        of shape `(num_samples, ...)`, bare (unit-stripped) numbers for
        every non-fixed, prior-bearing parameter -- fixed parameters never
        appear here (they're not sample sites; see `_build_model`) and are
        the caller's responsibility to fill in from their declared `value`.
        Uses `numpyro.infer.Predictive` (unconditioned prior-predictive
        sampling) if `not self.has_factors()`; otherwise
        `numpyro.infer.MCMC(numpyro.infer.NUTS(...))` over the prior+factor
        joint -- still never touching chi2/orbit integration.
        """
        if self.has_factors():
            if num_warmup is None:
                raise ValueError(
                    "num_warmup is required when the prior includes factor terms."
                )
            mcmc = numpyro.infer.MCMC(
                numpyro.infer.NUTS(self._model),
                num_warmup=num_warmup,
                num_samples=num_samples,
                progress_bar=False,
            )
            mcmc.run(key)
            return mcmc.get_samples()
        predictive = numpyro.infer.Predictive(self._model, num_samples=num_samples)
        return predictive(key)
