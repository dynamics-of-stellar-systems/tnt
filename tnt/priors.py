"""TNT's user-extensibility mechanism: a run's complete search-space prior.

Three tiers, matching numpyro's own `sample`/`factor` distinction:

- Every non-fixed potential parameter's declared `prior` (a
  `numpyro.distributions` class name + constructor args, see
  `tnt.configuration.validation._validate_prior`) becomes one
  `numpyro.sample` site.
- A component whose resolved `(type, parameterization)` has a registered
  default prior (`tnt.potential.registry.DefaultPriorSpec` --
  `get_default_prior`) is always sampled from it for that spec's parameters
  as a whole joint draw, for a validity constraint that can't be expressed
  as independent per-parameter `prior:` entries (e.g. the `pqu`
  parameterization's data-dependent bounds on `q` and `u`, which depend on
  the MGE's own anchor axis ratio, not just sibling parameters). These
  parameters no longer require a declared `prior`; if one is declared
  anyway, it doesn't replace the default-prior sample, it adds a
  `numpyro.factor` scoring the sampled value -- see `_build_model`. A
  default prior is itself built only from `numpyro.sample` statements,
  never a `factor`, so a component covered only by one (no plugins) stays
  on the cheap `Predictive` path below.
- Each configured prior plugin (a user's own `.py` file, resolved relative
  to `input_directory`, loaded via `load_prior_plugins`) is a plain
  function `def fn(context: PriorContext) -> None` that adds `numpyro.factor`
  terms only -- never a `sample` or `deterministic` call, so a plugin can
  never independently assign or overwrite a parameter's value; it can only
  add a soft log-density preference over values already established by
  ordinary `sample` sites. This is what lets the tiers compose safely with
  no coordination between them: TNT never has to know which
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
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.diagnostics
import numpyro.distributions
import numpyro.infer
from jax import Array
from unxt import AbstractUnitSystem, Quantity

from tnt.mge import LightMGE, MassMGE
from tnt.potential import Potential, get_default_prior
from tnt.potential.components import ResolvedPotentialComponent
from tnt.potential.registry import DefaultPriorSpec


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

    A plugin that needs a derived quantity of the whole potential (an
    enclosed mass, a circular velocity) calls `build_potential()`: it
    assembles this draw's `tnt.potential.Potential` from `candidate` via
    `Potential.build_with_validity`, so it runs inside the traced model.
    Evaluate it with `build_potential().to_galax(context.unit_system)`. The
    run's `resolved` potential and cosmology are captured when the `Prior`
    is built -- a plugin never handles those. The first call in a draw
    (by `build_potential()`, `build_potential_with_validity()`, or
    `factor_if_valid()` -- whichever runs first) registers a `numpyro.factor`
    site giving an invalid geometry exactly `-inf` log-probability (the
    formal statement that point has no support, not a `nan` relying on
    gradient propagation to get rejected) and caches the potential/validity
    pair, so every later call in the same draw -- whether from the same
    plugin or another -- reuses it rather than rebuilding and registering a
    second site.

    A derived quantity computed *from* that potential can still be
    individually invalid in a way the placeholder potential itself doesn't
    capture (e.g. a derived formula dividing by a zero-placeholder
    parameter) -- on an invalid draw this can produce a `NaN` factor that
    poisons the model's total log density even though the `-inf` validity
    factor is separately registered: `-inf + NaN` is `NaN`, not `-inf`, so
    this is not automatically safe just because the validity factor exists.
    Guard any such computation with `factor_if_valid()` instead of calling
    `numpyro.factor()` directly.

    Attributes (continued):
        unit_system: The run's internal unit system, for
            `build_potential().to_galax(...)`; `None` when the `Prior` was
            built without potential context.

    New fields are added here as plugins gain access to more of a run's
    state; because a plugin only ever names this one argument, that stays a
    backward-compatible change.
    """

    candidate: dict[str, dict[str, Any]]
    mges: Mapping[str, LightMGE | MassMGE]
    unit_system: AbstractUnitSystem | None = None
    _build_potential_with_validity: (
        Callable[[], tuple[Potential, Array]] | None
    ) = eqx.field(default=None, static=True)

    def build_potential_with_validity(self) -> tuple[Potential, Array]:
        """This draw's `(Potential, valid)` pair, both built together via
        `Potential.build_with_validity`.

        The first call in a draw (by this method, `build_potential()`, or
        `factor_if_valid()`) registers this draw's validity as a
        `numpyro.factor` site (see this class's own docstring) and caches
        the pair; later calls in the same draw, by any of the three, are
        free.

        Raises:
            RuntimeError: If this `Prior` was constructed without the run's
                resolved potential, cosmology and unit system (e.g. in a
                unit test exercising only the sample sites).
        """
        if self._build_potential_with_validity is None:
            raise RuntimeError(
                "context.build_potential() needs the run's resolved potential, "
                "cosmological parameters and unit system; this Prior was built "
                "without them."
            )
        return self._build_potential_with_validity()

    def build_potential(self) -> Potential:
        """This draw's `Potential` alone -- see `build_potential_with_validity`."""
        return self.build_potential_with_validity()[0]

    def factor_if_valid(self, name: str, compute: Callable[[], Array]) -> None:
        """Register `name`'s factor as `compute()`'s log-density -- but only
        ever evaluate `compute` when this draw's potential is valid.

        Uses `jax.lax.cond`, not `jnp.where`: `lax.cond` evaluates (and
        differentiates) only the taken branch, so a `compute` that's only
        well-defined on a valid potential (e.g. dividing by a parameter
        that's a zero placeholder on an invalid draw) never actually runs,
        and so can never produce a `NaN` that silently overrides the
        separate `-inf` validity factor (`jnp.where` would still compute
        and differentiate both branches, leaking the `NaN` back in even
        though its *selected* value looks fine).
        """
        _, valid = self.build_potential_with_validity()
        log_prob = jax.lax.cond(
            valid, lambda _: compute(), lambda _: jnp.asarray(0.0), None
        )
        numpyro.factor(name, log_prob)


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


def _make_potential_builder(
    potential_settings: Mapping[str, Mapping[str, Any]],
    resolved_potential: Mapping[str, ResolvedPotentialComponent],
    cosmological_parameters: Mapping[str, Quantity],
    unit_system: AbstractUnitSystem,
) -> Callable[[Mapping[str, Any]], tuple[Potential, Array]]:
    """A closure that turns one draw's bare `candidate` into a `Potential`
    and its scalar validity flag.

    Mirrors `tnt.parameter_generator.samples._parameter_sets_from_samples`: each
    value is re-wrapped as a `Quantity` in its parameter's own declared
    `unit` (absent -> dimensionless), then `Potential.build_with_validity`
    runs, which is itself JAX-traceable. `unit_system` isn't
    `build_with_validity`'s concern (it's `Potential.to_galax`'s), but the
    caller passes it here so `Prior` has one place that gates the whole
    capability on all three being present. Registering the validity as a
    `numpyro.factor` site is `_build_model`'s concern, not this one's --
    this function only builds.
    """
    del unit_system
    units: dict[str, dict[str, str]] = {
        name: {
            parameter_name: parameter.get("unit", "")
            for parameter_name, parameter in component.get("parameters", {}).items()
        }
        for name, component in potential_settings.items()
    }

    def build(candidate: Mapping[str, Any]) -> tuple[Potential, Array]:
        parameter_values = {
            name: {
                parameter_name: Quantity(candidate[name][parameter_name], unit)
                for parameter_name, unit in component_units.items()
                if parameter_name in candidate.get(name, {})
            }
            for name, component_units in units.items()
        }
        return Potential.build_with_validity(
            resolved_potential, parameter_values, cosmological_parameters
        )

    return build


def _resolve_prior_distribution(
    prior: Mapping[str, Any], site: str
) -> numpyro.distributions.Distribution:
    """Build and validate one parameter's declared prior distribution.

    `tnt.configuration.validation._validate_prior` only checks structure
    (`distribution` is a string, `args` are numbers) -- it deliberately
    never imports `numpyro`, so resolving whether `distribution` actually
    names a usable distribution is this function's job, the first point
    that does import `numpyro`.

    `distribution` must name a real `numpyro.distributions.Distribution`
    subclass, not just any attribute of the module. The distribution is
    then constructed with `validate_args=True`, so every argument that
    numpyro itself declares a precise `arg_constraints` entry for actually
    gets checked (e.g. `Normal`'s `scale > 0`, `LogUniform`'s `low`/`high >
    0`) -- which otherwise silently isn't, since nothing before this point
    passes `validate_args` at all. `Uniform`/`LogUniform` additionally declare
    `low`/`high` as `constraints.dependent` -- numpyro's own escape hatch
    for "the framework can't check this generically, the distribution must
    do it itself in `__init__`" -- and neither actually does, so reversed
    or degenerate bounds (`low >= high`) pass silently even with
    `validate_args=True`; check that ourselves.
    """
    distribution_name = prior["distribution"]
    distribution_cls = getattr(numpyro.distributions, distribution_name, None)
    if not (
        isinstance(distribution_cls, type)
        and issubclass(distribution_cls, numpyro.distributions.Distribution)
    ):
        raise TypeError(
            f"{site}: prior distribution {distribution_name!r} is not a "
            "numpyro.distributions class."
        )
    try:
        distribution = distribution_cls(*prior["args"], validate_args=True)
    except Exception as error:
        raise ValueError(
            f"{site}: invalid prior distribution arguments {prior['args']} for "
            f"{distribution_name}: {error}"
        ) from error
    low = getattr(distribution, "low", None)
    high = getattr(distribution, "high", None)
    if low is not None and high is not None and not (low < high):
        raise ValueError(
            f"{site}: prior bounds must be ordered and distinct, got "
            f"low={low}, high={high}."
        )
    return distribution


def _build_model(
    potential_settings: Mapping[str, Mapping[str, Any]],
    prior_plugins: Mapping[str, PriorPlugin],
    mges: Mapping[str, LightMGE | MassMGE],
    build_potential: Callable[[Mapping[str, Any]], tuple[Potential, Array]] | None,
    unit_system: AbstractUnitSystem | None,
) -> Callable[[], dict[str, dict[str, Any]]]:
    """Compose one numpyro model from declared parameter priors, default
    priors, and plugins.

    Every non-fixed parameter with a declared `prior` becomes a
    `numpyro.sample` site named `"<component>.<parameter>"`; every fixed
    parameter contributes its bare declared `value` directly (not a sample
    site -- it's the same value on every draw). A component whose resolved
    `(type, parameterization)` has a registered default prior (see
    `tnt.potential.registry.DefaultPriorSpec`) always has its `sample`
    called for that spec's parameters, as one joint draw -- no opt-in, and
    no declared `prior` is required for them. Whichever of them the
    component leaves fixed are passed to `sample` as already `known`, so
    it draws only the rest (and may still use a fixed sibling's value, the
    same as a freshly-sampled one, in its own prior). A user-declared
    `prior` on a non-fixed covered parameter doesn't replace that sampling;
    it becomes an additional `numpyro.factor` site scoring the
    already-sampled value, so it never adds a second, competing way to
    assign the parameter. Plugins run last, each receiving a `PriorContext`
    over the assembled `candidate` (and, when available, `build_potential`)
    and adding factor terms only.

    Each site's distribution is resolved and validated once here (via
    `_resolve_prior_distribution`), not rebuilt and re-validated on every
    trace of the returned model -- so a malformed prior raises immediately,
    when the `Prior` is constructed, rather than lazily on first `sample()`.

    `build_potential` (if given) is wrapped in a per-draw, memoized,
    zero-arg closure before being handed to `PriorContext`: the first
    plugin to call `context.build_potential()` in a draw triggers the real
    build and registers that draw's validity as a `numpyro.factor` site
    (`-inf` log-probability for an invalid geometry, `0` otherwise); every
    later call in the same draw -- any plugin -- reuses the cached result
    rather than rebuilding and registering a second site.
    """
    default_prior_groups: list[
        tuple[str, DefaultPriorSpec, LightMGE | MassMGE | None]
    ] = []
    covered_parameters: dict[str, set[str]] = {}
    for component_name, component in potential_settings.items():
        component_type = component.get("type")
        parameterization = component.get("parameterization")
        if not (isinstance(component_type, str) and isinstance(parameterization, str)):
            continue
        spec = get_default_prior(component_type, parameterization)
        if spec is None:
            continue
        mge_name = component.get("mge")
        mge = mges[mge_name] if mge_name is not None else None
        default_prior_groups.append((component_name, spec, mge))
        covered_parameters[component_name] = set(spec.parameter_names)

    sample_sites: list[tuple[str, str, numpyro.distributions.Distribution]] = []
    fixed_values: list[tuple[str, str, Any]] = []
    default_prior_user_factors: list[
        tuple[str, str, numpyro.distributions.Distribution]
    ] = []
    for component_name, component in potential_settings.items():
        component_covered = covered_parameters.get(component_name, set())
        for parameter_name, parameter in component.get("parameters", {}).items():
            if parameter.get("fixed", True):
                fixed_values.append(
                    (component_name, parameter_name, parameter["value"])
                )
                continue
            prior = parameter.get("prior")
            if parameter_name in component_covered:
                if prior is not None:
                    site = f"{component_name}.{parameter_name}"
                    distribution = _resolve_prior_distribution(prior, site)
                    default_prior_user_factors.append(
                        (component_name, parameter_name, distribution)
                    )
                continue
            if prior is None:
                continue
            site = f"{component_name}.{parameter_name}"
            distribution = _resolve_prior_distribution(prior, site)
            sample_sites.append((component_name, parameter_name, distribution))

    def _model() -> dict[str, dict[str, Any]]:
        candidate: dict[str, dict[str, Any]] = {
            component_name: {} for component_name in potential_settings
        }
        for component_name, parameter_name, value in fixed_values:
            candidate[component_name][parameter_name] = value
        for component_name, parameter_name, distribution in sample_sites:
            site = f"{component_name}.{parameter_name}"
            candidate[component_name][parameter_name] = numpyro.sample(
                site, distribution
            )
        for component_name, spec, mge in default_prior_groups:
            component_candidate = candidate[component_name]
            known = {
                name: component_candidate[name]
                for name in spec.parameter_names
                if name in component_candidate
            }
            component_candidate.update(spec.sample(component_name, known, mge))
        for component_name, parameter_name, distribution in default_prior_user_factors:
            site = f"{component_name}.{parameter_name}_prior"
            numpyro.factor(
                site, distribution.log_prob(candidate[component_name][parameter_name])
            )

        cached: list[tuple[Potential, Array]] = []

        def build_potential_with_validity_once() -> tuple[Potential, Array]:
            if not cached:
                potential, valid = build_potential(candidate)
                numpyro.factor(
                    "_valid_potential_geometry", jnp.where(valid, 0.0, -jnp.inf)
                )
                cached.append((potential, valid))
            return cached[0]

        context = PriorContext(
            candidate=candidate,
            mges=mges,
            unit_system=unit_system,
            _build_potential_with_validity=(
                build_potential_with_validity_once
                if build_potential is not None
                else None
            ),
        )
        for prior_fn in prior_plugins.values():
            prior_fn(context)
        return candidate

    return _model


class PriorDiagnostics(NamedTuple):
    """Diagnostics for one `Prior.sample_with_diagnostics` draw.

    Always produced, whether sampling ran `numpyro.infer.MCMC`/`NUTS`
    (`Prior.has_factors()` is `True`) or an unconditioned
    `numpyro.infer.Predictive` draw. A `Predictive` draw is exact and
    independent by construction, so its `diverging` is trivially all
    `False`, and its `ess`/`split_rhat` are computed by treating the draws
    as a single pseudo-chain (the same `numpyro.diagnostics` functions the
    MCMC branch uses, applied to i.i.d. samples rather than a real chain)
    -- not a convergence check there (there's no chain to converge), just
    confirmation that the draws behave like the independent samples they're
    supposed to be: `ess` close to `num_samples`, `split_rhat` close to `1`.
    """

    diverging: Array
    """Per-draw divergence flag, shape `(num_samples,)`."""
    ess: dict[str, float]
    """Effective sample size per `"component.parameter"` site name."""
    split_rhat: dict[str, float]
    """Split R-hat per `"component.parameter"` site name."""


class Prior:
    """The complete, composed prior for one run's search space.

    Built once (see `tnt.model_iterator.ModelIterator.from_configuration`)
    from a resolved configuration's `potential` section, its configured
    prior plugins, and the run's named MGEs -- independent of which
    `AbstractParameterGenerator` ultimately draws from it.

    `resolved_potential` / `cosmological_parameters` / `unit_system` are
    optional: supplying all three lets a plugin call
    `PriorContext.build_potential()` (they're captured, never plugin-facing);
    omitting them -- as a unit test exercising only the sample sites may --
    just makes that method raise.
    """

    def __init__(
        self,
        potential_settings: Mapping[str, Mapping[str, Any]],
        prior_plugins: Mapping[str, PriorPlugin],
        mges: Mapping[str, LightMGE | MassMGE],
        *,
        resolved_potential: Mapping[str, ResolvedPotentialComponent] | None = None,
        cosmological_parameters: Mapping[str, Quantity] | None = None,
        unit_system: AbstractUnitSystem | None = None,
    ) -> None:
        build_potential: Callable[[Mapping[str, Any]], Potential] | None = None
        if (
            resolved_potential is not None
            and cosmological_parameters is not None
            and unit_system is not None
        ):
            build_potential = _make_potential_builder(
                potential_settings,
                resolved_potential,
                cosmological_parameters,
                unit_system,
            )
        self._build_potential = build_potential
        self._potential_settings = potential_settings
        self._model = _build_model(
            potential_settings,
            prior_plugins,
            mges,
            build_potential,
            unit_system if build_potential is not None else None,
        )

    def unconstrain(self, values: Mapping[str, Any]) -> dict[str, Array]:
        """Map one complete prior draw to NumPyro's unconstrained coordinates.

        Replays conditional supports for each draw; do not cache transforms
        from a single trace when bounds depend on sibling parameters. Point
        masses use NumPyro's identity transform; consumers must hold constant
        coordinates fixed. Discrete sites cannot be continuously optimized.
        """
        model = numpyro.handlers.seed(self._model, 0)
        trace = numpyro.handlers.trace(
            numpyro.handlers.substitute(model, data=values)
        ).get_trace()
        for name, site in trace.items():
            if site["type"] != "sample" or site.get("is_observed", False):
                continue
            if name not in values:
                raise ValueError(f"Missing prior sample site: {name}")
            if site["fn"].support.is_discrete:
                raise ValueError(
                    f"Unconstrained support points require continuous sites: {name}"
                )
        return numpyro.infer.util.unconstrain_fn(model, (), {}, values)

    def constrain(self, values: Mapping[str, Any]) -> dict[str, Array]:
        """Map one unconstrained position back, recomputing conditional bounds.

        This enforces distribution supports, not arbitrary plugin factors or
        extra geometry restrictions. Check `feasibility` on the result.
        """
        return numpyro.infer.util.constrain_fn(
            numpyro.handlers.seed(self._model, 0), (), {}, values
        )

    def feasibility(self, samples: Mapping[str, Any]) -> Array:
        """Check original-coordinate sites against conditional supports and factors.

        Substitution replays conditional default priors and every plugin. Explicit
        support checks are needed because some log_prob methods do not mask
        out-of-support values. Geometry is checked even without a plugin.
        This constructs potentials when context is available, never orbit
        libraries, likelihoods, or fitted models.
        """
        model = numpyro.handlers.substitute(self._model, data=samples)
        traced = numpyro.handlers.trace(numpyro.handlers.seed(model, 0)).get_trace()
        valid = jnp.asarray(True)
        for name, site in traced.items():
            if site["type"] != "sample":
                continue
            if not site.get("is_observed", False) and name not in samples:
                raise ValueError(f"Missing prior sample site: {name}")
            distribution, value = site["fn"], site["value"]
            valid = valid & jnp.all(distribution.support(value))
            valid = valid & jnp.all(jnp.isfinite(distribution.log_prob(value)))
        if self._build_potential is not None:
            candidate = {
                component: {
                    parameter: samples.get(
                        f"{component}.{parameter}", spec.get("value")
                    )
                    for parameter, spec in settings.get("parameters", {}).items()
                }
                for component, settings in self._potential_settings.items()
            }
            _, geometry_valid = self._build_potential(candidate)
            valid = valid & jnp.all(geometry_valid)
        return valid

    def is_feasible(self, samples: Mapping[str, Any]) -> bool:
        """Concrete Python wrapper around the JAX-compatible feasibility check."""
        return bool(self.feasibility(samples))

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
        self,
        key: Array,
        num_samples: int,
        num_warmup: int | None = None,
        target_accept_prob: float = 0.8,
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
        joint -- still never touching chi2/orbit integration. `target_accept_prob`
        (numpyro's own default) only affects the `MCMC`/`NUTS` branch's step-size
        adaptation; it's ignored when `Predictive` is used. A thin wrapper
        over `sample_with_diagnostics` that discards its diagnostics.
        """
        samples, _ = self.sample_with_diagnostics(
            key, num_samples, num_warmup, target_accept_prob
        )
        return samples

    def sample_with_diagnostics(
        self,
        key: Array,
        num_samples: int,
        num_warmup: int | None = None,
        target_accept_prob: float = 0.8,
    ) -> tuple[dict[str, Array], PriorDiagnostics]:
        """Like `sample`, but also returns diagnostics (see `PriorDiagnostics`).

        When `sample` would use `Predictive` instead of MCMC (`not
        self.has_factors()`), the draws are exact and independent by
        construction -- the diagnostics there are a sanity check on that
        independence, not a convergence check. See
        `tnt.parameter_generator.diagnostics.prior.plot_prior_diagnostics` for a
        human-reviewed visualization built from this -- divergence/ESS/
        R-hat thresholds can't distinguish a genuinely broken chain from a
        prior that's simply as wide as the science requires, so this is
        deliberately not an automated pass/fail gate.

        Raises:
            ValueError: If `num_samples < 4` -- both branches below compute
                a split R-hat over the draws (`numpyro.diagnostics
                .split_gelman_rubin`), which needs at least 4 to split a
                chain in half and compare the halves; this turns that
                precondition into a clear, named error here rather than a
                bare `AssertionError` from inside numpyro with no indication
                of which call or argument is at fault.
        """
        if num_samples < 4:
            raise ValueError(
                "sample_with_diagnostics requires num_samples >= 4 to compute a "
                f"split R-hat diagnostic; got num_samples={num_samples}."
            )
        if not self.has_factors():
            predictive = numpyro.infer.Predictive(self._model, num_samples=num_samples)
            samples = predictive(key)
            # Treat the i.i.d. draws as a single pseudo-chain (shape
            # `(1, num_samples, ...)`) so the same diagnostics functions
            # the MCMC branch uses below apply unchanged.
            pseudo_chain = {name: value[None, ...] for name, value in samples.items()}
            diagnostics = PriorDiagnostics(
                diverging=jnp.zeros(num_samples, dtype=bool),
                ess={
                    name: float(
                        jnp.min(
                            jnp.asarray(numpyro.diagnostics.effective_sample_size(value))
                        )
                    )
                    for name, value in pseudo_chain.items()
                },
                split_rhat={
                    name: float(
                        jnp.max(
                            jnp.asarray(numpyro.diagnostics.split_gelman_rubin(value))
                        )
                    )
                    for name, value in pseudo_chain.items()
                },
            )
            return samples, diagnostics
        if num_warmup is None:
            raise ValueError(
                "num_warmup is required when the prior includes factor terms."
            )
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(self._model, target_accept_prob=target_accept_prob),
            num_warmup=num_warmup,
            num_samples=num_samples,
            progress_bar=False,
        )
        mcmc.run(key, extra_fields=("diverging",))
        samples_by_chain = mcmc.get_samples(group_by_chain=True)
        diagnostics = PriorDiagnostics(
            diverging=mcmc.get_extra_fields()["diverging"],
            ess={
                name: float(
                    jnp.min(jnp.asarray(numpyro.diagnostics.effective_sample_size(value)))
                )
                for name, value in samples_by_chain.items()
            },
            split_rhat={
                name: float(
                    jnp.max(jnp.asarray(numpyro.diagnostics.split_gelman_rubin(value)))
                )
                for name, value in samples_by_chain.items()
            },
        )
        return mcmc.get_samples(), diagnostics
