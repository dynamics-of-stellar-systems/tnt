"""Proposing the next round of potential parameters to evaluate.

Mostly a signature-only scaffold -- `GridSearchParameterGenerator` still
raises `NotImplementedError` -- except `SinglePointParameterGenerator` and
`PriorSampler`, which are fully implemented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import equinox as eqx
import jax.random
from unxt import Quantity

from tnt.all_models import AllModels
from tnt.potential import raw_parameter_dimensions
from tnt.priors import Prior
from tnt.registry import register_typed_class
from tnt.validation import _mapping, _number, _required

ParameterSet = dict[str, dict[str, Quantity]]
"""One proposed point in parameter space: component name -> {parameter name: value}.

Mass parameters (`ml`, `mge_mass_scale`, ...) and shape parameters are both
just entries here -- `AbstractParameterGenerator` doesn't distinguish
between them. `parameter_space_settings.potential_rescalings` is a
separate, `ModelIterator`-owned mechanism layered on top of whichever
`ParameterSet`s the generator proposes; it isn't part of this type. Each
`Quantity` keeps whatever unit its parameter was declared in -- nothing
here converts into a shared internal unit system (see `tnt.potential`'s
module docstring for why).
"""


def _declared_parameter_quantity(
    parameter: Mapping[str, Any], dimension: str | None, path: str
) -> Quantity:
    """Parse one declared potential parameter's `{value, unit}` into a `Quantity`.

    Returns a `Quantity` in its own **declared** unit -- no unit-system
    conversion here, matching `tnt.spatial_binnings`' `_declared_angle_quantity`.
    `dimension=None` (not a recognized parameter) and `dimension="dimensionless"`
    both mean no unit -- `unit` must be absent then, treated identically
    (matching `tnt.units._validate_parameter_units`'s own handling of the
    same distinction). The declared unit's *physical*
    correctness is already guaranteed by `tnt.units.validate_configuration_quantities`,
    which runs during configuration resolution, well before any
    `AbstractParameterGenerator` exists -- this only checks `unit`'s
    presence matches `dimension`, the same structural check that
    validation already enforces, not a re-validation of unit correctness.
    """
    value = _number(_required(parameter, "value", path), f"{path}.value")
    if dimension is None or dimension == "dimensionless":
        if "unit" in parameter:
            raise ValueError(
                f"{path}.unit is not supported because this parameter is "
                "dimensionless or does not yet have a declared dimension."
            )
        return Quantity(value, "")
    if "unit" not in parameter:
        raise ValueError(f"{path} is missing required field: unit.")
    return Quantity(value, parameter["unit"])


class AbstractParameterGenerator(eqx.Module):
    """Proposes the next round of parameters to evaluate, given `AllModels` so far.

    `_type` matches `parameter_space_settings.generator_type`.

    `potential_settings` (a resolved configuration's `potential` section,
    as declared -- every parameter's value, unit, and, for a search, its
    declared `prior`) is common to every generator, since even a fixed
    single point is drawn from it.
    `generator_settings` is
    `parameter_space_settings.generator_settings`; not every generator type
    needs any of it, so `_required_generator_settings` names the subset of
    its keys this `_type` actually reads -- `tnt.configuration.validation`
    mirrors this set (it can't import this module -- see its own note)
    to validate `generator_settings` against whichever `generator_type`
    is configured.
    `max_new_mods_per_iter` (from `parameter_space_settings.stopping_criteria`,
    shared and generic -- not any one generator's own setting) caps how many
    candidates `generate_parameters` returns per call, uniformly across
    every generator regardless of how many its own `_propose_free_parameters`
    happens to produce.
    """

    _type: ClassVar[str]
    _required_generator_settings: ClassVar[frozenset[str]] = frozenset()

    potential_settings: Mapping[str, Mapping[str, Any]]
    generator_settings: Mapping[str, Any]
    max_new_mods_per_iter: int

    def generate_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        """Propose the next round of `ParameterSet`s, capped at `max_new_mods_per_iter`.

        Args:
            all_models: Every model evaluated so far (empty on the first
                iteration).
        """
        return list(self._propose_free_parameters(all_models))[
            : self.max_new_mods_per_iter
        ]

    def _propose_free_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        """This generator's own proposal logic, before the shared size cap.

        Args:
            all_models: Every model evaluated so far (empty on the first
                iteration).
        """
        raise NotImplementedError

    def _declared_component_values(
        self, component_name: str, component: Mapping[str, Any]
    ) -> dict[str, Quantity]:
        """This component's parameters, as declared, each as a `Quantity`.

        Every `AbstractParameterGenerator` subclass proposing fixed/declared
        (rather than sampled) values reads them through here, so "every such
        generator returns unit-ful `Quantity`s in their own declared unit"
        is a property of this base class, not something each subclass has
        to remember to do itself.
        """
        dimensions = raw_parameter_dimensions(
            component.get("type"), component.get("parameterization")
        )
        return {
            parameter_name: _declared_parameter_quantity(
                _mapping(
                    parameter,
                    f"potential.{component_name}.parameters.{parameter_name}",
                ),
                dimensions.get(parameter_name),
                f"potential.{component_name}.parameters.{parameter_name}",
            )
            for parameter_name, parameter in component.get("parameters", {}).items()
        }


_PARAMETER_GENERATOR_REGISTRY: dict[str, type[AbstractParameterGenerator]] = {}


def register_parameter_generator(
    cls: type[AbstractParameterGenerator],
) -> type[AbstractParameterGenerator]:
    """Register one concrete parameter generator for configuration dispatch."""
    return register_typed_class(
        _PARAMETER_GENERATOR_REGISTRY,
        cls,
        family="parameter generator",
    )


def get_parameter_generator_class(
    type_name: str,
) -> type[AbstractParameterGenerator] | None:
    """Return the registered generator class for ``type_name``, if any."""
    return _PARAMETER_GENERATOR_REGISTRY.get(type_name)


def parameter_generator_type_names() -> frozenset[str]:
    """Return every explicitly registered parameter-generator type name."""
    return frozenset(_PARAMETER_GENERATOR_REGISTRY)


def parameter_generator_required_settings() -> dict[str, frozenset[str]]:
    """Return required configuration-setting names for every generator type."""
    return {
        type_name: cls._required_generator_settings
        for type_name, cls in _PARAMETER_GENERATOR_REGISTRY.items()
    }


@register_parameter_generator
class GridSearchParameterGenerator(AbstractParameterGenerator):
    """Proposes parameters on a grid, per each parameter's declared `prior`."""

    _type: ClassVar[str] = "GridSearch"
    _required_generator_settings: ClassVar[frozenset[str]] = frozenset(
        {"delta_chi2_threshold"}
    )

    def _propose_free_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        raise NotImplementedError


@register_parameter_generator
class SinglePointParameterGenerator(AbstractParameterGenerator):
    """Proposes one fixed `ParameterSet`, taken from each parameter's declared value.

    Ignores `all_models` and always proposes the same point -- meant for
    evaluating one nominal potential rather than searching parameter space.
    `parameter_space_settings.stopping_criteria` (e.g. `n_new_iter: 1`) is
    what stops the search after it's evaluated. Needs no `generator_settings`.
    Never looks at `priors:` -- a single fixed point needs every parameter
    fully declared directly (`value`/`fixed`), not sampled.
    """

    _type: ClassVar[str] = "SinglePoint"

    def _propose_free_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        return [
            {
                component_name: self._declared_component_values(
                    component_name, component
                )
                for component_name, component in self.potential_settings.items()
            }
        ]


def _parameter_sets_from_samples(
    samples: Mapping[str, Any],
    potential_settings: Mapping[str, Mapping[str, Any]],
) -> list[ParameterSet]:
    """Convert `tnt.priors.Prior.sample`'s output into `ParameterSet`s.

    `samples` holds one array per non-fixed, prior-bearing parameter (dotted
    `"component.parameter"` site names -> bare, unit-stripped draws, see
    `Prior.sample`); fixed parameters aren't sample sites at all, so their
    declared `value` is filled in directly here instead. Every value is
    re-wrapped as a `Quantity` in its parameter's own declared `unit`
    (absent means dimensionless), matching `_declared_parameter_quantity`'s
    convention -- nothing here converts into a shared internal unit system.
    """
    num_samples = len(next(iter(samples.values()))) if samples else 1
    parameter_sets: list[ParameterSet] = []
    for index in range(num_samples):
        candidate: ParameterSet = {}
        for component_name, component in potential_settings.items():
            component_values: dict[str, Quantity] = {}
            for parameter_name, parameter in component.get("parameters", {}).items():
                site = f"{component_name}.{parameter_name}"
                value = samples[site][index] if site in samples else parameter["value"]
                component_values[parameter_name] = Quantity(
                    value, parameter.get("unit", "")
                )
            candidate[component_name] = component_values
        parameter_sets.append(candidate)
    return parameter_sets


@register_parameter_generator
class PriorSampler(AbstractParameterGenerator):
    """Proposes parameters by sampling from a `tnt.priors.Prior`.

    Ignores `all_models` -- `Prior.sample` doesn't condition on anything
    data-dependent this round; that's a future chi2-conditioned generator's
    job. A thin adapter: all model composition, factor detection, and the
    `Predictive`-vs-`MCMC` choice live on `Prior` itself, reusable by
    whatever generator needs them next -- this class only derives a PRNG
    key, calls `Prior.sample`, and converts the result into `ParameterSet`s.
    """

    _type: ClassVar[str] = "PriorSampler"
    _required_generator_settings: ClassVar[frozenset[str]] = frozenset(
        {"num_warmup", "seed"}
    )

    prior: Prior
    seed: int
    num_warmup: int

    def _propose_free_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        del all_models
        key = jax.random.PRNGKey(self.seed)
        samples = self.prior.sample(
            key, num_samples=self.max_new_mods_per_iter, num_warmup=self.num_warmup
        )
        return _parameter_sets_from_samples(samples, self.potential_settings)


def build_parameter_generator(
    parameter_space_settings: Mapping[str, Any],
    potential_settings: Mapping[str, Mapping[str, Any]],
    prior: Prior | None = None,
) -> AbstractParameterGenerator:
    """Build the `AbstractParameterGenerator` named by a resolved configuration.

    Args:
        parameter_space_settings: A resolved configuration's
            `parameter_space_settings` section.
        potential_settings: A resolved configuration's `potential` section,
            as declared.
        prior: This run's composed `tnt.priors.Prior`, e.g. from
            `tnt.model_iterator.ModelIterator.from_configuration` -- only
            required when `generator_type` is `"PriorSampler"`.

    Returns:
        The `AbstractParameterGenerator` matching
        `parameter_space_settings.generator_type`.
    """
    generator_type = parameter_space_settings["generator_type"]
    generator_cls = get_parameter_generator_class(generator_type)
    if generator_cls is None:
        allowed = ", ".join(sorted(parameter_generator_type_names()))
        raise ValueError(
            "Unknown parameter_space_settings.generator_type: "
            f"{generator_type!r}; expected one of: {allowed}."
        )
    generator_settings = parameter_space_settings["generator_settings"]
    max_new_mods_per_iter = parameter_space_settings["stopping_criteria"][
        "max_new_mods_per_iter"
    ]
    if generator_cls is PriorSampler:
        if prior is None:
            raise ValueError(
                "build_parameter_generator: generator_type 'PriorSampler' "
                "requires a built Prior."
            )
        return PriorSampler(
            potential_settings=potential_settings,
            generator_settings=generator_settings,
            max_new_mods_per_iter=max_new_mods_per_iter,
            prior=prior,
            seed=generator_settings["seed"],
            num_warmup=generator_settings["num_warmup"],
        )
    return generator_cls(
        potential_settings=potential_settings,
        generator_settings=generator_settings,
        max_new_mods_per_iter=max_new_mods_per_iter,
    )
