"""Proposing the next round of potential parameters to evaluate.

Mostly a signature-only scaffold -- `GridSearchParameterGenerator` still
raises `NotImplementedError` -- except `SinglePointParameterGenerator` and
`build_parameter_generator`, which are fully implemented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import equinox as eqx

from tnt.all_models import AllModels

ParameterSet = dict[str, dict[str, float]]
"""One proposed point in parameter space: component name -> {parameter name: value}.

Mass parameters (`ml`, `mge_mass_scale`, ...) and shape parameters are both
just entries here -- `AbstractParameterGenerator` doesn't distinguish
between them. `parameter_space_settings.potential_rescalings` is a
separate, `ModelIterator`-owned mechanism layered on top of whichever
`ParameterSet`s the generator proposes; it isn't part of this type.
"""


class AbstractParameterGenerator(eqx.Module):
    """Proposes the next round of parameters to evaluate, given `AllModels` so far.

    `_type` matches `parameter_space_settings.generator_type`.

    `potential_settings` (a resolved configuration's `potential` section --
    every parameter's declared value, and, for a search, its allowed
    range/step) is common to every generator, since even a fixed single
    point is drawn from it. `generator_settings` is
    `parameter_space_settings.generator_settings`; not every generator type
    needs any of it, so `_required_generator_settings` names the subset of
    its keys this `_type` actually reads -- `configuration_validation`
    mirrors this set (it can't import this module -- see its own note)
    to validate `generator_settings` against whichever `generator_type`
    is configured.
    """

    _type: ClassVar[str]
    _required_generator_settings: ClassVar[frozenset[str]] = frozenset()

    potential_settings: Mapping[str, Mapping[str, Any]]
    generator_settings: Mapping[str, Any]

    def generate_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        """Propose the next round of `ParameterSet`s.

        Args:
            all_models: Every model evaluated so far (empty on the first
                iteration).
        """
        raise NotImplementedError


class GridSearchParameterGenerator(AbstractParameterGenerator):
    """Proposes parameters on a grid, per each parameter's `generator_settings`."""

    _type: ClassVar[str] = "GridSearch"
    _required_generator_settings: ClassVar[frozenset[str]] = frozenset(
        {"delta_chi2_threshold"}
    )

    def generate_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        raise NotImplementedError


class SinglePointParameterGenerator(AbstractParameterGenerator):
    """Proposes one fixed `ParameterSet`, taken from each parameter's `value`.

    Ignores `all_models` and always proposes the same point -- meant for
    evaluating one nominal potential rather than searching parameter space.
    `parameter_space_settings.stopping_criteria` (e.g. `n_new_iter: 1`) is
    what stops the search after it's evaluated. Needs no `generator_settings`.
    """

    _type: ClassVar[str] = "SinglePoint"

    def generate_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        return [
            {
                component_name: {
                    parameter_name: parameter["value"]
                    for parameter_name, parameter in component.get(
                        "parameters", {}
                    ).items()
                }
                for component_name, component in self.potential_settings.items()
            }
        ]


_GENERATOR_CLASSES = (GridSearchParameterGenerator, SinglePointParameterGenerator)


def build_parameter_generator(
    parameter_space_settings: Mapping[str, Any],
    potential_settings: Mapping[str, Mapping[str, Any]],
) -> AbstractParameterGenerator:
    """Build the `AbstractParameterGenerator` named by a resolved configuration.

    Args:
        parameter_space_settings: A resolved configuration's
            `parameter_space_settings` section.
        potential_settings: A resolved configuration's `potential` section.

    Returns:
        The `AbstractParameterGenerator` matching
        `parameter_space_settings.generator_type`.
    """
    generator_type = parameter_space_settings["generator_type"]
    for generator_cls in _GENERATOR_CLASSES:
        if generator_type == generator_cls._type:
            return generator_cls(
                potential_settings=potential_settings,
                generator_settings=parameter_space_settings["generator_settings"],
            )
    raise ValueError(
        f"Unknown parameter_space_settings.generator_type: {generator_type!r}"
    )
