"""Proposing the next round of potential parameters to evaluate.

Mostly a signature-only scaffold -- `GridSearchParameterGenerator` still
raises `NotImplementedError` -- except `SinglePointParameterGenerator` and
`build_parameter_generator`, which are fully implemented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import equinox as eqx
from unxt import Quantity

from tnt.all_models import AllModels
from tnt.potential import raw_parameter_dimensions
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
    (matching `tnt.units._validate_parameter_units`/`_normalize_parameters`'s
    own handling of the same distinction). The declared unit's *physical*
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
    allowed range/step) is common to every generator, since even a fixed
    single point is drawn from it.
    `generator_settings` is
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

    def _declared_component_values(
        self, component_name: str, component: Mapping[str, Any]
    ) -> dict[str, Quantity]:
        """This component's parameters, as declared, each as a `Quantity`.

        Every `AbstractParameterGenerator` subclass proposing values reads
        them through here, so "every generator returns unit-ful
        `Quantity`s in their own declared unit" is a property of this base
        class, not something each subclass has to remember to do itself.
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


class GridSearchParameterGenerator(AbstractParameterGenerator):
    """Proposes parameters on a grid, per each parameter's `generator_settings`."""

    _type: ClassVar[str] = "GridSearch"
    _required_generator_settings: ClassVar[frozenset[str]] = frozenset(
        {"delta_chi2_threshold"}
    )

    def generate_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        raise NotImplementedError


class SinglePointParameterGenerator(AbstractParameterGenerator):
    """Proposes one fixed `ParameterSet`, taken from each parameter's declared value.

    Ignores `all_models` and always proposes the same point -- meant for
    evaluating one nominal potential rather than searching parameter space.
    `parameter_space_settings.stopping_criteria` (e.g. `n_new_iter: 1`) is
    what stops the search after it's evaluated. Needs no `generator_settings`.
    """

    _type: ClassVar[str] = "SinglePoint"

    def generate_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        return [
            {
                component_name: self._declared_component_values(
                    component_name, component
                )
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
        potential_settings: A resolved configuration's `potential` section,
            as declared.

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
