"""Proposals from explicitly declared parameter values."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from tnt.all_models import AllModels
from tnt.parameter_generator.base import AbstractParameterGenerator, ParameterSet
from tnt.parameter_generator.registry import register_parameter_generator


@register_parameter_generator
class SinglePointParameterGenerator(AbstractParameterGenerator):
    """Proposes one fixed `ParameterSet`, taken from each parameter's declared value.

    The shared duplicate filter returns this point only if it is absent from
    `all_models`. This evaluates one nominal potential rather than searching
    parameter space. Needs no `generator_settings`.
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
