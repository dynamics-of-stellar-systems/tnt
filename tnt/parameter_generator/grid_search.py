"""Grid-search parameter generator scaffold."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from tnt.all_models import AllModels
from tnt.parameter_generator.base import AbstractParameterGenerator, ParameterSet
from tnt.parameter_generator.registry import register_parameter_generator


@register_parameter_generator
class GridSearchParameterGenerator(AbstractParameterGenerator):
    """Proposes parameters on a grid, per each parameter's declared `prior`."""

    _type: ClassVar[str] = "GridSearch"
    _required_generator_settings: ClassVar[frozenset[str]] = frozenset(
        {"delta_chi2_threshold"}
    )

    def _propose_free_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        raise NotImplementedError
