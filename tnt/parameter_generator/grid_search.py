"""Grid-search parameter generator scaffold."""

from __future__ import annotations

from typing import ClassVar

from tnt.all_models import AllModels
from tnt.parameter_generator.base import AbstractParameterGenerator
from tnt.parameter_generator.registry import register_parameter_generator
from tnt.parameter_generator.types import ParameterBatch


@register_parameter_generator
class GridSearchParameterGenerator(AbstractParameterGenerator):
    """Proposes parameters on a grid, per each parameter's declared `prior`."""

    _type: ClassVar[str] = "GridSearch"
    _required_generator_settings: ClassVar[frozenset[str]] = frozenset(
        {"delta_chi2_threshold"}
    )

    def _generate(
        self, all_models: AllModels, *, record_history: bool = False
    ) -> ParameterBatch:
        raise NotImplementedError
