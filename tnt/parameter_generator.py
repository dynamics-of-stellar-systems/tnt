"""Proposing the next round of potential parameters to evaluate.

Signature-only scaffold: every method raises `NotImplementedError`.
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
    """

    _type: ClassVar[str]

    def generate_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        """Propose the next round of `ParameterSet`s.

        Args:
            all_models: Every model evaluated so far (empty on the first
                iteration).
        """
        raise NotImplementedError


class GridSearchParameterGenerator(AbstractParameterGenerator):
    """Proposes parameters on a grid, per `generator_settings`."""

    _type: ClassVar[str] = "GridSearch"
    generator_settings: Mapping[str, Any]

    def generate_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        raise NotImplementedError
