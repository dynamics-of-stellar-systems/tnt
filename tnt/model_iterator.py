"""Orchestrates the generate -> evaluate -> record -> stop? model search loop.

Signature-only scaffold: every method raises `NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from tnt.all_models import AllModels
from tnt.kinematics import AbstractKinematics
from tnt.model import Model
from tnt.orbit_library import AbstractOrbitDithering, AbstractOrbitSampler
from tnt.parameter_generator import AbstractParameterGenerator, ParameterSet
from tnt.weight_solver import AbstractWeightSolver


class ModelIterator:
    """Owns the parameter search loop and its stopping decision.

    `potential_settings`/`mges` hold the fixed, per-run potential-component
    structure (types, MGE references, `include` flags) that
    `AbstractParameterGenerator` fills in with concrete parameter values
    each iteration -- see `tnt.potential.build_potential`.
    """

    potential_settings: Mapping[str, Mapping[str, Any]]
    kinematic_data: Mapping[str, AbstractKinematics]
    weight_solver: AbstractWeightSolver
    parameter_generator: AbstractParameterGenerator
    orbit_library_settings: Mapping[str, Any]
    orbit_sampler: AbstractOrbitSampler
    orbit_dithering: AbstractOrbitDithering
    potential_rescalings: Mapping[str, Any]
    stopping_criteria: Mapping[str, Any]
    execution_settings: Mapping[str, Any]

    @classmethod
    def from_configuration(cls, config: Mapping[str, Any]) -> Self:
        """Build a `ModelIterator` from a fully resolved configuration.

        Runtime construction is ordered by dependency: build named MGEs,
        build named spatial binnings, then call `build_kinematics` with both
        registries and `io_settings.input_directory` before constructing the
        iterator's remaining model-search services.
        """
        raise NotImplementedError

    def run(self, all_models: AllModels | None = None) -> AllModels:
        """Iterate until a stopping criterion is met.

        Args:
            all_models: Models evaluated in a previous run to resume from,
                or `None` to start fresh.

        Returns:
            The final `AllModels`, including every model evaluated.
        """
        raise NotImplementedError

    def _evaluate(self, parameters: ParameterSet) -> list[Model]:
        """Evaluate one proposed `ParameterSet`, and its cheap mass rescalings.

        Builds one `Potential`/`OrbitLibrary` for `parameters` --
        `Potential.generate_orbit_library` takes `self.orbit_sampler`/
        `self.orbit_dithering` alongside `self.orbit_library_settings` --
        then -- if `potential_rescalings.enabled` -- also produces
        `range_count` additional `Model`s at nearby mass scales via
        `Potential.rescale`/`OrbitLibrary.rescaled`, reusing the same
        orbit-library integration rather than re-integrating once per mass
        scale.
        """
        raise NotImplementedError
