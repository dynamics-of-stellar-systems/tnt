"""`Potential`: the sum of included components, and its module-level helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self

import equinox as eqx
import galax.potential
from unxt import AbstractUnitSystem, Quantity

from tnt.mge import LightMGE, MassMGE
from tnt.orbit_library import AbstractOrbitDithering, AbstractOrbitSampler, OrbitLibrary
from tnt.potential.components import (
    AbstractPotentialComponent,
    ResolvedPotentialComponent,
)
from tnt.validation import _mapping

if TYPE_CHECKING:
    from tnt.parameter_generator import ParameterSet


class Potential(eqx.Module):
    """The sum of included potential components, at one point in parameter space."""

    components: dict[str, AbstractPotentialComponent]

    @classmethod
    def resolve(
        cls,
        settings: Mapping[str, Mapping[str, Any]],
        mges: Mapping[str, LightMGE | MassMGE],
    ) -> dict[str, ResolvedPotentialComponent]:
        """Resolve every included component's static structure, once per run.

        See `AbstractPotentialComponent.resolve` -- a caller building many
        `Potential`s from the same configuration (e.g. `ModelIterator`,
        once per proposed `ParameterSet`) should call this once and reuse
        the result via `Potential.build`.
        """
        resolved: dict[str, ResolvedPotentialComponent] = {}
        for name, component_value in settings.items():
            path = f"potential.{name}"
            component_settings = _mapping(component_value, path)
            if not component_settings.get("include", True):
                continue
            resolved[name] = AbstractPotentialComponent.resolve(
                component_settings, mges, path=path
            )
        if not resolved:
            raise ValueError("potential must contain at least one included component.")
        return resolved

    @classmethod
    def build(
        cls,
        resolved: Mapping[str, ResolvedPotentialComponent],
        parameter_values: ParameterSet,
        unit_system: AbstractUnitSystem,
        cosmological_parameters: Mapping[str, Quantity],
    ) -> Self:
        """Build a `Potential` from resolved static structure and a proposed point.

        Args:
            resolved: Every included component's static structure, e.g.
                from `Potential.resolve`.
            parameter_values: The current point in parameter space, e.g. a
                `tnt.parameter_generator.ParameterSet`.
            unit_system: Passed through to each component's construction --
                see `ResolvedPotentialComponent.build`.
            cosmological_parameters: A resolved configuration's
                `cosmological_parameters` section -- used only by
                parameterizations that need it, e.g. NFW's `concentration_m200`.
        """
        return cls(
            components={
                name: component.build(
                    parameter_values.get(name, {}), unit_system, cosmological_parameters
                )
                for name, component in resolved.items()
            }
        )

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Mapping[str, Any]],
        parameter_values: ParameterSet,
        mges: Mapping[str, LightMGE | MassMGE],
        unit_system: AbstractUnitSystem,
        cosmological_parameters: Mapping[str, Quantity],
    ) -> Self:
        """Build a `Potential` from a resolved configuration's `potential` section.

        A one-shot convenience combining `Potential.resolve`/`Potential.build`
        -- callers that build many `Potential`s from the same configuration
        (e.g. `ModelIterator`) should call those directly instead, resolving
        once and reusing the result.
        """
        return cls.build(
            cls.resolve(settings, mges),
            parameter_values,
            unit_system,
            cosmological_parameters,
        )

    def to_galax(
        self, unit_system: AbstractUnitSystem
    ) -> galax.potential.AbstractPotential:
        """This potential's included components, composed into one `galax` potential."""
        return galax.potential.CompositePotential(
            {
                name: component.to_galax(unit_system)
                for name, component in self.components.items()
            },
            units=unit_system,
        )

    def generate_orbit_library(
        self,
        orbit_library_settings: Mapping[str, Any],
        orbit_sampler: AbstractOrbitSampler,
        orbit_dithering: AbstractOrbitDithering,
    ) -> OrbitLibrary:
        """Integrate this potential's `OrbitLibrary`.

        Args:
            orbit_library_settings: A resolved configuration's
                `orbit_library_settings` section.
            orbit_sampler: Determines the number of orbit bundles
                (`OrbitLibrary.orbits`' leading axis), e.g. from
                `tnt.orbit_library.build_orbit_sampler`.
            orbit_dithering: Determines the number of dithered orbits per
                bundle (`OrbitLibrary.orbits`' second axis), e.g. from
                `tnt.orbit_library.build_orbit_dithering`.
        """
        raise NotImplementedError

    def rescale(self, mass_scale: float) -> Self:
        """Multiply every component's mass parameter by `mass_scale`.

        See `AbstractPotentialComponent.rescale` for why every component,
        `fixed` or not, must be rescaled together.
        """
        return type(self)(
            components={
                name: component.rescale(mass_scale)
                for name, component in self.components.items()
            }
        )


def build_potential(
    resolved: Mapping[str, ResolvedPotentialComponent],
    parameter_values: ParameterSet,
    unit_system: AbstractUnitSystem,
    cosmological_parameters: Mapping[str, Quantity],
) -> Potential:
    """Build the `Potential` from pre-resolved static structure and a proposed point.

    Args:
        resolved: Every included component's static structure, e.g. from
            `Potential.resolve(config["potential"], mges)`, called once per
            run.
        parameter_values: The current point in parameter space, e.g. a
            `tnt.parameter_generator.ParameterSet`.
        unit_system: Passed through to each component's construction.
        cosmological_parameters: A resolved configuration's
            `cosmological_parameters` section -- used only by
            parameterizations that need it, e.g. NFW's `concentration_m200`.

    Returns:
        A `Potential` assembled from every included component.
    """
    return Potential.build(
        resolved, parameter_values, unit_system, cosmological_parameters
    )


def raw_potential_parameters(
    potential_settings: Mapping[str, Mapping[str, Any]],
    potential: Potential,
    unit_system: AbstractUnitSystem,
    cosmological_parameters: Mapping[str, Quantity],
) -> dict[str, dict[str, Quantity]]:
    """Every included component's parameters, in the config's own parameterization.

    The inverse of `build_potential`/`Potential.from_settings`: where those
    convert each raw config parameter into `galax`'s native constructor
    kwargs, this converts back, e.g. NFW's `concentration_m200`'s native
    `(m, r_s)` back to `(c, M_200)`. `AllModels` uses this to report every
    model in the parameterization its configuration actually specifies,
    regardless of `Potential.rescale`, which only knows how to scale
    native parameters (see `GalaxPotentialComponent.raw_parameters`).

    `potential_settings` is the source of "which parameterization was
    configured" for each component -- `potential` itself doesn't carry that,
    since `AbstractPotentialComponent.parameters` is deliberately
    parameterization-independent.

    Args:
        potential_settings: A resolved configuration's `potential` section
            (e.g. `ModelIterator.potential_settings`) -- only each
            component's `parameterization` is used.
        potential: The resolved `Potential` to report, e.g. from
            `build_potential`, possibly after `Potential.rescale`.
        unit_system: Passed through to a registered `parameterization`'s
            inverse converter, e.g. NFW's `concentration_m200` needs it to
            compute a critical density.
        cosmological_parameters: A resolved configuration's
            `cosmological_parameters` section -- used only by
            parameterizations that need it, e.g. NFW's `concentration_m200`.

    Returns:
        A mapping from each included component's name to its raw
        parameters, keyed exactly as its configuration's `parameters` are.
    """
    return {
        name: component.raw_parameters(
            _mapping(potential_settings.get(name, {}), f"potential.{name}").get(
                "parameterization"
            ),
            unit_system,
            cosmological_parameters,
        )
        for name, component in potential.components.items()
    }
