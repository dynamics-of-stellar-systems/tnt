"""Galactic potentials, assembled from named, `galax`-backed components.

This module is a signature-only scaffold: every method raises
`NotImplementedError`. It exists to commit the shape of the potential/model
architecture to the repository so it can be filled in incrementally, one
object at a time -- the same approach already used for `ProjectedBinning`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import equinox as eqx
import galax.potential
from unxt import Quantity

from tnt.mge import LightMGE, MassMGE
from tnt.orbit_library import AbstractOrbitDithering, AbstractOrbitSampler, OrbitLibrary


class AbstractPotentialComponent(eqx.Module):
    """One named term of the total potential (e.g. a halo, a light MGE).

    `_type` matches a `potential.<name>` config entry's `type` field.
    Every component has exactly one mass-normalization parameter on top of
    an otherwise-fixed shape (`ml` for a light MGE, `mge_mass_scale` for a
    mass MGE, a characteristic mass for `nfw`/`plummer`); `rescale`
    multiplies that parameter, holding shape fixed.
    """

    _type: ClassVar[str]
    parameters: dict[str, Quantity]

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        mges: Mapping[str, LightMGE | MassMGE],
    ) -> Self:
        """Build one potential component from its resolved config entry."""
        raise NotImplementedError

    def to_galax(self) -> galax.potential.AbstractPotentialBase:
        """This component as a `galax` potential."""
        raise NotImplementedError

    def rescale(self, mass_scale: float) -> Self:
        """Multiply this component's mass-normalization parameter.

        Used for cheap re-exploration of nearby mass scales without
        re-integrating orbits (`parameter_space_settings.potential_rescalings`
        via `ModelIterator`). This applies even when that parameter is
        `fixed`: `fixed` only stops `ParameterGenerator` from proposing
        independent values for it across shape points -- it doesn't exempt
        it from this uniform rescale, which every component must undergo
        together, or the potential's shape (and the orbit library
        integrated in it) would silently no longer match.
        """
        raise NotImplementedError


class NFWComponent(AbstractPotentialComponent):
    """A Navarro-Frenk-White halo component."""

    _type: ClassVar[str] = "nfw"

    def rescale(self, mass_scale: float) -> Self:
        raise NotImplementedError


class PlummerComponent(AbstractPotentialComponent):
    """A Plummer sphere component."""

    _type: ClassVar[str] = "plummer"

    def rescale(self, mass_scale: float) -> Self:
        raise NotImplementedError


class TriaxialLightMGEComponent(AbstractPotentialComponent):
    """A triaxial potential from a light MGE, via its `ml` parameter.

    `to_galax` converts `mge` to a mass MGE via `mge.to_mass(ml)` before
    building the `galax` potential.
    """

    _type: ClassVar[str] = "triaxial_light_mge"
    mge: LightMGE

    def rescale(self, mass_scale: float) -> Self:
        rescaled = dict(self.parameters)
        rescaled["ml"] = rescaled["ml"] * mass_scale
        return eqx.tree_at(lambda c: c.parameters, self, rescaled)


class TriaxialMassMGEComponent(AbstractPotentialComponent):
    """A triaxial potential from an already-mass-calibrated MGE.

    `mge_mass_scale` is the analogue of `ml` for a component whose shape
    template is already in mass units: a normalization on top of an
    otherwise-fixed mass map, typically left `fixed` (see `rescale`'s
    docstring for why it can still move regardless).

    `to_galax` scales `mge` by `mge_mass_scale` before building the `galax`
    potential -- this requires an `AbstractMGE.rescale`-style method that
    doesn't exist yet in `tnt.mge`; adding it is a real follow-up, not part
    of this signature scaffold.
    """

    _type: ClassVar[str] = "triaxial_mass_mge"
    mge: MassMGE

    def rescale(self, mass_scale: float) -> Self:
        rescaled = dict(self.parameters)
        rescaled["mge_mass_scale"] = rescaled["mge_mass_scale"] * mass_scale
        return eqx.tree_at(lambda c: c.parameters, self, rescaled)


class Potential(eqx.Module):
    """The sum of included potential components, at one point in parameter space."""

    components: dict[str, AbstractPotentialComponent]

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        mges: Mapping[str, LightMGE | MassMGE],
    ) -> Self:
        """Build a `Potential` from a resolved configuration's `potential` section."""
        raise NotImplementedError

    def to_galax(self) -> galax.potential.AbstractPotentialBase:
        """This potential's included components, composed into one `galax` potential."""
        raise NotImplementedError

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
    potential: Mapping[str, Mapping[str, Any]],
    mges: Mapping[str, LightMGE | MassMGE],
) -> Potential:
    """Build the `Potential` from a resolved configuration's `potential` section.

    Args:
        potential: A resolved configuration's `potential` section.
        mges: Named MGEs, e.g. from `tnt.mge.build_mges`.

    Returns:
        A `Potential` assembled from every included component.
    """
    raise NotImplementedError
