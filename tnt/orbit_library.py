"""Orbit libraries integrated in a `tnt.potential.Potential`.

Signature-only scaffold: every method raises `NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self

import equinox as eqx
import jax.numpy as jnp
from unxt import Quantity

if TYPE_CHECKING:
    from tnt.potential import Potential


class AbstractOrbitSampler(eqx.Module):
    """How orbit bundles' initial conditions/orbit families are chosen.

    `_type` matches `orbit_library_settings.orbit_sampler.type`. Concrete
    subclasses hold their own resolved settings as fields (e.g.
    `GridOrbitSampler.nE`), built once by `build_orbit_sampler` rather than
    re-reading `orbit_library_settings` on every call.
    """

    _type: ClassVar[str]

    def n_bundles(self) -> int:
        """Total number of orbit bundles this scheme produces."""
        raise NotImplementedError

    def generate_ics(
        self,
        potential: Potential,
        dithering: AbstractOrbitDithering,
    ) -> jnp.ndarray:
        """Generate every bundle's dithered initial conditions.

        Args:
            potential: The potential orbits will be integrated in --
                initial conditions (e.g. energies, turning points)
                generally depend on its shape.
            dithering: Determines how many closely-spaced orbits
                (`dithering.n_orbits_per_bundle()`) to generate per bundle.

        Returns:
            A `(self.n_bundles(), dithering.n_orbits_per_bundle(), 6)` array
            of phase-space initial conditions `(x, y, z, vx, vy, vz)`.
        """
        raise NotImplementedError


class GridOrbitSampler(AbstractOrbitSampler):
    """Box + tube + counter-rotating-tube grid over (E, I2, I3).

    `n_bundles` is `3 * nE * nI2 * nI3`: a box-orbit library and a
    tube-orbit library, each of size `nE * nI2 * nI3`, plus a
    counter-rotating copy of the tube library.
    """

    _type: ClassVar[str] = "Grid"
    logrmin: float
    logrmax: float
    nE: int
    nI2: int
    nI3: int

    def n_bundles(self) -> int:
        raise NotImplementedError

    def generate_ics(
        self,
        potential: Potential,
        dithering: AbstractOrbitDithering,
    ) -> jnp.ndarray:
        raise NotImplementedError


class RandomOrbitSampler(AbstractOrbitSampler):
    """Orbit bundles from randomly sampled initial conditions.

    Fields beyond `logrmin`/`logrmax` are still undecided -- only `type`,
    `logrmin`, and `logrmax` are validated so far
    (`configuration_validation._validate_orbit_sampler`).
    """

    _type: ClassVar[str] = "Random"
    logrmin: float
    logrmax: float

    def n_bundles(self) -> int:
        raise NotImplementedError

    def generate_ics(
        self,
        potential: Potential,
        dithering: AbstractOrbitDithering,
    ) -> jnp.ndarray:
        raise NotImplementedError


class AbstractOrbitDithering(eqx.Module):
    """How each orbit bundle's closely-spaced individual orbits are generated.

    `_type` matches `orbit_library_settings.dithering.type`. Concrete
    subclasses hold their own resolved settings as fields (e.g.
    `CubicOrbitDithering.n_dither`), built once by `build_orbit_dithering`.
    """

    _type: ClassVar[str]

    def n_orbits_per_bundle(self) -> int:
        """Number of dithered orbits generated per bundle."""
        raise NotImplementedError


class CubicOrbitDithering(AbstractOrbitDithering):
    """`n_dither^3` orbits per bundle -- the current implementation."""

    _type: ClassVar[str] = "Cubic"
    n_dither: int

    def n_orbits_per_bundle(self) -> int:
        raise NotImplementedError


def build_orbit_sampler(
    orbit_library_settings: Mapping[str, Any],
) -> AbstractOrbitSampler:
    """Build the `AbstractOrbitSampler` named by
    `orbit_library_settings.orbit_sampler`.
    """
    raise NotImplementedError


def build_orbit_dithering(
    orbit_library_settings: Mapping[str, Any],
) -> AbstractOrbitDithering:
    """Build the `AbstractOrbitDithering` named by
    `orbit_library_settings.dithering`.
    """
    raise NotImplementedError


class OrbitLibrary(eqx.Module):
    """Orbits integrated in one `Potential`, plus their orbital periods.

    `orbits` has shape `(n_bundles, n_orbits_per_bundle, n_stored_timesteps,
    6)`: `n_bundles` (from `AbstractOrbitSampler.n_bundles`) orbit bundles,
    each with `n_orbits_per_bundle` (from
    `AbstractOrbitDithering.n_orbits_per_bundle`) closely-dithered orbits,
    each stored at `orbit_library_settings.n_stored_timesteps` points along
    its trajectory, each point a phase-space vector `(x, y, z, vx, vy, vz)`.

    Deliberately has no `project`/binning-aware method -- projecting orbits
    onto a `ProjectedBinning` is `AbstractKinematics.design_matrix`'s
    responsibility (`tnt.kinematics`), so it isn't duplicated here.
    """

    orbits: jnp.ndarray
    orbital_periods: Quantity

    def rescaled(self, mass_scale: float) -> Self:
        """Rescale this library to a new mass without re-integrating.

        Valid only alongside `Potential.rescale(mass_scale)` on the same
        potential: the shape of every orbit is unchanged, but its energy,
        velocity, and time scaling depend on the potential's overall mass.
        """
        raise NotImplementedError
