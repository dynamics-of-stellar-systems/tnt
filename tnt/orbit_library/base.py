"""Shared orbit-sampler/dithering contracts, `OrbitLibrary`, and their builders.

`build_orbit_sampler`/`build_orbit_dithering` remain unimplemented, and
`AbstractOrbitDithering`'s own concrete scheme isn't consumed by
`generate_ics` yet. Concrete samplers live one per sibling module
(`tnt.orbit_library.stationary_grid`, `.xz_grid_from_origin`,
`.xz_grid_from_boundary`), the same split `tnt.potential` uses for its own
composite types.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self

import equinox as eqx
import galax.potential as gp
import jax.numpy as jnp
from unxt import AbstractUnitSystem, Quantity

if TYPE_CHECKING:
    from tnt.potential import Potential


class AbstractOrbitSampler(eqx.Module):
    """How orbit bundles' initial conditions/orbit families are chosen.

    `_type` matches `orbit_library_settings.orbit_sampler.type`. Concrete
    subclasses hold their own resolved settings as fields (e.g.
    `StationaryGridOrbitSampler.nE`), built once by `build_orbit_sampler`
    rather than re-reading `orbit_library_settings` on every call.

    `_add_reverse_copies` is fixed by which orbit family this sampler
    produces, not a free modeling choice -- `True` for a sampler whose
    orbits have a definite sense of circulation (an `(x, z)`-plane/tube
    population needs its counter-rotating partner to allow net rotation in
    the fit), `False` for one that doesn't (a box orbit has no such sense
    to begin with). Every concrete subclass must declare it explicitly,
    the same way `_type` has no default here.
    """

    _type: ClassVar[str]
    _add_reverse_copies: ClassVar[bool]

    def n_bundles(self) -> int:
        """Total number of orbit bundles this scheme produces."""
        raise NotImplementedError

    def generate_ics(self, potential: gp.AbstractPotential) -> jnp.ndarray:
        """Generate every bundle's initial conditions.

        Args:
            potential: The already-built `galax` potential orbits will be
                integrated in (the module-level `generate_ics` below builds
                this from a `tnt.potential.Potential` and a unit system) --
                initial conditions (e.g. energies, turning points) generally
                depend on its shape.

        Returns:
            A `(self.n_bundles(), 6)` array of phase-space bundle-centre
            initial conditions `(x, y, z, vx, vy, vz)`. Dithering each
            bundle into several closely-spaced orbits
            (`AbstractOrbitDithering`) is a separate, not-yet-implemented
            concern -- see that class's docstring.
        """
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


def generate_ics(
    potential: Potential,
    unit_system: AbstractUnitSystem,
    orbit_sampler: AbstractOrbitSampler,
) -> jnp.ndarray:
    """Generate an orbit sampler's bundle-centre initial conditions in one
    `Potential`.

    A free function, not a `Potential` method -- like
    `raw_potential_parameters` in `tnt.potential.core`, this needs
    config-level context (`unit_system`) beyond the potential's own
    identity. Builds the `galax` potential (`potential.to_galax(unit_system)`)
    once and hands it to `orbit_sampler.generate_ics` -- the one place
    `unit_system` is needed at all; `AbstractOrbitSampler.generate_ics`
    itself only ever takes the already-built `galax` potential.
    """
    return orbit_sampler.generate_ics(potential.to_galax(unit_system))


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
