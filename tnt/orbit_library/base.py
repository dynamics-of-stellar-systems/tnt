"""Shared orbit-sampler/dithering contracts and their builders.

Builders consume the validated configuration schema. Only one member per
bundle is supported; larger cubic dithering is explicitly rejected.
Concrete samplers live one per sibling module
(`tnt.orbit_library.stationary_grid`, `.xz_grid_from_origin`,
`.xz_grid_from_boundary`), the same split `tnt.potential` uses for its own
composite types.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import equinox as eqx
import galax.potential as gp
import jax.numpy as jnp
from unxt import AbstractUnitSystem, Quantity

from tnt.orbit_library.data import OrbitInitialConditions
from tnt.orbit_library.integration import require_static_potential
from tnt.units import reference_unit
from tnt.validation import _integer

if TYPE_CHECKING:
    from tnt.potential import Potential


class AbstractOrbitSampler(eqx.Module):
    """How orbit bundles' initial conditions/orbit families are chosen.

    `_type` matches `orbit_library_settings.orbit_sampler.type`. Concrete
    subclasses hold their own resolved settings as fields (e.g.
    `StationaryGridOrbitSampler.nE`). `build_orbit_sampler` constructs a
    sampler from resolved configuration. Calls use stored fields rather than
    re-reading `orbit_library_settings`.

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

    def __check_init__(self) -> None:
        """Enforce every sampler's shared `0 < rmin < rmax` radial domain.

        Runs on construction regardless of how a sampler is built --
        `_validate_orbit_sampler` enforces the same domain at the
        configuration boundary, but a sampler built directly (e.g. in a
        test, or once `build_orbit_sampler` exists and is called with
        already-resolved values) does not go through that check. `rmin`
        sets the logarithmic energy-grid lower bound, and also the radial
        floor for the two `(x, z)` samplers. This check rejects zero or
        negative values before logarithms can produce non-finite radii.
        """
        length_unit = reference_unit("length")
        rmin = self.rmin.ustrip(length_unit)
        rmax = self.rmax.ustrip(length_unit)
        if not (0 < rmin < rmax):
            raise ValueError(
                f"rmin ({self.rmin}) must be positive and less than rmax ({self.rmax})."
            )

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

    def generate_initial_conditions(
        self, potential: gp.AbstractPotential, *, reference_periods: Quantity
    ) -> OrbitInitialConditions:
        """Generate launches with explicit, shared energy-shell clocks.

        Stationary/origin samplers require supplied clocks, normally the
        boundary sampler's DYNAMITE periods on the same energy grid. No
        independent launch-radius period or silent fallback is introduced.
        """
        require_static_potential(potential)
        return self._with_reference_periods(
            potential, self.generate_ics(potential), reference_periods
        )

    def _with_reference_periods(
        self, potential: gp.AbstractPotential, ics: jnp.ndarray,
        reference_periods: Quantity,
    ) -> OrbitInitialConditions:
        periods = reference_periods.ustrip(potential.units["time"])
        if periods.shape != (self.nE,):
            raise ValueError(
                "Supply one reference period per energy shell, shape (nE,)."
            )
        if not bool(jnp.all(jnp.isfinite(periods) & (periods > 0))):
            raise ValueError("Reference periods must be finite and positive.")
        periods = jnp.repeat(periods, self.nI2 * self.nI3)[:, None]
        return OrbitInitialConditions(
            positions=Quantity(ics[:, None, :3], potential.units["length"]),
            velocities=Quantity(
                ics[:, None, 3:], potential.units["length"] / potential.units["time"]
            ),
            reference_periods=Quantity(periods, potential.units["time"]),
            bundle_ids=jnp.arange(self.n_bundles()),
        )


class AbstractOrbitDithering(eqx.Module):
    """How each orbit bundle's closely-spaced individual orbits are generated.

    `_type` matches `orbit_library_settings.dithering.type`. Concrete
    subclasses hold their own resolved settings as fields (e.g.
    `CubicOrbitDithering.n_dither`). The configuration builder currently
    permits exactly one member per bundle.
    """

    _type: ClassVar[str]

    def n_orbits_per_bundle(self) -> int:
        """Number of dithered orbits generated per bundle."""
        raise NotImplementedError


class CubicOrbitDithering(AbstractOrbitDithering):
    """One member per bundle; larger cubic dithering is deferred."""

    _type: ClassVar[str] = "Cubic"
    n_dither: int

    def __check_init__(self) -> None:
        if _integer(self.n_dither, "n_dither") < 1:
            raise ValueError("n_dither must be positive.")
        if self.n_dither != 1:
            raise NotImplementedError("Only n_dither=1 is implemented.")

    def n_orbits_per_bundle(self) -> int:
        return 1


def build_orbit_sampler(
    orbit_library_settings: Mapping[str, Any],
) -> AbstractOrbitSampler:
    """Build the `AbstractOrbitSampler` named by
    `orbit_library_settings.orbit_sampler`.
    """
    # Concrete modules import this base: resolve their constructors lazily.
    from tnt.configuration.validation import _validate_orbit_sampler
    from tnt.orbit_library.stationary_grid import StationaryGridOrbitSampler
    from tnt.orbit_library.xz_grid_from_boundary import XZGridFromBoundaryOrbitSampler
    from tnt.orbit_library.xz_grid_from_origin import XZGridFromOriginOrbitSampler
    from tnt.units import declared_quantity

    settings = orbit_library_settings["orbit_sampler"]
    _validate_orbit_sampler(settings, "orbit_library_settings.orbit_sampler")
    sampler_types = {
        cls._type: cls for cls in (
            StationaryGridOrbitSampler, XZGridFromOriginOrbitSampler,
            XZGridFromBoundaryOrbitSampler,
        )
    }
    return sampler_types[settings["type"]](
        rmin=declared_quantity(settings["rmin"], "length", "orbit_sampler.rmin"),
        rmax=declared_quantity(settings["rmax"], "length", "orbit_sampler.rmax"),
        nE=settings["nE"], nI2=settings["nI2"], nI3=settings["nI3"],
    )


def build_orbit_dithering(
    orbit_library_settings: Mapping[str, Any],
) -> AbstractOrbitDithering:
    """Build the `AbstractOrbitDithering` named by
    `orbit_library_settings.dithering`.
    """
    from tnt.configuration.validation import _validate_dithering

    settings = orbit_library_settings["dithering"]
    _validate_dithering(settings, "orbit_library_settings.dithering")
    return CubicOrbitDithering(n_dither=settings["n_dither"])


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
