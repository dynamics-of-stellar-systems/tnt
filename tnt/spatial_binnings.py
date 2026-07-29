"""Spatial binning schemes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from unxt import AbstractUnitSystem, Quantity

from tnt import quantity_conversions


class SphericalGrid(eqx.Module):
    """Cell edges of a spherical ``(r, theta, phi)`` grid, one octant.

    `r_edges` has ``n_r + 1`` edges bounding ``n_r`` bins spanning ``(0,
    infinity)``: ``r_edges[0] == 0``, ``r_edges[-1] == inf``, and the bins
    between are logarithmically spaced from `r_min` to `r_max`.

    `theta` bins (`cos_theta_edges`, ``n_theta + 1`` edges) are spaced linearly
    in ``cos(theta)`` rather than `theta` itself -- from ``cos(0) = 1`` to
    ``cos(pi/2) = 0`` -- so that every (theta, phi) cell spans equal solid
    angle at fixed r. Quenneville, Liepold & Ma (2021) sec. 4.4 found this
    necessary to avoid under-sampling mass near the poles.

    `phi_edges` (``n_phi + 1`` edges) is linearly spaced over ``[0,pi/2]``.
    """

    r_edges: Quantity
    cos_theta_edges: Quantity
    phi_edges: Quantity

    def __init__(
        self, n_r: int, n_theta: int, n_phi: int, r_min: Quantity, r_max: Quantity
    ) -> None:
        """Build the grid's cell edges.

        Args:
            n_r: Number of radial bins; must be at least 2.
            n_theta: Number of polar-angle bins.
            n_phi: Number of azimuthal-angle bins.
            r_min: Inner edge of the logarithmically spaced radial region.
            r_max: Outer edge of the logarithmically spaced radial region.

        Raises:
            ValueError: If `n_r` is less than 3. (With only 2 bins, the single
                interior edge can't be both `r_min` and `r_max`.)
        """
        if n_r < 3:
            raise ValueError(f"n_r must be at least 3, got {n_r}")

        length_unit = r_min.unit
        r_min_v = r_min.ustrip(length_unit)
        r_max_v = r_max.ustrip(length_unit)

        interior_edges = jnp.geomspace(r_min_v, r_max_v, n_r - 1)
        r_edges = jnp.concatenate(
            [jnp.zeros(1), interior_edges, jnp.array([jnp.inf])]
        )

        self.r_edges = Quantity(r_edges, length_unit)
        self.cos_theta_edges = Quantity(jnp.linspace(1.0, 0.0, n_theta + 1), "")
        self.phi_edges = Quantity(jnp.linspace(0.0, jnp.pi / 2, n_phi + 1), "rad")

    @property
    def n_r(self) -> int:
        return self.r_edges.shape[0] - 1

    @property
    def n_theta(self) -> int:
        return self.cos_theta_edges.shape[0] - 1

    @property
    def n_phi(self) -> int:
        return self.phi_edges.shape[0] - 1


# Each declared quantity field, and whether it must be strictly positive.
_QUANTITY_FIELDS = {
    "min_x": False,
    "min_y": False,
    "x_extent": True,
    "y_extent": True,
    "PA": False,
}


class ProjectedBinning(eqx.Module):
    """A projected-plane aperture grid and its pixel-to-bin assignment.

    `min_x`/`min_y` locate the grid's lower corner and `x_extent`/`y_extent`
    give its extent, so pixels span ``(min_x, min_x + x_extent)`` by ``(min_y,
    min_y + y_extent)``. `PA` is the position angle of the galaxy's major
    axis. `bins` is a ``(npix_x, npix_y)`` array of integer bin IDs, one per
    pixel; a value of 0 marks a pixel with no associated bin.
    """

    min_x: Quantity
    min_y: Quantity
    x_extent: Quantity
    y_extent: Quantity
    PA: Quantity
    bins: jnp.ndarray

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        bins: jnp.ndarray,
        unit_system: AbstractUnitSystem,
    ) -> ProjectedBinning:
        """Build a `ProjectedBinning`, validating and converting its raw fields.

        Args:
            settings: A resolved ``spatial_binnings`` entry: `min_x`, `min_y`,
                `x_extent`, `y_extent`, and `PA`, each an explicit
                ``{value, unit}`` mapping.
            bins: The entry's already-loaded 2D ``(npix_x, npix_y)`` array of
                integer bin IDs.
            unit_system: The unit system to convert the quantities into.

        Returns:
            A `ProjectedBinning` with its quantities converted to
            `unit_system`'s angle unit.

        Raises:
            TypeError: If a declared value isn't a number, or `bins` isn't
                integer-valued.
            ValueError: If a required field is missing or malformed, a
                declared value is non-finite or (for `x_extent`/`y_extent`)
                not positive, a declared unit string doesn't parse, `bins`
                isn't 2D, or `bins` contains a negative bin ID.
            astropy.units.UnitConversionError: If a declared unit isn't
                dimensionally an angle.
        """
        angle = unit_system.angle
        quantities = {
            key: _declared_angle_quantity(settings, key, angle, positive=positive)
            for key, positive in _QUANTITY_FIELDS.items()
        }
        return cls(bins=_validated_bins(bins), **quantities)

    @property
    def npix_x(self) -> int:
        return self.bins.shape[0]

    @property
    def npix_y(self) -> int:
        return self.bins.shape[1]

    def angular_to_physical(self, distance: Quantity) -> ProjectedBinning:
        """Convert `min_x`, `min_y`, `x_extent`, and `y_extent` to physical units.

        `PA` (an orientation angle, not a spatial size) and `bins` are
        unaffected and carried over unchanged.

        Args:
            distance: The distance to the object.

        Returns:
            A new `ProjectedBinning` with its spatial fields in `distance`'s
            unit.
        """
        return type(self)(
            min_x=quantity_conversions.angular_to_physical(self.min_x, distance),
            min_y=quantity_conversions.angular_to_physical(self.min_y, distance),
            x_extent=quantity_conversions.angular_to_physical(
                self.x_extent, distance
            ),
            y_extent=quantity_conversions.angular_to_physical(
                self.y_extent, distance
            ),
            PA=self.PA,
            bins=self.bins,
        )


def _declared_angle_quantity(
    settings: Mapping[str, Any],
    key: str,
    angle: Any,
    *,
    positive: bool,
) -> Quantity:
    """Parse, validate, and convert one declared ``{value, unit}`` field."""
    if key not in settings:
        raise ValueError(f"ProjectedBinning is missing required field: {key}.")
    field = settings[key]
    if not isinstance(field, Mapping) or set(field) != {"value", "unit"}:
        raise ValueError(
            f"ProjectedBinning.{key} must be a mapping with exactly 'value' "
            "and 'unit' keys."
        )
    value = field["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"ProjectedBinning.{key}.value must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"ProjectedBinning.{key}.value must be finite.")
    if positive and value <= 0:
        raise ValueError(f"ProjectedBinning.{key} must be greater than zero.")

    return Quantity(float(value), field["unit"]).uconvert(angle)


def _validated_bins(bins: Any) -> jnp.ndarray:
    """Check that `bins` is a 2D array of non-negative integer bin IDs."""
    array = jnp.asarray(bins)
    if array.ndim != 2:
        raise ValueError(
            "ProjectedBinning.bins must be a 2D (npix_x, npix_y) array, got "
            f"shape {array.shape}."
        )
    if not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError("ProjectedBinning.bins must have an integer dtype.")
    if bool(jnp.any(array < 0)):
        raise ValueError("ProjectedBinning.bins must not contain negative bin IDs.")
    return array


def build_spatial_binnings(
    spatial_binnings: Mapping[str, Mapping[str, Any]],
    input_directory: str | Path,
    unit_system: AbstractUnitSystem,
) -> dict[str, ProjectedBinning]:
    """Build the named `ProjectedBinning`s from a resolved configuration.

    Each binning's aperture geometry (`min_x`, `min_y`, `x_extent`,
    `y_extent`, `PA`) comes from the configuration itself rather than a data
    file, and is validated and converted by `ProjectedBinning.from_settings`
    -- only its ``bins_file`` is read from disk here.

    Args:
        spatial_binnings: Mapping of unique identifiers to each binning's
            resolved settings, e.g. a resolved configuration's
            ``spatial_binnings`` section.
        input_directory: Directory that each ``bins_file`` is resolved
            against, e.g. a resolved configuration's
            ``io_settings.input_directory``.
        unit_system: The unit system to convert each binning's quantities
            into.

    Returns:
        A dict mapping each identifier to its `ProjectedBinning`.
    """
    directory = Path(input_directory)
    return {
        name: ProjectedBinning.from_settings(
            settings,
            jnp.asarray(np.load(directory / settings["bins_file"])),
            unit_system,
        )
        for name, settings in spatial_binnings.items()
    }
