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


def _gauss_legendre(quad_order: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fixed-order Gauss-Legendre quadrature nodes/weights on ``[-1, 1]``."""
    nodes, weights = np.polynomial.legendre.leggauss(quad_order)
    return jnp.asarray(nodes), jnp.asarray(weights)


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

    `cos_theta_nodes`/`theta_weights` and `phi_nodes`/`phi_weights` are
    fixed-order Gauss-Legendre quadrature nodes and weights within each
    theta/phi cell (shape ``(n_theta, quad_order)``/``(n_phi, quad_order)``),
    precomputed here so that integrating over this grid repeatedly -- e.g.
    `Deprojected3DMGE.spherical_mass_grid` for many different MGEs -- doesn't
    redo this construction on every call.
    """

    r_edges: Quantity
    cos_theta_edges: Quantity
    phi_edges: Quantity
    cos_theta_nodes: jnp.ndarray
    theta_weights: jnp.ndarray
    phi_nodes: jnp.ndarray
    phi_weights: jnp.ndarray

    def __init__(
        self,
        n_r: int,
        n_theta: int,
        n_phi: int,
        r_min: Quantity,
        r_max: Quantity,
        quad_order: int,
    ) -> None:
        """Build the grid's cell edges and quadrature nodes.

        Args:
            n_r: Number of radial bins; must be at least 2.
            n_theta: Number of polar-angle bins.
            n_phi: Number of azimuthal-angle bins.
            r_min: Inner edge of the logarithmically spaced radial region.
            r_max: Outer edge of the logarithmically spaced radial region.
            quad_order: Order of the fixed Gauss-Legendre quadrature used for
                the theta/phi integral within each cell (see
                `Deprojected3DMGE.spherical_mass_grid`).

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

        nodes, weights = _gauss_legendre(quad_order)

        # The sin(theta) dtheta Jacobian is exactly -d(cos(theta)), so
        # quadrature nodes/weights in cos(theta) already include it -- no
        # extra sin(theta) factor is needed when these are later contracted
        # against an integrand.
        cos_theta_edges = self.cos_theta_edges.ustrip("")
        cos_theta_lo, cos_theta_hi = cos_theta_edges[1:], cos_theta_edges[:-1]
        cos_theta_mid = (cos_theta_hi + cos_theta_lo) / 2
        cos_theta_half = (cos_theta_hi - cos_theta_lo) / 2
        self.cos_theta_nodes = (
            cos_theta_mid[:, None] + cos_theta_half[:, None] * nodes[None, :]
        )
        self.theta_weights = cos_theta_half[:, None] * weights[None, :]

        phi_edges = self.phi_edges.ustrip("rad")
        phi_lo, phi_hi = phi_edges[:-1], phi_edges[1:]
        phi_mid, phi_half = (phi_hi + phi_lo) / 2, (phi_hi - phi_lo) / 2
        self.phi_nodes = phi_mid[:, None] + phi_half[:, None] * nodes[None, :]
        self.phi_weights = phi_half[:, None] * weights[None, :]

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
_SETTINGS_FIELDS = set(_QUANTITY_FIELDS) | {"bins_file"}


class ProjectedBinning(eqx.Module):
    """A projected-plane aperture grid and its pixel-to-bin assignment.

    `min_x`/`min_y` locate the grid's lower corner and `x_extent`/`y_extent`
    give its extent, so pixels span ``(min_x, min_x + x_extent)`` by ``(min_y,
    min_y + y_extent)``. `PA` is the position angle of the galaxy's major
    axis. `bins` is a ``(npix_x, npix_y)`` array of integer bin IDs, one per
    pixel; a value of 0 marks a pixel with no associated bin.

    `x_lo`/`x_hi` and `y_lo`/`y_hi` are each pixel's edges along x and y, and
    `x_nodes`/`x_weights` are fixed-order Gauss-Legendre quadrature nodes and
    weights within each x pixel (shape ``(npix_x, quad_order)``) -- all
    precomputed here, in `min_x`'s unit, so that integrating over this grid
    repeatedly -- e.g. `AbstractMGE.get_projected_mass` for many different
    MGEs -- doesn't redo this construction on every call.

    `n_bins` (`bins`'s largest ID) is precomputed here too, as a static
    (non-array) Python `int`, so that `AbstractMGE.get_projected_mass` never
    needs to concretize `bins` itself to size its `segment_sum` -- keeping it
    usable under `jax.jit` even when other `get_projected_mass` inputs (the
    MGE's parameters) are traced.
    """

    min_x: Quantity
    min_y: Quantity
    x_extent: Quantity
    y_extent: Quantity
    PA: Quantity
    bins: jnp.ndarray
    n_bins: int = eqx.field(static=True)
    x_lo: jnp.ndarray
    x_hi: jnp.ndarray
    y_lo: jnp.ndarray
    y_hi: jnp.ndarray
    x_nodes: jnp.ndarray
    x_weights: jnp.ndarray

    def __init__(
        self,
        *,
        min_x: Quantity,
        min_y: Quantity,
        x_extent: Quantity,
        y_extent: Quantity,
        PA: Quantity,  # noqa: N803
        bins: jnp.ndarray,
        quad_order: int,
    ) -> None:
        """Build the aperture grid's pixel edges and quadrature nodes.

        Args:
            min_x: Lower x edge of the aperture grid.
            min_y: Lower y edge of the aperture grid.
            x_extent: Extent of the aperture grid along x.
            y_extent: Extent of the aperture grid along y.
            PA: Position angle of the galaxy's major axis.
            bins: A 2D ``(npix_x, npix_y)`` array of integer bin IDs.
            quad_order: Order of the fixed Gauss-Legendre quadrature used for
                the x integral within each pixel (see
                `AbstractMGE.get_projected_mass`).
        """
        self.min_x = min_x
        self.min_y = min_y
        self.x_extent = x_extent
        self.y_extent = y_extent
        self.PA = PA
        self.bins = bins
        self.n_bins = int(jnp.max(bins))

        coord_unit = min_x.unit
        npix_x, npix_y = bins.shape[0], bins.shape[1]
        x_edges = min_x.ustrip(coord_unit) + jnp.linspace(
            0.0, x_extent.ustrip(coord_unit), npix_x + 1
        )
        y_edges = min_y.ustrip(coord_unit) + jnp.linspace(
            0.0, y_extent.ustrip(coord_unit), npix_y + 1
        )
        self.x_lo, self.x_hi = x_edges[:-1], x_edges[1:]
        self.y_lo, self.y_hi = y_edges[:-1], y_edges[1:]

        nodes, weights = _gauss_legendre(quad_order)
        x_mid, x_half = (self.x_hi + self.x_lo) / 2, (self.x_hi - self.x_lo) / 2
        self.x_nodes = x_mid[:, None] + x_half[:, None] * nodes[None, :]
        self.x_weights = x_half[:, None] * weights[None, :]

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        bins: jnp.ndarray,
        unit_system: AbstractUnitSystem,
        quad_order: int,
    ) -> ProjectedBinning:
        """Build a `ProjectedBinning`, validating and converting its raw fields.

        Args:
            settings: A resolved ``spatial_binnings`` entry: `min_x`, `min_y`,
                `x_extent`, `y_extent`, and `PA`, each an explicit
                ``{value, unit}`` mapping.
            bins: The entry's already-loaded 2D ``(npix_x, npix_y)`` array of
                integer bin IDs.
            unit_system: The unit system to convert the quantities into.
            quad_order: Order of the fixed Gauss-Legendre quadrature used for
                the x integral within each pixel (see
                `AbstractMGE.get_projected_mass`).

        Returns:
            A `ProjectedBinning` with its quantities converted to
            `unit_system`'s angle unit.

        Raises:
            TypeError: If the settings aren't a mapping, a declared value
                isn't a number, `bins_file` is present but not a non-empty
                string, or `bins` isn't integer-valued.
            ValueError: If a required field is missing, an unknown field is
                present, a declared value is malformed, non-finite, or (for
                `x_extent`/`y_extent`) not positive, a declared unit string
                doesn't parse, or `bins` is not a non-empty 2D array of
                non-negative IDs.
            astropy.units.UnitConversionError: If a declared unit isn't
                dimensionally an angle.
        """
        settings = _validated_settings(settings, require_bins_file=False)
        quantities = _declared_quantities(settings, unit_system.angle)
        return cls(
            bins=_validated_bins(bins), quad_order=quad_order, **quantities
        )

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
            quad_order=self.x_nodes.shape[1],
        )


def _validated_settings(
    settings: Any,
    *,
    require_bins_file: bool,
    path: str = "ProjectedBinning",
) -> Mapping[str, Any]:
    """Validate one spatial-binning entry's top-level schema."""
    if not isinstance(settings, Mapping):
        raise TypeError(f"{path} must be a mapping.")

    required_fields = set(_QUANTITY_FIELDS)
    if require_bins_file:
        required_fields.add("bins_file")
    missing = required_fields - set(settings)
    if missing:
        if len(missing) == 1:
            field = next(iter(missing))
            raise ValueError(f"{path} is missing required field: {field}.")
        missing_fields = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required field(s): {missing_fields}.")

    unknown = set(settings) - _SETTINGS_FIELDS
    if unknown:
        unknown_fields = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"{path} contains unknown field(s): {unknown_fields}.")

    if "bins_file" in settings:
        bins_file = settings["bins_file"]
        if not isinstance(bins_file, str) or not bins_file.strip():
            raise TypeError(f"{path}.bins_file must be a non-empty string.")
    return settings


def _declared_quantities(
    settings: Mapping[str, Any], angle: Any, *, path: str = "ProjectedBinning"
) -> dict[str, Quantity]:
    """Validate and convert all declared spatial-binning quantities."""
    return {
        key: _declared_angle_quantity(
            settings, key, angle, positive=positive, path=path
        )
        for key, positive in _QUANTITY_FIELDS.items()
    }


def _declared_angle_quantity(
    settings: Mapping[str, Any],
    key: str,
    angle: Any,
    *,
    positive: bool,
    path: str,
) -> Quantity:
    """Parse, validate, and convert one declared ``{value, unit}`` field."""
    field = settings[key]
    if not isinstance(field, Mapping) or set(field) != {"value", "unit"}:
        raise ValueError(
            f"{path}.{key} must be a mapping with exactly 'value' "
            "and 'unit' keys."
        )
    value = field["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path}.{key}.value must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"{path}.{key}.value must be finite.")
    if positive and value <= 0:
        raise ValueError(f"{path}.{key} must be greater than zero.")

    return Quantity(float(value), field["unit"]).uconvert(angle)


def _validated_bins(
    bins: Any, *, path: str = "ProjectedBinning.bins"
) -> jnp.ndarray:
    """Check `bins` is a 2D array of non-negative, contiguous integer bin IDs.

    "Contiguous" means its positive IDs run ``1, 2, ..., n_bins`` with no gaps
    -- `AbstractMGE.get_projected_mass` returns one value per ID in that
    range, so a missing ID would silently waste a `segment_sum` slot (sized
    up to the largest ID) rather than signal the corrupt bin map.
    """
    array = jnp.asarray(bins)
    if array.ndim != 2:
        raise ValueError(
            f"{path} must be a 2D (npix_x, npix_y) array, got shape "
            f"{array.shape}."
        )
    if 0 in array.shape:
        raise ValueError(f"{path} dimensions must not be empty.")
    if not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError(f"{path} must have an integer dtype.")
    if bool(jnp.any(array < 0)):
        raise ValueError(f"{path} must not contain negative bin IDs.")
    positive_ids = jnp.unique(array[array > 0])
    if not jnp.array_equal(positive_ids, jnp.arange(1, positive_ids.size + 1)):
        raise ValueError(
            f"{path}'s positive bin IDs must be contiguous, running from 1 to "
            f"their maximum with no gaps; got {sorted(int(i) for i in positive_ids)}."
        )
    return array


def build_spatial_binnings(
    spatial_binnings: Mapping[str, Mapping[str, Any]],
    input_directory: str | Path,
    unit_system: AbstractUnitSystem,
    quad_order: int,
) -> dict[str, ProjectedBinning]:
    """Build the named `ProjectedBinning`s from a resolved configuration.

    Each binning's aperture geometry (`min_x`, `min_y`, `x_extent`,
    `y_extent`, `PA`) comes from the configuration itself rather than a data
    file. The same schema and quantity validators as
    `ProjectedBinning.from_settings` run before its ``bins_file`` is read.

    Args:
        spatial_binnings: Mapping of unique identifiers to each binning's
            resolved settings, e.g. a resolved configuration's
            ``spatial_binnings`` section.
        input_directory: Directory that each ``bins_file`` is resolved
            against, e.g. a resolved configuration's
            ``io_settings.input_directory``.
        unit_system: The unit system to convert each binning's quantities
            into.
        quad_order: Order of the fixed Gauss-Legendre quadrature used for the
            x integral within each pixel, e.g. a resolved configuration's
            ``mge_settings.projected_mass_quad_order``.

    Returns:
        A dict mapping each identifier to its `ProjectedBinning`.
    """
    directory = Path(input_directory)
    binnings: dict[str, ProjectedBinning] = {}
    for name, raw_settings in spatial_binnings.items():
        path = f"spatial_binnings.{name}"
        settings = _validated_settings(
            raw_settings, require_bins_file=True, path=path
        )
        quantities = _declared_quantities(
            settings, unit_system.angle, path=path
        )
        bins = _validated_bins(
            np.load(directory / settings["bins_file"]), path=f"{path}.bins"
        )
        binnings[name] = ProjectedBinning(
            bins=bins, quad_order=quad_order, **quantities
        )
    return binnings
