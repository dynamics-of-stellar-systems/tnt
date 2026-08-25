"""Line-of-sight velocity distributions sampled in velocity bins."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar, Self

import astropy.units as au
import jax.numpy as jnp
import numpy as np
import unxt as u
from astropy.table import QTable
from unxt import AbstractUnitSystem, Quantity

from tnt.kinematics.base import (
    AbstractKinematics,
    Histogram,
    _explicit_histogram,
    _nonnegative_finite,
    _odd_ceiling,
    _read_dimensionless_column,
    _same_length,
)
from tnt.mge import LightMGE, MassMGE
from tnt.spatial_binnings import ProjectedBinning, _validate_bin_ids_cover_binning
from tnt.units import normalize_unitful_value
from tnt.validation import (
    ConfigMapping,
    _finite,
    _mapping,
    _positive_finite,
    _positive_number,
    _read_bin_ids,
    _reject_unknown_keys,
    _required,
)


class BayesLOSVD(AbstractKinematics):
    """Line-of-sight velocity distributions sampled in velocity bins."""

    _type: ClassVar[str] = "bayes_losvd"
    _allowed_settings: ClassVar[set[str]] = {
        "binning",
        "data_file",
        "histogram",
        "mge",
        "type",
    }

    velocity_centers: Quantity
    velocity_bin_width: Quantity
    losvd: jnp.ndarray
    losvd_uncertainty: jnp.ndarray
    bin_flux: jnp.ndarray
    mean_velocity: Quantity
    dispersion: Quantity

    @classmethod
    def from_config(
        cls,
        *,
        name: str,
        settings: ConfigMapping,
        data_file: Path,
        binning: ProjectedBinning,
        mge: LightMGE | MassMGE | None,
        unit_system: AbstractUnitSystem,
    ) -> Self:
        """Read and construct one configured Bayesian LOSVD data set."""
        path = f"kinematic_data.{name}"
        _reject_unknown_keys(settings, cls._allowed_settings, path)
        table = QTable.read(data_file, format="ascii.ecsv")
        bin_ids = _read_bin_ids(table, data_file)
        _validate_bin_ids_cover_binning(bin_ids, binning, data_file)
        _same_length(table, bin_ids.shape[0], data_file)
        speed_unit = unit_system[u.dimension("speed")]
        centers, data_bin_width = _read_losvd_velocity_grid(
            table, speed_unit, data_file
        )
        losvd, uncertainty = _read_losvd_columns(table, centers.shape[0], data_file)
        flux = _read_dimensionless_column(table, "bin_flux", data_file)
        _nonnegative_finite(losvd, f"{data_file}: LOSVD values")
        _positive_finite(uncertainty, f"{data_file}: LOSVD uncertainties")
        _nonnegative_finite(flux, f"{data_file}: bin_flux")
        if not bool(jnp.sum(flux) > 0):
            raise ValueError(f"{data_file}: bin_flux must have a positive sum.")

        mean_velocity, dispersion = _losvd_moments(losvd, centers, data_file)
        histogram_settings = _mapping(
            _required(settings, "histogram", path), f"{path}.histogram"
        )
        if set(histogram_settings) == {"width", "center", "bins"}:
            histogram = _explicit_histogram(histogram_settings, unit_system, path)
        else:
            centers, mean_velocity, histogram = _build_losvd_histogram(
                histogram_settings,
                centers,
                data_bin_width,
                mean_velocity,
                flux,
                unit_system,
                f"{path}.histogram",
            )
        return cls(
            name=name,
            data_file=data_file,
            binning=binning,
            mge=mge,
            histogram=histogram,
            bin_ids=bin_ids,
            velocity_centers=centers,
            velocity_bin_width=data_bin_width,
            losvd=losvd,
            losvd_uncertainty=uncertainty,
            bin_flux=flux,
            mean_velocity=mean_velocity,
            dispersion=dispersion,
        )

    def observed_values_and_uncertainties(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return LOSVD values and uncertainties in solver order."""
        return self.losvd, self.losvd_uncertainty


def _read_losvd_velocity_grid(
    table: QTable, speed_unit: Any, data_file: Path
) -> tuple[Quantity, Quantity]:
    missing = {"vcent", "dv", "velocity_unit"} - set(table.meta)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{data_file} metadata is missing required field(s): {names}.")
    source_unit = au.Unit(table.meta["velocity_unit"])
    centers = source_unit.to(speed_unit, np.asarray(table.meta["vcent"], dtype=float))
    data_bin_width = float(source_unit.to(speed_unit, table.meta["dv"]))
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError(f"{data_file}: metadata vcent must be a non-empty vector.")
    _finite(centers, f"{data_file}: metadata vcent")
    if data_bin_width <= 0 or not math.isfinite(data_bin_width):
        raise ValueError(f"{data_file}: metadata dv must be positive and finite.")
    if centers.size > 1 and not np.allclose(np.diff(centers), data_bin_width):
        raise ValueError(f"{data_file}: vcent must be uniformly spaced by metadata dv.")
    return (
        Quantity(jnp.asarray(centers), speed_unit),
        Quantity(data_bin_width, speed_unit),
    )


def _read_losvd_columns(
    table: QTable, velocity_bins: int, data_file: Path
) -> tuple[jnp.ndarray, jnp.ndarray]:
    values, errors = [], []
    for index in range(velocity_bins):
        values.append(_read_dimensionless_column(table, f"losvd_{index}", data_file))
        errors.append(_read_dimensionless_column(table, f"dlosvd_{index}", data_file))
    return jnp.column_stack(values), jnp.column_stack(errors)


def _losvd_moments(
    losvd: jnp.ndarray, centers: Quantity, data_file: Path
) -> tuple[Quantity, Quantity]:
    sums = jnp.sum(losvd, axis=1)
    _positive_finite(sums, f"{data_file}: LOSVD row sums")
    probability = losvd / sums[:, None]
    velocity = centers.ustrip(centers.unit)
    mean = jnp.sum(probability * velocity, axis=1)
    variance = jnp.sum(probability * (velocity[None, :] - mean[:, None]) ** 2, axis=1)
    _positive_finite(variance, f"{data_file}: LOSVD velocity variances")
    return Quantity(mean, centers.unit), Quantity(jnp.sqrt(variance), centers.unit)


def _build_losvd_histogram(
    settings: ConfigMapping,
    centers: Quantity,
    data_bin_width: Quantity,
    mean_velocity: Quantity,
    flux: jnp.ndarray,
    unit_system: AbstractUnitSystem,
    path: str,
) -> tuple[Quantity, Quantity, Histogram]:
    speed_unit = unit_system[u.dimension("speed")]
    allowed = {"center", "oversampling_factor", "systemic_velocity", "width_scale"}
    _reject_unknown_keys(settings, allowed, path)
    missing = allowed - set(settings)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required field(s): {names}.")
    if settings["systemic_velocity"] != "flux_weighted":
        raise ValueError(f"{path}.systemic_velocity must be 'flux_weighted'.")
    scale = _positive_number(settings["width_scale"], f"{path}.width_scale")
    oversampling = _positive_number(
        settings["oversampling_factor"], f"{path}.oversampling_factor"
    )
    center = normalize_unitful_value(
        settings["center"], "speed", unit_system, f"{path}.center"
    )
    systemic = jnp.sum(flux * mean_velocity.ustrip(speed_unit)) / jnp.sum(flux)
    shifted_centers = centers - Quantity(systemic, speed_unit)
    shifted_mean = mean_velocity - Quantity(systemic, speed_unit)
    edge_extent = jnp.max(jnp.abs(shifted_centers.ustrip(speed_unit)))
    edge_extent = edge_extent + data_bin_width.ustrip(speed_unit) / 2
    width = float(2 * scale * edge_extent)
    approximate_bin_width = float(data_bin_width.ustrip(speed_unit) / oversampling)
    bins = _odd_ceiling(width / approximate_bin_width)
    histogram = Histogram(
        width=Quantity(width, speed_unit),
        center=Quantity(center, speed_unit),
        bins=bins,
    )
    return shifted_centers, shifted_mean, histogram
