"""Two-dimensional proper-motion velocity distributions."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import ClassVar, Self

import astropy.units as au
import jax.numpy as jnp
import numpy as np
from unxt import Quantity

from tnt.kinematics.base import (
    AbstractKinematics,
    Histogram2D,
    _explicit_histogram,
    _nonnegative_finite,
)
from tnt.kinematics.registry import register_kinematics
from tnt.mge import LightMGE, MassMGE
from tnt.spatial_binnings import ProjectedBinning, _validate_bin_ids_cover_binning
from tnt.units import validate_dimension
from tnt.validation import (
    BIN_ID_COLUMN,
    ConfigMapping,
    _mapping,
    _positive_finite,
    _positive_number,
    _reject_unknown_keys,
    _required,
    _validated_bin_ids,
)

_LOGGER = logging.getLogger(__name__)


@register_kinematics
class ProperMotions(AbstractKinematics):
    """Two-dimensional proper-motion velocity distributions."""

    _type: ClassVar[str] = "proper_motions"
    _allowed_settings: ClassVar[set[str]] = {
        "binning",
        "data_file",
        "histogram",
        "mge",
        "observational_errors",
        "type",
        "warning_thresholds",
    }

    distribution: jnp.ndarray
    uncertainty: jnp.ndarray
    stars_per_bin: jnp.ndarray
    normalization: jnp.ndarray

    @classmethod
    def from_config(
        cls,
        *,
        name: str,
        settings: ConfigMapping,
        data_file: Path,
        binning: ProjectedBinning,
        mge: LightMGE | MassMGE | None,
    ) -> Self:
        """Read and construct one configured proper-motion data set.

        The velocity ranges keep the unit declared in the NPZ archive; no
        unit-system conversion happens here (see `tnt.units`' module docstring).
        """
        path = f"kinematic_data.{name}"
        _reject_unknown_keys(settings, cls._allowed_settings, path)
        variance_scale = _proper_motion_variance_scale(settings, path)
        thresholds = _proper_motion_warning_thresholds(settings, path)

        with np.load(data_file, allow_pickle=False) as archive:
            required = {
                "PM_2dhist",
                "PM_2dhist_sigma",
                BIN_ID_COLUMN,
                "nstarbin",
                "velocity_unit",
                "vxrange",
                "vyrange",
            }
            missing = required - set(archive.files)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"{data_file} is missing required array(s): {names}.")
            distribution = jnp.asarray(archive["PM_2dhist"])
            uncertainty = jnp.asarray(archive["PM_2dhist_sigma"])
            bin_ids = _validated_bin_ids(archive[BIN_ID_COLUMN], data_file)
            stars = jnp.asarray(archive["nstarbin"])
            source_unit = au.Unit(str(archive["velocity_unit"].item()))
            validate_dimension(source_unit, "speed", f"{data_file}: velocity_unit")
            ranges = np.asarray([archive["vxrange"], archive["vyrange"]], dtype=float)

        _validate_bin_ids_cover_binning(bin_ids, binning, data_file)
        _validate_proper_motion_arrays(
            distribution, uncertainty, bin_ids, stars, data_file
        )
        uncertainty = uncertainty * math.sqrt(variance_scale)
        _positive_finite(uncertainty, f"{data_file}: PM_2dhist_sigma")
        normalization = jnp.sum(distribution, axis=(1, 2))
        distribution = distribution / normalization[:, None, None]
        uncertainty = uncertainty / normalization[:, None, None]
        observed_histogram = Histogram2D(
            width=Quantity(2 * jnp.asarray(ranges), source_unit),
            center=Quantity(jnp.zeros(2), source_unit),
            bins=(int(distribution.shape[1]), int(distribution.shape[2])),
        )
        histogram = _proper_motion_histogram(settings, observed_histogram, path)
        _warn_about_proper_motion_sampling(
            name, distribution, observed_histogram, thresholds
        )
        return cls(
            name=name,
            data_file=data_file,
            binning=binning,
            mge=mge,
            histogram=histogram,
            bin_ids=bin_ids,
            distribution=distribution,
            uncertainty=uncertainty,
            stars_per_bin=stars,
            normalization=normalization,
        )

    def observed_values_and_uncertainties(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return flattened velocity distributions in solver order."""
        shape = (self.n_spatial_bins, -1)
        return self.distribution.reshape(shape), self.uncertainty.reshape(shape)


def _proper_motion_variance_scale(settings: ConfigMapping, path: str) -> float:
    errors_path = f"{path}.observational_errors"
    errors = _mapping(_required(settings, "observational_errors", path), errors_path)
    _reject_unknown_keys(errors, {"variance_scale"}, errors_path)
    return _positive_number(
        _required(errors, "variance_scale", errors_path),
        f"{errors_path}.variance_scale",
    )


def _proper_motion_warning_thresholds(
    settings: ConfigMapping, path: str
) -> tuple[float, float]:
    thresholds_path = f"{path}.warning_thresholds"
    thresholds = _mapping(
        _required(settings, "warning_thresholds", path), thresholds_path
    )
    keys = {"max_bin_width_sigma_ratio", "min_histogram_width_sigma_ratio"}
    _reject_unknown_keys(thresholds, keys, thresholds_path)
    missing = keys - set(thresholds)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{thresholds_path} is missing required field(s): {names}.")
    return (
        _positive_number(
            thresholds["max_bin_width_sigma_ratio"],
            f"{thresholds_path}.max_bin_width_sigma_ratio",
        ),
        _positive_number(
            thresholds["min_histogram_width_sigma_ratio"],
            f"{thresholds_path}.min_histogram_width_sigma_ratio",
        ),
    )


def _validate_proper_motion_arrays(
    distribution: jnp.ndarray,
    uncertainty: jnp.ndarray,
    bin_ids: jnp.ndarray,
    stars: jnp.ndarray,
    data_file: Path,
) -> None:
    if distribution.ndim != 3 or distribution.shape != uncertainty.shape:
        raise ValueError(
            f"{data_file}: PM_2dhist and PM_2dhist_sigma must have the same "
            "(spatial bins, vx bins, vy bins) shape."
        )
    if distribution.shape[0] != bin_ids.shape[0] or stars.shape != bin_ids.shape:
        raise ValueError(
            f"{data_file}: bin_id and nstarbin must match the spatial-bin axis."
        )
    if distribution.shape[1] % 2 == 0 or distribution.shape[2] % 2 == 0:
        raise ValueError(f"{data_file}: proper-motion velocity-bin counts must be odd.")
    _nonnegative_finite(distribution, f"{data_file}: PM_2dhist")
    _positive_finite(uncertainty, f"{data_file}: PM_2dhist_sigma")
    _positive_finite(stars, f"{data_file}: nstarbin")
    _positive_finite(jnp.sum(distribution, axis=(1, 2)), f"{data_file}: PM_2dhist sums")


def _proper_motion_histogram(
    settings: ConfigMapping,
    observed: Histogram2D,
    path: str,
) -> Histogram2D:
    if "histogram" not in settings:
        return observed
    configured = _explicit_histogram(
        _mapping(settings["histogram"], f"{path}.histogram"), path
    )
    # Match the observed histogram's unit -- the local reference this data
    # set is expressed in -- rather than any run's unit system.
    speed_unit = observed.width.unit
    return Histogram2D(
        width=Quantity(jnp.repeat(configured.width.ustrip(speed_unit), 2), speed_unit),
        center=Quantity(
            jnp.repeat(configured.center.ustrip(speed_unit), 2), speed_unit
        ),
        bins=(configured.bins, configured.bins),
    )


def _warn_about_proper_motion_sampling(
    name: str,
    distribution: jnp.ndarray,
    histogram: Histogram2D,
    thresholds: tuple[float, float],
) -> None:
    max_bin_ratio, min_width_ratio = thresholds
    global_distribution = jnp.sum(distribution, axis=0)
    global_distribution = global_distribution / jnp.sum(global_distribution)
    widths = histogram.width.ustrip(histogram.width.unit)
    bin_widths = histogram.bin_width.ustrip(histogram.width.unit)
    centers = [
        jnp.linspace(-width / 2 + step / 2, width / 2 - step / 2, bins)
        for width, step, bins in zip(widths, bin_widths, histogram.bins, strict=True)
    ]
    marginals = (
        jnp.sum(global_distribution, axis=1),
        jnp.sum(global_distribution, axis=0),
    )
    for axis, values, marginal, width, step in zip(
        ("vx", "vy"), centers, marginals, widths, bin_widths, strict=True
    ):
        mean = jnp.sum(values * marginal)
        sigma = float(jnp.sqrt(jnp.sum((values - mean) ** 2 * marginal)))
        if sigma <= 0 or not math.isfinite(sigma):
            _LOGGER.warning("Kinematics %s has zero global %s dispersion.", name, axis)
            continue
        if float(step) / sigma > max_bin_ratio:
            _LOGGER.warning(
                "Kinematics %s %s bin width exceeds %.3g times its global dispersion.",
                name,
                axis,
                max_bin_ratio,
            )
        if float(width) / sigma < min_width_ratio:
            _LOGGER.warning(
                "Kinematics %s %s histogram width is below %.3g times its "
                "global dispersion.",
                name,
                axis,
                min_width_ratio,
            )
