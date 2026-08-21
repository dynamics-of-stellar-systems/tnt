"""Runtime kinematics objects built from resolved TNT configuration data."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

import astropy.units as au
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import unxt as u
from astropy.table import QTable
from unxt import AbstractUnitSystem, Quantity

from tnt.config_parsing import (
    BIN_ID_COLUMN,
    ConfigMapping,
    _finite,
    _integer,
    _mapping,
    _nonnegative_number,
    _positive_finite,
    _positive_number,
    _read_bin_ids,
    _reject_unknown_keys,
    _required,
    _required_string,
    _resolve_typed_reference,
    _string,
    _validated_bin_ids,
)
from tnt.mge import LightMGE, MassMGE
from tnt.spatial_binnings import ProjectedBinning, _validate_bin_ids_cover_binning
from tnt.units import normalize_unitful_value

if TYPE_CHECKING:
    from tnt.orbit_library import OrbitLibrary

_LOGGER = logging.getLogger(__name__)


class Histogram(eqx.Module):
    """A uniformly spaced one-dimensional velocity histogram."""

    width: Quantity
    center: Quantity
    bins: int = eqx.field(static=True)

    @property
    def bin_width(self) -> Quantity:
        """Return the width of one velocity bin."""
        return self.width / self.bins

    @property
    def edges(self) -> Quantity:
        """Return the velocity-bin edges."""
        offsets = jnp.linspace(-0.5, 0.5, self.bins + 1)
        return self.center + offsets * self.width

    @property
    def centers(self) -> Quantity:
        """Return the velocity-bin centers."""
        edges = self.edges
        return (edges[:-1] + edges[1:]) / 2


class Histogram2D(eqx.Module):
    """Two uniformly spaced proper-motion velocity histograms."""

    width: Quantity
    center: Quantity
    bins: tuple[int, int] = eqx.field(static=True)

    @property
    def bin_width(self) -> Quantity:
        """Return the bin width along each velocity axis."""
        return self.width / jnp.asarray(self.bins)


class AbstractKinematics(eqx.Module):
    """Shared runtime state for a named kinematics data set."""

    _type: ClassVar[str]
    name: str = eqx.field(static=True)
    data_file: Path = eqx.field(static=True)
    binning: ProjectedBinning
    mge: LightMGE | MassMGE | None
    histogram: Histogram | Histogram2D
    bin_ids: jnp.ndarray

    @property
    def n_spatial_bins(self) -> int:
        """Return the number of observed spatial bins."""
        return int(self.bin_ids.shape[0])

    def observed_values_and_uncertainties(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return observations and uncertainties in solver column order."""
        raise NotImplementedError

    def design_matrix(self, orbit_library: OrbitLibrary) -> jnp.ndarray:
        """Project an orbit library into this data set's observables.

        This interface belongs to the model-architecture scaffold. Concrete
        projections will be implemented with the orbit integration and weight
        solving layers; until then, calling it fails explicitly.
        """
        raise NotImplementedError


class GaussHermite(AbstractKinematics):
    """Line-of-sight kinematics represented by Gauss-Hermite moments."""

    _type: ClassVar[str] = "gauss_hermite"
    _allowed_settings: ClassVar[set[str]] = {
        "binning",
        "data_file",
        "histogram",
        "maximum_gh_order",
        "mge",
        "observational_errors",
        "type",
    }

    maximum_gh_order: int = eqx.field(static=True)
    velocity: Quantity
    velocity_uncertainty: Quantity
    dispersion: Quantity
    dispersion_uncertainty: Quantity
    coefficients: jnp.ndarray
    coefficient_uncertainties: jnp.ndarray

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
        """Read and construct one configured Gauss-Hermite data set."""
        path = f"kinematic_data.{name}"
        _reject_unknown_keys(settings, cls._allowed_settings, path)
        maximum_order = _integer(
            _required(settings, "maximum_gh_order", path),
            f"{path}.maximum_gh_order",
        )
        if maximum_order < 2:
            raise ValueError(f"{path}.maximum_gh_order must be at least 2.")

        systematics = _gauss_hermite_systematics(
            settings, maximum_order, unit_system, path
        )
        table = QTable.read(data_file, format="ascii.ecsv")
        bin_ids = _read_bin_ids(table, data_file)
        _validate_bin_ids_cover_binning(bin_ids, binning, data_file)
        speed_unit = unit_system[u.dimension("speed")]

        velocity = _read_quantity_column(table, "v", speed_unit, data_file)
        velocity_error = _read_quantity_column(table, "dv", speed_unit, data_file)
        dispersion = _read_quantity_column(table, "sigma", speed_unit, data_file)
        dispersion_error = _read_quantity_column(
            table, "dsigma", speed_unit, data_file
        )
        _same_length(table, bin_ids.shape[0], data_file)
        _positive_finite(dispersion.ustrip(speed_unit), f"{data_file}: sigma")

        velocity_error = _quadrature_quantity(
            velocity_error, systematics["v"], speed_unit
        )
        dispersion_error = _quadrature_quantity(
            dispersion_error, systematics["sigma"], speed_unit
        )
        _positive_finite(velocity_error.ustrip(speed_unit), f"{data_file}: dv")
        _positive_finite(
            dispersion_error.ustrip(speed_unit), f"{data_file}: dsigma"
        )

        coefficients, coefficient_errors = _read_gh_coefficients(
            table, maximum_order, systematics, data_file
        )
        histogram = _build_gh_histogram(
            settings,
            velocity,
            dispersion,
            unit_system,
            path,
        )
        return cls(
            name=name,
            data_file=data_file,
            binning=binning,
            mge=mge,
            histogram=histogram,
            bin_ids=bin_ids,
            maximum_gh_order=maximum_order,
            velocity=velocity,
            velocity_uncertainty=velocity_error,
            dispersion=dispersion,
            dispersion_uncertainty=dispersion_error,
            coefficients=coefficients,
            coefficient_uncertainties=coefficient_errors,
        )

    def observed_values_and_uncertainties(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return observations and uncertainties in solver column order."""
        speed_unit = self.velocity.unit
        values = jnp.column_stack(
            (
                self.velocity.ustrip(speed_unit),
                self.dispersion.ustrip(speed_unit),
                self.coefficients,
            )
        )
        uncertainties = jnp.column_stack(
            (
                self.velocity_uncertainty.ustrip(speed_unit),
                self.dispersion_uncertainty.ustrip(speed_unit),
                self.coefficient_uncertainties,
            )
        )
        return values, uncertainties


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
        unit_system: AbstractUnitSystem,
    ) -> Self:
        """Read and construct one configured proper-motion data set."""
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
            speed_unit = unit_system[u.dimension("speed")]
            ranges = source_unit.to(
                speed_unit,
                np.asarray([archive["vxrange"], archive["vyrange"]], dtype=float),
            )

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
            width=Quantity(2 * jnp.asarray(ranges), speed_unit),
            center=Quantity(jnp.zeros(2), speed_unit),
            bins=(int(distribution.shape[1]), int(distribution.shape[2])),
        )
        histogram = _proper_motion_histogram(
            settings, observed_histogram, unit_system, path
        )
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


def _kinematics_class_registry() -> dict[str, type[AbstractKinematics]]:
    """Derive the configured type registry from concrete subclasses."""
    registry: dict[str, type[AbstractKinematics]] = {}
    for cls in AbstractKinematics.__subclasses__():
        kind = getattr(cls, "_type", None)
        if not isinstance(kind, str) or not kind:
            raise TypeError(f"{cls.__name__}._type must be a non-empty string.")
        if kind in registry:
            raise ValueError(
                f"Duplicate kinematics type {kind!r} on "
                f"{registry[kind].__name__} and {cls.__name__}."
            )
        registry[kind] = cls
    return registry


_KINEMATICS_CLASSES = _kinematics_class_registry()


def build_kinematics(
    kinematic_data: Mapping[str, ConfigMapping],
    input_directory: str | Path,
    unit_system: AbstractUnitSystem,
    spatial_binnings: Mapping[str, ProjectedBinning],
    mges: Mapping[str, LightMGE | MassMGE] | None = None,
) -> dict[str, AbstractKinematics]:
    """Build named kinematics objects from resolved configuration data.

    Args:
        kinematic_data: Resolved ``kinematic_data`` registry.
        input_directory: Directory against which data filenames are resolved.
        unit_system: Internal unit system used by runtime arrays.
        spatial_binnings: Already-built named spatial-binning objects.
        mges: Already-built named MGEs. May be omitted when no data set refers
            to an MGE.

    Returns:
        A mapping from configured names to concrete kinematics objects.
    """
    directory = Path(input_directory)
    available_mges: Mapping[str, LightMGE | MassMGE] = (
        {} if mges is None else mges
    )
    built: dict[str, AbstractKinematics] = {}
    for name, settings_value in kinematic_data.items():
        path = f"kinematic_data.{name}"
        if not isinstance(name, str) or not name:
            raise ValueError("kinematic_data names must be non-empty strings.")
        settings = _mapping(settings_value, path)
        kind = _required_string(settings, "type", path)
        try:
            cls = _KINEMATICS_CLASSES[kind]
        except KeyError as error:
            allowed = ", ".join(sorted(_KINEMATICS_CLASSES))
            raise ValueError(
                f"Unsupported {path}.type {kind!r}; expected one of: {allowed}."
            ) from error
        filename = _required_string(settings, "data_file", path)
        binning_name = _required_string(settings, "binning", path)
        binning = _resolve_typed_reference(
            spatial_binnings,
            binning_name,
            f"{path}.binning",
            "spatial_binnings",
            ProjectedBinning,
        )
        mge = None
        if "mge" in settings:
            mge_name = _string(settings["mge"], f"{path}.mge")
            mge = _resolve_typed_reference(
                available_mges,
                mge_name,
                f"{path}.mge",
                "MGEs",
                (LightMGE, MassMGE),
            )
        data_file = Path(filename)
        if not data_file.is_absolute():
            data_file = directory / data_file
        built[name] = cls.from_config(
            name=name,
            settings=settings,
            data_file=data_file,
            binning=binning,
            mge=mge,
            unit_system=unit_system,
        )
    return built


def _same_length(table: QTable, expected: int, data_file: Path) -> None:
    if len(table) != expected or expected == 0:
        raise ValueError(f"{data_file}: all columns must have at least one row.")


def _read_quantity_column(
    table: QTable, name: str, target_unit: Any, data_file: Path
) -> Quantity:
    if name not in table.colnames:
        raise ValueError(f"{data_file} is missing required column: {name}.")
    column = table[name]
    if column.unit is None:
        raise ValueError(f"{data_file}: column {name} must declare a unit.")
    quantity = Quantity.from_(column.to(target_unit))
    _finite(quantity.ustrip(target_unit), f"{data_file}: {name}")
    return quantity


def _read_dimensionless_column(
    table: QTable, name: str, data_file: Path
) -> jnp.ndarray:
    if name not in table.colnames:
        raise ValueError(f"{data_file} is missing required column: {name}.")
    column = table[name]
    if column.unit is not None and not column.unit.is_equivalent(
        au.dimensionless_unscaled
    ):
        raise au.UnitConversionError(
            f"{data_file}: column {name} must be dimensionless."
        )
    values = jnp.asarray(column)
    _finite(values, f"{data_file}: {name}")
    return values


def _nonnegative_finite(values: Any, path: str) -> None:
    _finite(values, path)
    if not bool(jnp.all(jnp.asarray(values) >= 0)):
        raise ValueError(f"{path} must contain only nonnegative values.")


def _quadrature_quantity(error: Quantity, systematic: float, unit: Any) -> Quantity:
    values = jnp.sqrt(error.ustrip(unit) ** 2 + systematic**2)
    return Quantity(values, unit)


def _gauss_hermite_systematics(
    settings: ConfigMapping,
    maximum_order: int,
    unit_system: AbstractUnitSystem,
    path: str,
) -> dict[str, float]:
    errors = _mapping(
        _required(settings, "observational_errors", path),
        f"{path}.observational_errors",
    )
    _reject_unknown_keys(
        errors, {"systematic_uncertainties"}, f"{path}.observational_errors"
    )
    systematics_path = f"{path}.observational_errors.systematic_uncertainties"
    systematics = _mapping(
        _required(errors, "systematic_uncertainties", f"{path}.observational_errors"),
        systematics_path,
    )
    expected = {"v", "sigma"} | {
        f"h{order}" for order in range(3, maximum_order + 1)
    }
    _reject_unknown_keys(systematics, expected, systematics_path)
    missing = expected - set(systematics)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{systematics_path} is missing required field(s): {names}.")
    result = {}
    for key in expected:
        key_path = f"{systematics_path}.{key}"
        value = (
            normalize_unitful_value(systematics[key], "speed", unit_system, key_path)
            if key in {"v", "sigma"}
            else systematics[key]
        )
        result[key] = _nonnegative_number(value, key_path)
    return result


def _read_gh_coefficients(
    table: QTable,
    maximum_order: int,
    systematics: Mapping[str, float],
    data_file: Path,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    values: list[jnp.ndarray] = []
    errors: list[jnp.ndarray] = []
    rows = len(table)
    for order in range(3, maximum_order + 1):
        value_name, error_name = f"h{order}", f"dh{order}"
        present = (value_name in table.colnames, error_name in table.colnames)
        if present[0] != present[1]:
            raise ValueError(
                f"{data_file}: columns {value_name} and {error_name} "
                "must appear together."
            )
        systematic = systematics[value_name]
        if present[0]:
            coefficient = _read_dimensionless_column(table, value_name, data_file)
            uncertainty = _read_dimensionless_column(table, error_name, data_file)
            uncertainty = jnp.sqrt(uncertainty**2 + systematic**2)
        else:
            coefficient = jnp.zeros(rows)
            uncertainty = jnp.full(rows, systematic)
        _positive_finite(uncertainty, f"{data_file}: {error_name}")
        values.append(coefficient)
        errors.append(uncertainty)
    if not values:
        return jnp.empty((rows, 0)), jnp.empty((rows, 0))
    return jnp.column_stack(values), jnp.column_stack(errors)


def _explicit_histogram(
    settings: ConfigMapping,
    unit_system: AbstractUnitSystem,
    parent_path: str,
) -> Histogram:
    path = (
        f"{parent_path}.histogram"
        if not parent_path.endswith(".histogram")
        else parent_path
    )
    _reject_unknown_keys(settings, {"bins", "center", "width"}, path)
    missing = {"bins", "center", "width"} - set(settings)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required field(s): {names}.")
    speed_unit = unit_system[u.dimension("speed")]
    width = normalize_unitful_value(
        settings["width"], "speed", unit_system, f"{path}.width"
    )
    _positive_number(width, f"{path}.width.value")
    center = normalize_unitful_value(
        settings["center"], "speed", unit_system, f"{path}.center"
    )
    bins = _integer(settings["bins"], f"{path}.bins")
    if bins <= 0 or bins % 2 == 0:
        raise ValueError(f"{path}.bins must be a positive odd integer.")
    return Histogram(
        width=Quantity(width, speed_unit),
        center=Quantity(center, speed_unit),
        bins=bins,
    )


def _build_gh_histogram(
    settings: ConfigMapping,
    velocity: Quantity,
    dispersion: Quantity,
    unit_system: AbstractUnitSystem,
    path: str,
) -> Histogram:
    speed_unit = unit_system[u.dimension("speed")]
    histogram = _mapping(
        _required(settings, "histogram", path), f"{path}.histogram"
    )
    if set(histogram) == {"width", "center", "bins"}:
        return _explicit_histogram(histogram, unit_system, path)
    histogram_path = f"{path}.histogram"
    allowed = {"bin_width_sigma_fraction", "center", "sigma_extent"}
    _reject_unknown_keys(histogram, allowed, histogram_path)
    missing = allowed - set(histogram)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{histogram_path} is missing required field(s): {names}.")
    extent = _positive_number(
        histogram["sigma_extent"], f"{histogram_path}.sigma_extent"
    )
    fraction = _positive_number(
        histogram["bin_width_sigma_fraction"],
        f"{histogram_path}.bin_width_sigma_fraction",
    )
    center = normalize_unitful_value(
        histogram["center"], "speed", unit_system, f"{histogram_path}.center"
    )
    v = velocity.ustrip(speed_unit)
    sigma = dispersion.ustrip(speed_unit)
    width = float(2 * jnp.max(jnp.abs(v) + extent * sigma))
    bins = _odd_ceiling(width / (fraction * float(jnp.min(sigma))))
    return Histogram(
        width=Quantity(width, speed_unit),
        center=Quantity(center, speed_unit),
        bins=bins,
    )


def _odd_ceiling(value: float) -> int:
    result = math.ceil(value)
    return result if result % 2 == 1 else result + 1


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
    _positive_finite(
        jnp.sum(distribution, axis=(1, 2)), f"{data_file}: PM_2dhist sums"
    )


def _proper_motion_histogram(
    settings: ConfigMapping,
    observed: Histogram2D,
    unit_system: AbstractUnitSystem,
    path: str,
) -> Histogram2D:
    if "histogram" not in settings:
        return observed
    configured = _explicit_histogram(
        _mapping(settings["histogram"], f"{path}.histogram"), unit_system, path
    )
    speed_unit = unit_system[u.dimension("speed")]
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
