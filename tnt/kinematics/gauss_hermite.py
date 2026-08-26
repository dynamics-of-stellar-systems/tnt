"""Line-of-sight kinematics represented by Gauss-Hermite moments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Self

import equinox as eqx
import jax.numpy as jnp
import unxt as u
from astropy.table import QTable
from unxt import AbstractUnitSystem, Quantity

from tnt.kinematics.base import (
    AbstractKinematics,
    Histogram,
    _explicit_histogram,
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
    _integer,
    _mapping,
    _nonnegative_number,
    _positive_finite,
    _positive_number,
    _read_bin_ids,
    _reject_unknown_keys,
    _required,
)


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
        dispersion_error = _read_quantity_column(table, "dsigma", speed_unit, data_file)
        _same_length(table, bin_ids.shape[0], data_file)
        _positive_finite(dispersion.ustrip(speed_unit), f"{data_file}: sigma")

        velocity_error = _quadrature_quantity(
            velocity_error, systematics["v"], speed_unit
        )
        dispersion_error = _quadrature_quantity(
            dispersion_error, systematics["sigma"], speed_unit
        )
        _positive_finite(velocity_error.ustrip(speed_unit), f"{data_file}: dv")
        _positive_finite(dispersion_error.ustrip(speed_unit), f"{data_file}: dsigma")

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
    expected = {"v", "sigma"} | {f"h{order}" for order in range(3, maximum_order + 1)}
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


def _build_gh_histogram(
    settings: ConfigMapping,
    velocity: Quantity,
    dispersion: Quantity,
    unit_system: AbstractUnitSystem,
    path: str,
) -> Histogram:
    speed_unit = unit_system[u.dimension("speed")]
    histogram = _mapping(_required(settings, "histogram", path), f"{path}.histogram")
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
