"""Runtime population-data objects built from resolved TNT configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Self

import astropy.units as au
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import unxt as u
from astropy.table import QTable
from unxt import AbstractUnitSystem, Quantity

from tnt.config_parsing import (
    ConfigMapping,
    _finite,
    _mapping,
    _positive_finite,
    _read_bin_ids,
    _reject_unknown,
    _required,
    _resolve_typed_reference,
    _string,
)
from tnt.spatial_binnings import ProjectedBinning


class Populations(eqx.Module):
    """Population measurements observed in one projected spatial binning.

    Population properties may have different physical dimensions, so their
    values and uncertainties are stored as parallel tuples of `Quantity`
    arrays rather than combined into one unitless matrix. `property_names`
    gives both tuples' order.
    """

    name: str = eqx.field(static=True)
    data_file: Path = eqx.field(static=True)
    binning: ProjectedBinning
    bin_ids: jnp.ndarray
    property_names: tuple[str, ...] = eqx.field(static=True)
    values: tuple[Quantity, ...]
    uncertainties: tuple[Quantity, ...]

    @property
    def n_spatial_bins(self) -> int:
        """Return the number of observed spatial bins."""
        return int(self.bin_ids.shape[0])

    @property
    def n_properties(self) -> int:
        """Return the number of population properties in each spatial bin."""
        return len(self.property_names)

    def values_and_uncertainties(self, name: str) -> tuple[Quantity, Quantity]:
        """Return one named population property's values and uncertainties."""
        try:
            index = self.property_names.index(name)
        except ValueError as error:
            available = ", ".join(self.property_names)
            raise KeyError(
                f"Unknown population property {name!r}; available: {available}."
            ) from error
        return self.values[index], self.uncertainties[index]

    @classmethod
    def from_config(
        cls,
        *,
        name: str,
        settings: ConfigMapping,
        data_file: Path,
        binning: ProjectedBinning,
        unit_system: AbstractUnitSystem,
    ) -> Self:
        """Read and construct one configured population data set."""
        path = f"population_data.{name}"
        _reject_unknown(settings, {"binning", "data_file"}, path)
        table = QTable.read(data_file, format="ascii.ecsv")
        bin_ids = _read_bin_ids(table, "vbin_id", data_file)
        _validate_bin_ids_against_binning(bin_ids, binning, data_file)
        property_names = _population_property_names(table, data_file)

        values: list[Quantity] = []
        uncertainties: list[Quantity] = []
        for property_name in property_names:
            value, uncertainty = _read_property_pair(
                table,
                property_name,
                unit_system,
                data_file,
            )
            values.append(value)
            uncertainties.append(uncertainty)

        return cls(
            name=name,
            data_file=data_file,
            binning=binning,
            bin_ids=bin_ids,
            property_names=property_names,
            values=tuple(values),
            uncertainties=tuple(uncertainties),
        )


def build_populations(
    population_data: Mapping[str, ConfigMapping],
    input_directory: str | Path,
    unit_system: AbstractUnitSystem,
    spatial_binnings: Mapping[str, ProjectedBinning],
) -> dict[str, Populations]:
    """Build named population objects from resolved configuration data.

    Args:
        population_data: Resolved ``population_data`` registry.
        input_directory: Directory against which data filenames are resolved.
        unit_system: Internal unit system used by runtime quantities.
        spatial_binnings: Already-built named spatial-binning objects.

    Returns:
        A mapping from configured names to population runtime objects.
    """
    directory = Path(input_directory)
    built: dict[str, Populations] = {}
    for name, settings_value in population_data.items():
        path = f"population_data.{name}"
        if not isinstance(name, str) or not name:
            raise ValueError("population_data names must be non-empty strings.")
        settings = _mapping(settings_value, path)
        _reject_unknown(settings, {"binning", "data_file"}, path)
        filename = _string(_required(settings, "data_file", path), f"{path}.data_file")
        binning_name = _string(_required(settings, "binning", path), f"{path}.binning")
        binning = _resolve_typed_reference(
            spatial_binnings,
            binning_name,
            f"{path}.binning",
            "spatial_binnings",
            ProjectedBinning,
        )
        data_file = Path(filename)
        if not data_file.is_absolute():
            data_file = directory / data_file
        built[name] = Populations.from_config(
            name=name,
            settings=settings,
            data_file=data_file,
            binning=binning,
            unit_system=unit_system,
        )
    return built


def _validate_bin_ids_against_binning(
    bin_ids: jnp.ndarray,
    binning: ProjectedBinning,
    data_file: Path,
) -> None:
    available = set(np.asarray(binning.bins).ravel().tolist()) - {0}
    unknown = sorted(set(np.asarray(bin_ids).tolist()) - available)
    if unknown:
        names = ", ".join(str(value) for value in unknown)
        raise ValueError(
            f"{data_file}: spatial bin ID(s) absent from the referenced "
            f"binning: {names}."
        )


def _population_property_names(table: QTable, data_file: Path) -> tuple[str, ...]:
    columns = [name for name in table.colnames if name != "vbin_id"]
    names = set(columns)
    properties = [name for name in columns if f"d{name}" in names]
    covered = set(properties) | {f"d{name}" for name in properties}
    overlapping = set(properties) & {f"d{name}" for name in properties}
    if overlapping:
        names_text = ", ".join(sorted(overlapping))
        raise ValueError(
            f"{data_file}: ambiguous population value/error columns: {names_text}."
        )
    unpaired = names - covered
    if unpaired:
        names_text = ", ".join(sorted(unpaired))
        raise ValueError(
            f"{data_file}: population columns must occur as value/dvalue "
            f"pairs; unpaired column(s): {names_text}."
        )
    if not properties:
        raise ValueError(
            f"{data_file}: at least one population value/dvalue pair is required."
        )
    return tuple(properties)


def _read_property_pair(
    table: QTable,
    name: str,
    unit_system: AbstractUnitSystem,
    data_file: Path,
) -> tuple[Quantity, Quantity]:
    uncertainty_name = f"d{name}"
    value_column = table[name]
    uncertainty_column = table[uncertainty_name]
    value_unit = value_column.unit or au.dimensionless_unscaled
    uncertainty_unit = uncertainty_column.unit or au.dimensionless_unscaled
    if not uncertainty_unit.is_equivalent(value_unit):
        raise au.UnitConversionError(
            f"{data_file}: columns {name} and {uncertainty_name} must have "
            "equivalent units."
        )
    try:
        target_unit = unit_system[u.dimension(value_unit.physical_type)]
        values = Quantity.from_(value_column.to(target_unit))
        uncertainties = Quantity.from_(uncertainty_column.to(target_unit))
    except (TypeError, ValueError, au.UnitConversionError) as error:
        raise au.UnitConversionError(
            f"{data_file}: population property {name} has unsupported unit "
            f"{value_unit}."
        ) from error
    _finite(values.ustrip(target_unit), f"{data_file}: {name}")
    _positive_finite(
        uncertainties.ustrip(target_unit), f"{data_file}: {uncertainty_name}"
    )
    return values, uncertainties
