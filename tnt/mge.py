"""Multi-Gaussian Expansion (MGE) models."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
from astropy.table import QTable
from unxt import AbstractUnitSystem, Quantity


class MGE(eqx.Module):
    """A Multi-Gaussian Expansion model of a surface-brightness distribution.

    Each Gaussian component is described by its peak intensity ``I``, width
    ``sigma``, axial ratio ``q``, and position-angle twist ``PA_twist``,
    stored as arrays converted to a unit system's units.
    """

    I: Quantity  # noqa: E741
    sigma: Quantity
    q: Quantity
    PA_twist: Quantity

    @classmethod
    def from_qtable(cls, table: QTable, unit_system: AbstractUnitSystem) -> MGE:
        """Build an MGE from a table, validating and converting its columns.

        Args:
            table: A table with columns ``I``, ``sigma``, ``q``, and
                ``PA_twist``, each carrying an astropy unit.
            unit_system: The unit system to convert the columns into.

        Returns:
            An `MGE` with columns converted to `unit_system`'s units.

        Raises:
            astropy.units.UnitConversionError: If a column's unit is not
                dimensionally consistent with the expected physical type.
        """
        intensity_unit = unit_system.power / unit_system.angle**2
        target_units = {
            "I": intensity_unit,
            "sigma": unit_system.angle,
            "q": "",
            "PA_twist": unit_system.angle,
        }

        columns = {
            name: Quantity.from_(table[name].to(unit))
            for name, unit in target_units.items()
        }

        return cls(**columns)

    @classmethod
    def read(cls, path: str | Path, unit_system: AbstractUnitSystem) -> MGE:
        """Read an MGE from an ECSV file, converting into a unit system's units.

        Args:
            path: Path to the ECSV file.
            unit_system: The unit system to convert the columns into.

        Returns:
            An `MGE` with columns converted to `unit_system`'s units.
        """
        table = QTable.read(path, format="ascii.ecsv")
        return cls.from_qtable(table, unit_system)
