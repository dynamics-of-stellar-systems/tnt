"""Multi-Gaussian Expansion (MGE) models."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Self

import equinox as eqx
from astropy.table import QTable
from unxt import AbstractUnitSystem, Quantity

from tnt import units


class AbstractMGE(eqx.Module):
    """Shared structure and behaviour for MGE models.

    Each Gaussian component is described by its peak intensity ``I``, width
    ``sigma``, axial ratio ``q``, and position-angle twist ``PA_twist``,
    stored as arrays converted to a unit system's units. Subclasses fix which
    physical dimension ``I`` represents (e.g. light or mass) by setting
    `_intensity_attr` to the corresponding `unxt.AbstractUnitSystem`
    attribute name. Not meant to be instantiated directly -- use `LightMGE`
    or `MassMGE`.
    """

    _intensity_attr: ClassVar[str]

    I: Quantity  # noqa: E741
    sigma: Quantity
    q: Quantity
    PA_twist: Quantity

    @classmethod
    def from_qtable(cls, table: QTable, unit_system: AbstractUnitSystem) -> Self:
        """Build an MGE from a table, validating and converting its columns.

        Args:
            table: A table with columns ``I``, ``sigma``, ``q``, and
                ``PA_twist``, each carrying an astropy unit.
            unit_system: The unit system to convert the columns into.

        Returns:
            An MGE with columns converted to `unit_system`'s units.

        Raises:
            astropy.units.UnitConversionError: If a column's unit is not
                dimensionally consistent with the expected physical type.
        """
        intensity_unit = getattr(unit_system, cls._intensity_attr) / unit_system.angle**2
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
    def read(cls, path: str | Path, unit_system: AbstractUnitSystem) -> Self:
        """Read an MGE from an ECSV file, converting into a unit system's units.

        Args:
            path: Path to the ECSV file.
            unit_system: The unit system to convert the columns into.

        Returns:
            An MGE with columns converted to `unit_system`'s units.
        """
        table = QTable.read(path, format="ascii.ecsv")
        return cls.from_qtable(table, unit_system)

    def angular_to_physical(self, distance: Quantity) -> Self:
        """Convert `sigma` and `I` from angular to physical (length) units.

        `q` (dimensionless) and `PA_twist` (an orientation angle, not a
        spatial size) are unaffected and carried over unchanged.

        Args:
            distance: The distance to the object.

        Returns:
            A new MGE with `sigma` in `distance`'s unit and `I` converted
            to match.
        """
        sigma_physical = units.angular_to_physical(self.sigma, distance)
        solid_angle = Quantity(1.0, f"{self.sigma.unit}2")
        I_physical = self.I * solid_angle / distance**2  # noqa: N806

        return type(self)(
            I=I_physical, sigma=sigma_physical, q=self.q, PA_twist=self.PA_twist
        )

    def physical_to_angular(self, distance: Quantity) -> Self:
        """Convert `sigma` and `I` from physical (length) to angular units.

        Inverse of `angular_to_physical`.

        Args:
            distance: The distance to the object.

        Returns:
            A new MGE with `sigma` in radians and `I` converted to match.
        """
        sigma_angular = units.physical_to_angular(self.sigma, distance)
        solid_angle = Quantity(1.0, f"{sigma_angular.unit}2")
        I_angular = self.I * distance**2 / solid_angle  # noqa: N806

        return type(self)(
            I=I_angular, sigma=sigma_angular, q=self.q, PA_twist=self.PA_twist
        )


class LightMGE(AbstractMGE):
    """An MGE of a surface-brightness distribution (``I`` in e.g. Lsun/arcsec2)."""

    _intensity_attr: ClassVar[str] = "power"


class MassMGE(AbstractMGE):
    """An MGE of a mass surface-density distribution (``I`` in e.g. Msun/arcsec2)."""

    _intensity_attr: ClassVar[str] = "mass"
