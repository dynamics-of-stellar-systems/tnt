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

    def to_mass(self, m_over_l: Quantity) -> MassMGE:
        """Convert to a MassMGE given a mass-to-light ratio.

        `sigma`, `q`, and `PA_twist` are unaffected and carried over
        unchanged -- only `I` (and hence what it represents) changes.

        Args:
            m_over_l: The mass-to-light ratio (e.g. in Msun/Lsun), either a
                single value applied to every component, or an array with
                one value per Gaussian component.

        Returns:
            A `MassMGE` with ``I = self.I * m_over_l``.

        Raises:
            ValueError: If `m_over_l` is array-valued and its length
                doesn't match the number of Gaussian components.
        """
        if m_over_l.ndim > 0 and m_over_l.shape[0] != self.I.shape[0]:
            raise ValueError(
                f"m_over_l has {m_over_l.shape[0]} components, but this MGE "
                f"has {self.I.shape[0]}."
            )

        return MassMGE(
            I=self.I * m_over_l, sigma=self.sigma, q=self.q, PA_twist=self.PA_twist
        )


class MassMGE(AbstractMGE):
    """An MGE of a mass surface-density distribution (``I`` in e.g. Msun/arcsec2)."""

    _intensity_attr: ClassVar[str] = "mass"


_MGE_CLASSES: tuple[type[AbstractMGE], ...] = (LightMGE, MassMGE)


def read_mge(path: str | Path, unit_system: AbstractUnitSystem) -> AbstractMGE:
    """Read an MGE from an ECSV file, inferring whether it's light or mass.

    The kind is inferred from the declared unit of the file's ``I`` column:
    whichever of `LightMGE` (power/angle**2) or `MassMGE` (mass/angle**2) it
    is dimensionally consistent with. This makes the check meaningful -- a
    file with the wrong kind of units for its intended use is rejected here,
    rather than silently accepted.

    Args:
        path: Path to the ECSV file.
        unit_system: The unit system to convert the columns into.

    Returns:
        A `LightMGE` or `MassMGE`, whichever matches the file's ``I`` column.

    Raises:
        ValueError: If the ``I`` column's unit doesn't match any known MGE
            kind.
    """
    table = QTable.read(path, format="ascii.ecsv")
    intensity_unit = table["I"].unit

    for cls in _MGE_CLASSES:
        target_unit = getattr(unit_system, cls._intensity_attr) / unit_system.angle**2
        if intensity_unit.is_equivalent(target_unit):
            return cls.from_qtable(table, unit_system)

    expected = [
        getattr(unit_system, cls._intensity_attr) / unit_system.angle**2
        for cls in _MGE_CLASSES
    ]
    raise ValueError(
        f"Could not infer MGE kind for {path}: its I column has unit "
        f"{intensity_unit!r}, which is not equivalent to any of {expected!r}."
    )
