"""Shared kinematics base classes and helpers used by more than one concrete type."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import astropy.units as au
import equinox as eqx
import jax.numpy as jnp
from astropy.table import QTable
from unxt import Quantity

from tnt.mge import LightMGE, MassMGE
from tnt.spatial_binnings import ProjectedBinning
from tnt.units import declared_quantity
from tnt.validation import (
    ConfigMapping,
    _finite,
    _integer,
    _positive_number,
    _reject_unknown_keys,
)

if TYPE_CHECKING:
    from tnt.orbit_library import OrbitLibrary


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


def _same_length(table: QTable, expected: int, data_file: Path) -> None:
    if len(table) != expected or expected == 0:
        raise ValueError(f"{data_file}: all columns must have at least one row.")


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


def _explicit_histogram(
    settings: ConfigMapping,
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
    width = declared_quantity(settings["width"], "speed", f"{path}.width")
    _positive_number(float(width.ustrip(width.unit)), f"{path}.width.value")
    center = declared_quantity(settings["center"], "speed", f"{path}.center")
    bins = _integer(settings["bins"], f"{path}.bins")
    if bins <= 0 or bins % 2 == 0:
        raise ValueError(f"{path}.bins must be a positive odd integer.")
    return Histogram(width=width, center=center, bins=bins)


def _odd_ceiling(value: float) -> int:
    result = math.ceil(value)
    return result if result % 2 == 1 else result + 1
