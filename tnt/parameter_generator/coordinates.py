"""Exact proposal identity in complete, declared parameter coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import astropy.units as au
import numpy as np

from tnt.all_models import AllModels


def parameter_units(
    potential_settings: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Return all parameter columns, including fixed values, in stable order."""
    return {
        f"{component}.{parameter}": settings.get("unit", "")
        for component, spec in sorted(potential_settings.items())
        for parameter, settings in sorted(spec.get("parameters", {}).items())
    }


def existing_parameter_rows(
    all_models: AllModels, units: Mapping[str, str]
) -> np.ndarray:
    """Read requested columns in declared units, without changing the table."""
    rows = np.empty((len(all_models), len(units)), dtype=float)
    if not len(all_models):
        return rows
    for index, (site, unit) in enumerate(units.items()):
        if site not in all_models.table.colnames:
            raise ValueError(f"AllModels is missing parameter column: {site}")
        try:
            rows[:, index] = au.Quantity(all_models.table[site]).to_value(unit)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Cannot read AllModels parameter {site} in unit {unit!r}"
            ) from error
    if not np.isfinite(rows).all():
        raise ValueError("AllModels contains non-finite parameter values")
    return rows


def proposal_key(
    candidate: Mapping[str, Mapping[str, Any]], units: Mapping[str, str]
) -> tuple[float, ...]:
    """Identify a complete finite scalar proposal, converting compatible units."""
    values = {
        f"{component}.{parameter}": value
        for component, parameters in candidate.items()
        for parameter, value in parameters.items()
    }
    if values.keys() != units.keys():
        raise ValueError("Proposal parameter columns do not match the potential")
    key = []
    for site, unit in units.items():
        scalar = np.asarray(values[site].ustrip(unit))
        if scalar.ndim != 0 or not np.isfinite(scalar):
            raise ValueError(f"Proposal parameter {site} must be a finite scalar")
        key.append(float(scalar))
    return tuple(key)
