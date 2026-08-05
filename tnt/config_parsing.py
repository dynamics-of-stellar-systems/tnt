"""Shared helpers for building named runtime objects from resolved TNT
configuration entries and their data files.

Used by `tnt.kinematics` and `tnt.populations` (and any future module that
reads a `<registry>.<name>` config entry plus a data file into a validated
runtime object) -- as distinct from `tnt.configuration_validation`, which
validates the whole resolved config's schema up front, before any data file
is opened.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from astropy.table import QTable

ConfigMapping = Mapping[str, Any]


def _mapping(value: Any, path: str) -> ConfigMapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    return value


def _required(settings: ConfigMapping, key: str, path: str) -> Any:
    if key not in settings:
        raise ValueError(f"{path} is missing required field: {key}.")
    return settings[key]


def _reject_unknown(settings: ConfigMapping, allowed: set[str], path: str) -> None:
    unknown = set(settings) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{path} contains unknown field(s): {names}.")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{path} must be a non-empty string.")
    return value


def _resolve_reference(
    registry: Mapping[str, Any], name: str, path: str, registry_name: str
) -> Any:
    try:
        return registry[name]
    except KeyError as error:
        raise ValueError(
            f"{path} references unknown {registry_name} entry {name!r}."
        ) from error


def _resolve_typed_reference(
    registry: Mapping[str, Any],
    name: str,
    path: str,
    registry_name: str,
    expected_type: type[Any] | tuple[type[Any], ...],
) -> Any:
    """Resolve a named object and enforce its runtime type."""
    value = _resolve_reference(registry, name, path, registry_name)
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            expected = " or ".join(cls.__name__ for cls in expected_type)
        else:
            expected = expected_type.__name__
        raise TypeError(
            f"{path} must resolve to {expected}, got {type(value).__name__}."
        )
    return value


def _read_bin_ids(table: QTable, column: str, data_file: Path) -> jnp.ndarray:
    if column not in table.colnames:
        raise ValueError(f"{data_file} is missing required column: {column}.")
    return _validated_bin_ids(table[column], data_file)


def _validated_bin_ids(values: Any, data_file: Path) -> jnp.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{data_file}: spatial bin IDs must be a non-empty vector.")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{data_file}: spatial bin IDs must be integers.")
    if np.any(array <= 0) or np.unique(array).size != array.size:
        raise ValueError(f"{data_file}: spatial bin IDs must be positive and unique.")
    return jnp.asarray(array)


def _finite(values: Any, path: str) -> None:
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError(f"{path} must contain only finite values.")


def _positive_finite(values: Any, path: str) -> None:
    _finite(values, path)
    if not bool(jnp.all(jnp.asarray(values) > 0)):
        raise ValueError(f"{path} must contain only positive values.")
