"""Shared helpers for parsing and validating resolved TNT configuration data.

Used across the package wherever a piece of resolved configuration --
anywhere from the whole config dict (`tnt.configuration_validation`) down to
one `<registry>.<name>` entry plus its data file (`tnt.kinematics`,
`tnt.populations`) -- needs the same handful of primitives: check a value is
a mapping, look up a required field, reject unknown fields, validate a
number/string, or resolve a named cross-reference. Kept independent of any
one caller's domain so it doesn't accumulate domain-specific logic itself.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
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


def _reject_unknown_keys(
    mapping: ConfigMapping, allowed: Collection[str], path: str
) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown field(s): {', '.join(unknown)}.")


def _require_keys(mapping: ConfigMapping, required: set[str], path: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"{path} is missing required field(s): {', '.join(missing)}.")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} must be a non-empty string.")
    return value


def _required_mapping(mapping: ConfigMapping, key: str, path: str) -> ConfigMapping:
    """Return one required nested mapping."""
    return _mapping(_required(mapping, key, path), f"{path}.{key}")


def _required_string(mapping: ConfigMapping, key: str, path: str) -> str:
    """Return one required non-empty string."""
    return _string(_required(mapping, key, path), f"{path}.{key}")


def _optional_mapping(mapping: ConfigMapping, key: str, path: str) -> ConfigMapping:
    """Return one nested mapping, defaulting a missing key to empty."""
    return _mapping(mapping.get(key, {}), f"{path}.{key}")


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite.")
    return number


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer.")
    return value


def _positive_number(value: Any, path: str) -> float:
    number = _number(value, path)
    if number <= 0:
        raise ValueError(f"{path} must be greater than zero.")
    return number


def _nonnegative_number(value: Any, path: str) -> float:
    number = _number(value, path)
    if number < 0:
        raise ValueError(f"{path} must not be negative.")
    return number


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
