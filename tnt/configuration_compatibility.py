"""Configuration compatibility policy for appending to one `AllModels` search."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from tnt.all_models import AllModels
from tnt.config_parsing import _mapping as _parse_mapping
from tnt.config_parsing import _required_mapping as _parse_required_mapping
from tnt.configuration import ConfigDict, _read_yaml_bytes_mapping
from tnt.run_config_log import RunManifestReference

_SEARCH_PARAMETER_KEYS = {
    "fixed",
    "generator_settings",
    "latex_label",
    "logarithmic",
    "value",
}


class ConfigurationCompatibilityError(ValueError):
    """A resumed search would mix scientifically incompatible models."""


def ensure_resume_compatible(
    current_critical_configuration: Mapping[str, Any],
    baseline_run: RunManifestReference | None,
    all_models: AllModels,
    which_chi2: str,
) -> None:
    """Reject an incompatible baseline run or an unusable chi2 selection."""
    if baseline_run is not None:
        path = baseline_run.absolute_resolved_config_path
        historical_config = _read_yaml_bytes_mapping(
            path.read_bytes(),
            f"resolved configuration for baseline run {baseline_run.run_id}",
        )
        historical_critical = _critical_configuration(historical_config)
        differences = _different_paths(
            historical_critical,
            current_critical_configuration,
            "critical_configuration",
        )
        if differences:
            detail = ", ".join(differences[:20])
            if len(differences) > 20:
                detail += f", and {len(differences) - 20} more"
            raise ConfigurationCompatibilityError(
                f"Configuration for baseline run {baseline_run.run_id} is "
                f"incompatible with the current run; changed fields: {detail}."
            )
    _validate_selected_chi2(all_models, which_chi2)
    _validate_parameter_columns(all_models, current_critical_configuration)


def _critical_configuration(config: Mapping[str, Any]) -> ConfigDict:
    """Project a resolved runtime configuration onto scientifically fixed fields."""
    system_attributes = {
        key: deepcopy(value)
        for key, value in _mapping(config, "system_attributes").items()
        if key != "name"
    }
    return {
        "units": {"internal": deepcopy(_mapping(config, "units")["internal"])},
        "cosmological_parameters": deepcopy(
            _mapping(config, "cosmological_parameters")
        ),
        "system_attributes": system_attributes,
        "mge_settings": deepcopy(_mapping(config, "mge_settings")),
        # The current compatibility contract rejects every numerics change.
        "numerics_settings": deepcopy(_mapping(config, "numerics_settings")),
        "MGEs": deepcopy(_mapping(config, "MGEs")),
        "spatial_binnings": deepcopy(_mapping(config, "spatial_binnings")),
        "potential": _potential_schema(_mapping(config, "potential")),
        "kinematic_data": deepcopy(_mapping(config, "kinematic_data")),
        "population_data": deepcopy(_mapping(config, "population_data")),
        "orbit_library_settings": deepcopy(_mapping(config, "orbit_library_settings")),
        "weight_solver_settings": deepcopy(_mapping(config, "weight_solver_settings")),
    }


def _potential_schema(potential: Mapping[str, Any]) -> ConfigDict:
    """Keep component/parameter schema while excluding future search controls."""
    result: ConfigDict = {}
    for component_name, raw_component in potential.items():
        component = _require_mapping(raw_component, Path(f"potential.{component_name}"))
        schema = {
            key: deepcopy(value)
            for key, value in component.items()
            if key != "parameters"
        }
        parameters = _require_mapping(
            component.get("parameters", {}),
            Path(f"potential.{component_name}.parameters"),
        )
        schema["parameters"] = {
            parameter_name: {
                key: deepcopy(value)
                for key, value in _require_mapping(
                    raw_parameter,
                    Path(f"potential.{component_name}.parameters.{parameter_name}"),
                ).items()
                if key not in _SEARCH_PARAMETER_KEYS
            }
            for parameter_name, raw_parameter in parameters.items()
        }
        result[component_name] = schema
    return result


def _validate_selected_chi2(all_models: AllModels, which_chi2: str) -> None:
    """Require the current search metric for every successful historical model."""
    if not all_models.has_successful_model():
        return
    if which_chi2 not in all_models.table.colnames:
        raise ConfigurationCompatibilityError(
            f"Cannot resume with which_chi2={which_chi2!r}: the historical "
            "AllModels table has no such column."
        )
    successful = np.asarray(all_models.table["weights_done"], dtype=bool)
    values = np.asarray(all_models.table[which_chi2][successful], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ConfigurationCompatibilityError(
            f"Cannot resume with which_chi2={which_chi2!r}: it is not finite "
            "for every successful historical model."
        )


def _validate_parameter_columns(
    all_models: AllModels,
    critical_configuration: Mapping[str, Any],
) -> None:
    """Require every included potential parameter in a nonempty model table."""
    if not len(all_models):
        return
    potential = _mapping(critical_configuration, "potential")
    expected = {
        f"{component_name}.{parameter_name}"
        for component_name, raw_component in potential.items()
        for component in [_require_mapping(raw_component, Path(component_name))]
        if component.get("include", True)
        for parameter_name in _require_mapping(
            component.get("parameters", {}), Path(f"{component_name}.parameters")
        )
    }
    missing = sorted(expected.difference(all_models.table.colnames))
    if missing:
        raise ConfigurationCompatibilityError(
            "AllModels is missing potential-parameter columns required by the "
            f"current configuration: {missing!r}."
        )


def _different_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    """Recursively report differing mapping/list/scalar paths."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(_different_paths(left[key], right[key], path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        differences = []
        pairs = zip(left, right, strict=True)
        for index, (left_value, right_value) in enumerate(pairs):
            differences.extend(
                _different_paths(left_value, right_value, f"{prefix}[{index}]")
            )
        return differences
    return [] if left == right else [prefix]


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return one required mapping from a resolved configuration.

    Delegates the actual shape check to `tnt.config_parsing`'s shared
    `_required_mapping`, translating its generic `TypeError`/`ValueError`
    into this module's own `ConfigurationCompatibilityError` -- the check
    itself isn't duplicated, only the exception type this module commits to
    raising everywhere.
    """
    try:
        return dict(_parse_required_mapping(config, key, "configuration"))
    except (TypeError, ValueError) as error:
        raise ConfigurationCompatibilityError(str(error)) from error


def _require_mapping(value: Any, path: Path) -> ConfigDict:
    """Return a plain mapping or raise a compatibility-specific error."""
    try:
        return dict(_parse_mapping(value, str(path)))
    except TypeError as error:
        raise ConfigurationCompatibilityError(str(error)) from error
