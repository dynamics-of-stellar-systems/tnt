"""Unit systems and unit-aware configuration normalization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Any

import unxt as u

ConfigDict = dict[str, Any]

_INTERNAL_DIMENSIONS = ("length", "time", "mass", "angle", "power")
_DISPLAY_DIMENSIONS = frozenset((*_INTERNAL_DIMENSIONS, "speed"))
_REFERENCE_UNITS = {
    "length": "m",
    "time": "s",
    "mass": "kg",
    "angle": "rad",
    "power": "W",
    "speed": "m / s",
    "inverse_time": "1 / s",
    "mass_to_light": "kg / W",
}
_COMPONENT_PARAMETER_DIMENSIONS = {
    "plummer": {"m": "mass", "a": "length"},
}
_SYSTEM_PARAMETER_DIMENSIONS = {"ml": "mass_to_light"}


@dataclass(frozen=True)
class UnitSystems:
    """TNT's computational and presentation unit systems."""

    internal: u.AbstractUnitSystem
    display: u.AbstractUnitSystem


def build_unit_systems(settings: Mapping[str, Any]) -> UnitSystems:
    """Validate unit settings and construct TNT's two unit systems."""
    path = "units"
    settings = _mapping(settings, path)
    _reject_unknown_keys(settings, {"internal", "display"}, path)
    _require_keys(settings, {"internal", "display"}, path)

    internal_settings = _mapping(settings["internal"], f"{path}.internal")
    _reject_unknown_keys(
        internal_settings, set(_INTERNAL_DIMENSIONS), f"{path}.internal"
    )
    _require_keys(internal_settings, set(_INTERNAL_DIMENSIONS), f"{path}.internal")
    internal_units = [
        _validated_declared_unit(
            internal_settings[dimension],
            dimension,
            f"{path}.internal.{dimension}",
        )
        for dimension in _INTERNAL_DIMENSIONS
    ]
    internal = u.unitsystem(*internal_units)

    display_settings = _mapping(settings["display"], f"{path}.display")
    _reject_unknown_keys(display_settings, _DISPLAY_DIMENSIONS, f"{path}.display")
    display_units = [
        _validated_declared_unit(
            unit_name,
            dimension,
            f"{path}.display.{dimension}",
        )
        for dimension, unit_name in display_settings.items()
    ]
    display = u.unitsystem(internal, *display_units)
    return UnitSystems(internal=internal, display=display)


def normalize_configuration_quantities(
    config: Mapping[str, Any],
    unit_systems: UnitSystems,
) -> ConfigDict:
    """Return a copy with supported quantities expressed in internal units."""
    resolved = deepcopy(_mapping(config, "configuration"))

    cosmology = _nested_mapping(resolved, "cosmological_parameters", "configuration")
    _normalize_field(
        cosmology,
        "H0",
        "inverse_time",
        unit_systems,
        "cosmological_parameters",
    )

    attributes = _nested_mapping(resolved, "system_attributes", "configuration")
    _normalize_field(
        attributes, "distance", "length", unit_systems, "system_attributes"
    )

    components = _nested_mapping(resolved, "system_components", "configuration")
    for component_name, component_value in components.items():
        component_path = f"system_components.{component_name}"
        component = _mapping(component_value, component_path)
        component_type = component.get("type")
        dimensions = _COMPONENT_PARAMETER_DIMENSIONS.get(component_type, {})
        parameters = component.get("parameters")
        if parameters is not None:
            _normalize_parameters(
                _mapping(parameters, f"{component_path}.parameters"),
                dimensions,
                unit_systems,
                f"{component_path}.parameters",
            )

        kinematics = component.get("kinematics")
        if kinematics is not None:
            _normalize_kinematics(
                _mapping(kinematics, f"{component_path}.kinematics"),
                unit_systems,
                f"{component_path}.kinematics",
            )

    system_parameters = _nested_mapping(resolved, "system_parameters", "configuration")
    _normalize_parameters(
        system_parameters,
        _SYSTEM_PARAMETER_DIMENSIONS,
        unit_systems,
        "system_parameters",
    )
    return resolved


def normalize_unitful_value(
    value: Any,
    dimension: str,
    unit_systems: UnitSystems,
    path: str,
) -> float:
    """Normalize a bare number or ``{value, unit}`` mapping."""
    if _is_number(value):
        numeric = float(value)
        _require_finite(numeric, path)
        return numeric
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{path} must be a number in TNT internal units or a mapping "
            "containing value and unit."
        )

    explicit = _mapping(value, path)
    _reject_unknown_keys(explicit, {"value", "unit"}, path)
    _require_keys(explicit, {"value", "unit"}, path)
    numeric_value = explicit["value"]
    if not _is_number(numeric_value):
        raise TypeError(f"{path}.value must be a number.")
    numeric = float(numeric_value)
    _require_finite(numeric, f"{path}.value")
    source = _validated_declared_unit(explicit["unit"], dimension, f"{path}.unit")
    target = _internal_unit(unit_systems.internal, dimension)
    return float(source.to(target, numeric))


def _normalize_kinematics(
    kinematics: ConfigDict,
    unit_systems: UnitSystems,
    path: str,
) -> None:
    for name, settings_value in kinematics.items():
        settings_path = f"{path}.{name}"
        settings = _mapping(settings_value, settings_path)
        histogram = settings.get("histogram")
        if histogram is not None:
            histogram = _mapping(histogram, f"{settings_path}.histogram")
            histogram_path = f"{settings_path}.histogram"
            _normalize_field(histogram, "width", "speed", unit_systems, histogram_path)
            _normalize_field(histogram, "center", "speed", unit_systems, histogram_path)

        errors = settings.get("observational_errors")
        if settings.get("type") == "gauss_hermite" and errors is not None:
            errors = _mapping(errors, f"{settings_path}.observational_errors")
            systematics = errors.get("systematic_uncertainties")
            if systematics is not None:
                systematics = _mapping(
                    systematics,
                    f"{settings_path}.observational_errors.systematic_uncertainties",
                )
                systematics_path = (
                    f"{settings_path}.observational_errors.systematic_uncertainties"
                )
                _normalize_field(
                    systematics, "v", "speed", unit_systems, systematics_path
                )
                _normalize_field(
                    systematics, "sigma", "speed", unit_systems, systematics_path
                )


def _normalize_parameters(
    parameters: ConfigDict,
    dimensions: Mapping[str, str],
    unit_systems: UnitSystems,
    path: str,
) -> None:
    for name, parameter_value in parameters.items():
        parameter_path = f"{path}.{name}"
        parameter = _mapping(parameter_value, parameter_path)
        declared_unit = parameter.pop("unit", None)
        dimension = dimensions.get(name)
        if declared_unit is None:
            continue
        if dimension is None:
            raise ValueError(
                f"{parameter_path}.unit is not supported because this parameter "
                "is dimensionless or does not yet have a declared dimension."
            )

        source = _validated_declared_unit(
            declared_unit, dimension, f"{parameter_path}.unit"
        )
        target = _internal_unit(unit_systems.internal, dimension)
        factor = float(source.to(target, 1.0))
        logarithmic = parameter.get("logarithmic", False)
        if not isinstance(logarithmic, bool):
            raise TypeError(f"{parameter_path}.logarithmic must be a boolean.")

        if logarithmic:
            offset = math.log10(factor)
            _shift_log_value(parameter, "value", offset, parameter_path)
            generator = parameter.get("generator_settings")
            if generator is not None:
                generator = _mapping(generator, f"{parameter_path}.generator_settings")
                for key in ("lower_bound", "upper_bound"):
                    _shift_log_value(
                        generator,
                        key,
                        offset,
                        f"{parameter_path}.generator_settings",
                    )
        else:
            _scale_field(parameter, "value", factor, parameter_path)
            generator = parameter.get("generator_settings")
            if generator is not None:
                generator = _mapping(generator, f"{parameter_path}.generator_settings")
                for key in ("lower_bound", "upper_bound", "step", "minimum_step"):
                    _scale_field(
                        generator,
                        key,
                        factor,
                        f"{parameter_path}.generator_settings",
                    )


def _normalize_field(
    mapping: ConfigDict,
    key: str,
    dimension: str,
    unit_systems: UnitSystems,
    path: str,
) -> None:
    if key in mapping:
        mapping[key] = normalize_unitful_value(
            mapping[key], dimension, unit_systems, f"{path}.{key}"
        )


def _scale_field(mapping: ConfigDict, key: str, factor: float, path: str) -> None:
    if key not in mapping:
        return
    value = mapping[key]
    if not _is_number(value):
        raise TypeError(f"{path}.{key} must be a number.")
    numeric = float(value)
    _require_finite(numeric, f"{path}.{key}")
    mapping[key] = numeric * factor


def _shift_log_value(mapping: ConfigDict, key: str, offset: float, path: str) -> None:
    if key not in mapping:
        return
    value = mapping[key]
    if not _is_number(value):
        raise TypeError(f"{path}.{key} must be a number.")
    numeric = float(value)
    _require_finite(numeric, f"{path}.{key}")
    mapping[key] = numeric + offset


def _internal_unit(
    unit_system: u.AbstractUnitSystem,
    dimension: str,
) -> Any:
    if dimension == "inverse_time":
        return 1 / unit_system[u.dimension("time")]
    if dimension == "mass_to_light":
        return unit_system[u.dimension("mass")] / unit_system[u.dimension("power")]
    return unit_system[u.dimension(dimension)]


def _validated_declared_unit(value: Any, dimension: str, path: str) -> Any:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty unit string.")
    try:
        parsed = u.unit(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} contains invalid unit {value!r}.") from error
    reference = u.unit(_REFERENCE_UNITS[dimension])
    if not parsed.is_equivalent(reference):
        raise ValueError(f"{path} must describe {dimension.replace('_', ' ')}.")
    return parsed


def _nested_mapping(mapping: ConfigDict, key: str, parent: str) -> ConfigDict:
    return _mapping(mapping.get(key, {}), f"{parent}.{key}")


def _mapping(value: Any, path: str) -> ConfigDict:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a YAML mapping.")
    return value


def _reject_unknown_keys(
    mapping: Mapping[str, Any], allowed: set[str] | frozenset[str], path: str
) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ValueError(f"{path} contains unknown field(s): {', '.join(unknown)}.")


def _require_keys(mapping: Mapping[str, Any], required: set[str], path: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"{path} is missing required field(s): {', '.join(missing)}.")


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _require_finite(value: float, path: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{path} must be finite.")
