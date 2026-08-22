"""Unit systems and unit-aware configuration normalization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Any

import unxt as u

from tnt.config_parsing import (
    _mapping,
    _optional_mapping,
    _reject_unknown_keys,
    _require_keys,
)
from tnt.potential import raw_parameter_dimensions

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

    cosmology = _optional_mapping(resolved, "cosmological_parameters", "configuration")
    _normalize_field(
        cosmology,
        "H0",
        "inverse_time",
        unit_systems,
        "cosmological_parameters",
    )

    attributes = _optional_mapping(resolved, "system_attributes", "configuration")
    _normalize_field(
        attributes, "distance", "length", unit_systems, "system_attributes"
    )

    potential = _optional_mapping(resolved, "potential", "configuration")
    for potential_name, potential_value in potential.items():
        potential_path = f"potential.{potential_name}"
        settings = _mapping(potential_value, potential_path)
        potential_type = settings.get("type")
        parameterization = settings.get("parameterization")
        # `type`/`parameterization` are validated as strings by
        # `_validate_potential`, which runs after normalization -- fall back
        # to no declared dimensions (nothing scaled) for a malformed value
        # here rather than duplicating that validation.
        dimensions = (
            raw_parameter_dimensions(potential_type, parameterization)
            if isinstance(potential_type, str)
            and (parameterization is None or isinstance(parameterization, str))
            else {}
        )
        parameters = settings.get("parameters")
        if parameters is not None:
            _normalize_parameters(
                _mapping(parameters, f"{potential_path}.parameters"),
                dimensions,
                unit_systems,
                f"{potential_path}.parameters",
            )

    kinematic_data = _optional_mapping(resolved, "kinematic_data", "configuration")
    _normalize_kinematics(
        kinematic_data,
        unit_systems,
        "kinematic_data",
    )
    return resolved


def normalize_unitful_value(
    value: Any,
    dimension: str,
    unit_systems: UnitSystems,
    path: str,
) -> float:
    """Normalize an explicit ``{value, unit}`` mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{path} must be a mapping containing value and unit; unitful "
            "configuration values must state their unit explicitly."
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
        dimension = dimensions.get(name)
        if dimension is None:
            if "unit" in parameter:
                raise ValueError(
                    f"{parameter_path}.unit is not supported because this "
                    "parameter is dimensionless or does not yet have a declared "
                    "dimension."
                )
            continue
        if "unit" not in parameter:
            raise ValueError(f"{parameter_path} is missing required field: unit.")
        declared_unit = parameter.pop("unit")

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


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _require_finite(value: float, path: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{path} must be finite.")
