"""Unit systems and unit-aware configuration validation/conversion."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Any

import unxt as u
from unxt import Quantity

from tnt.potential import raw_parameter_dimensions
from tnt.validation import (
    _mapping,
    _optional_mapping,
    _reject_unknown_keys,
    _require_keys,
)

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
    """Return a canonical copy with quantities expressed in internal units.

    This is a runtime/compatibility helper. Configuration preparation validates
    declared quantities with :func:`validate_configuration_quantities` and
    preserves their original ``{value, unit}`` representation.
    """
    resolved = deepcopy(_mapping(config, "configuration"))

    cosmology = _optional_mapping(resolved, "cosmological_parameters", "configuration")
    _normalize_field(
        cosmology,
        "H",
        "inverse_time",
        unit_systems.internal,
        "cosmological_parameters",
    )

    attributes = _optional_mapping(resolved, "system_attributes", "configuration")
    _normalize_field(
        attributes,
        "distance",
        "length",
        unit_systems.internal,
        "system_attributes",
    )

    spatial_binnings = _optional_mapping(resolved, "spatial_binnings", "configuration")
    for binning_name, binning_value in spatial_binnings.items():
        binning_path = f"spatial_binnings.{binning_name}"
        binning = _mapping(binning_value, binning_path)
        for field in ("min_x", "min_y", "x_extent", "y_extent", "PA"):
            _normalize_field(
                binning,
                field,
                "angle",
                unit_systems.internal,
                binning_path,
            )

    potential = _optional_mapping(resolved, "potential", "configuration")
    resolved["potential"] = normalize_potential_settings(
        potential, unit_systems.internal
    )

    kinematic_data = _optional_mapping(resolved, "kinematic_data", "configuration")
    _normalize_kinematics(
        kinematic_data,
        unit_systems.internal,
        "kinematic_data",
    )
    return resolved


def validate_configuration_quantities(config: Mapping[str, Any]) -> None:
    """Validate declared configuration quantities without converting them."""
    config = _mapping(config, "configuration")

    cosmology = _optional_mapping(config, "cosmological_parameters", "configuration")
    _validate_field(cosmology, "H", "inverse_time", "cosmological_parameters")

    attributes = _optional_mapping(config, "system_attributes", "configuration")
    _validate_field(attributes, "distance", "length", "system_attributes")

    potential = _optional_mapping(config, "potential", "configuration")
    for potential_name, potential_value in potential.items():
        potential_path = f"potential.{potential_name}"
        settings = _mapping(potential_value, potential_path)
        dimensions = _potential_parameter_dimensions(settings)
        parameters = settings.get("parameters")
        if parameters is not None:
            _validate_parameter_units(
                _mapping(parameters, f"{potential_path}.parameters"),
                dimensions,
                f"{potential_path}.parameters",
            )

    kinematics = _optional_mapping(config, "kinematic_data", "configuration")
    for name, settings_value in kinematics.items():
        settings_path = f"kinematic_data.{name}"
        settings = _mapping(settings_value, settings_path)
        histogram = settings.get("histogram")
        if histogram is not None:
            histogram = _mapping(histogram, f"{settings_path}.histogram")
            histogram_path = f"{settings_path}.histogram"
            _validate_field(histogram, "width", "speed", histogram_path)
            _validate_field(histogram, "center", "speed", histogram_path)

        errors = settings.get("observational_errors")
        if settings.get("type") == "gauss_hermite" and errors is not None:
            errors = _mapping(errors, f"{settings_path}.observational_errors")
            systematics = errors.get("systematic_uncertainties")
            if systematics is not None:
                systematics_path = (
                    f"{settings_path}.observational_errors.systematic_uncertainties"
                )
                systematics = _mapping(systematics, systematics_path)
                _validate_field(systematics, "v", "speed", systematics_path)
                _validate_field(systematics, "sigma", "speed", systematics_path)


def _potential_parameter_dimensions(settings: Mapping[str, Any]) -> Mapping[str, str]:
    """One potential component's raw parameter dimensions, from `tnt.potential`.

    `tnt.configuration.validation._validate_potential` already validates
    `type`/`parameterization` as strings, and runs before this (see that
    module's comment for why) -- but fall back to no declared dimensions
    (nothing scaled) for a malformed value here anyway, rather than depending
    on that ordering or duplicating its validation.
    """
    potential_type = settings.get("type")
    parameterization = settings.get("parameterization")
    if not (
        isinstance(potential_type, str)
        and (parameterization is None or isinstance(parameterization, str))
    ):
        return {}
    return raw_parameter_dimensions(potential_type, parameterization)


def normalize_potential_settings(
    potential: Mapping[str, Any],
    unit_system: u.AbstractUnitSystem,
) -> ConfigDict:
    """Canonicalize potential settings for physical compatibility comparison."""
    normalized = deepcopy(_mapping(potential, "potential"))
    for potential_name, potential_value in normalized.items():
        potential_path = f"potential.{potential_name}"
        settings = _mapping(potential_value, potential_path)
        dimensions = _potential_parameter_dimensions(settings)
        parameters = settings.get("parameters")
        if parameters is not None:
            _normalize_parameters(
                _mapping(parameters, f"{potential_path}.parameters"),
                dimensions,
                unit_system,
                f"{potential_path}.parameters",
            )
    return normalized


def resolve_cosmological_parameters(
    cosmological_parameters: Mapping[str, Any],
) -> dict[str, Quantity]:
    """Convert a resolved configuration's declared `cosmological_parameters`.

    Preserved as `{value, unit}` by configuration preparation (see this
    module's docstring); consumers that need cosmological context (e.g.
    `tnt.potential`'s NFW `concentration_m200` parameterization, via `H`)
    use the resulting `Quantity`s directly and let `unxt` handle unit
    conversion, rather than assuming a specific declared unit.
    """
    return {
        name: Quantity(declared["value"], declared["unit"])
        for name, declared in cosmological_parameters.items()
    }


def normalize_unitful_value(
    value: Any,
    dimension: str,
    unit_system: u.AbstractUnitSystem,
    path: str,
) -> float:
    """Normalize an explicit ``{value, unit}`` mapping."""
    numeric, source = _declared_quantity(value, dimension, path)
    target = _internal_unit(unit_system, dimension)
    return float(source.to(target, numeric))


def declared_quantity_value(value: Any, dimension: str, path: str) -> float:
    """Validate an explicit quantity and return its unconverted numeric value."""
    numeric, _ = _declared_quantity(value, dimension, path)
    return numeric


def _declared_quantity(value: Any, dimension: str, path: str) -> tuple[float, Any]:
    """Return one validated declared value and unit without converting it."""
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
    return numeric, source


def _normalize_kinematics(
    kinematics: ConfigDict,
    unit_system: u.AbstractUnitSystem,
    path: str,
) -> None:
    for name, settings_value in kinematics.items():
        settings_path = f"{path}.{name}"
        settings = _mapping(settings_value, settings_path)
        histogram = settings.get("histogram")
        if histogram is not None:
            histogram = _mapping(histogram, f"{settings_path}.histogram")
            histogram_path = f"{settings_path}.histogram"
            _normalize_field(histogram, "width", "speed", unit_system, histogram_path)
            _normalize_field(histogram, "center", "speed", unit_system, histogram_path)

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
                    systematics, "v", "speed", unit_system, systematics_path
                )
                _normalize_field(
                    systematics, "sigma", "speed", unit_system, systematics_path
                )


def _normalize_parameters(
    parameters: ConfigDict,
    dimensions: Mapping[str, str],
    unit_system: u.AbstractUnitSystem,
    path: str,
) -> None:
    for name, parameter_value in parameters.items():
        parameter_path = f"{path}.{name}"
        parameter = _mapping(parameter_value, parameter_path)
        dimension = dimensions.get(name)
        if dimension is None or dimension == "dimensionless":
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
        target = _internal_unit(unit_system, dimension)
        factor = float(source.to(target, 1.0))
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
    unit_system: u.AbstractUnitSystem,
    path: str,
) -> None:
    if key in mapping:
        mapping[key] = normalize_unitful_value(
            mapping[key], dimension, unit_system, f"{path}.{key}"
        )


def _validate_field(
    mapping: Mapping[str, Any], key: str, dimension: str, path: str
) -> None:
    if key in mapping:
        declared_quantity_value(mapping[key], dimension, f"{path}.{key}")


def _validate_parameter_units(
    parameters: Mapping[str, Any], dimensions: Mapping[str, str], path: str
) -> None:
    for name, parameter_value in parameters.items():
        parameter_path = f"{path}.{name}"
        parameter = _mapping(parameter_value, parameter_path)
        dimension = dimensions.get(name)
        if dimension is None or dimension == "dimensionless":
            if "unit" in parameter:
                raise ValueError(
                    f"{parameter_path}.unit is not supported because this "
                    "parameter is dimensionless or does not yet have a declared "
                    "dimension."
                )
            continue
        if "unit" not in parameter:
            raise ValueError(f"{parameter_path} is missing required field: unit.")
        _validated_declared_unit(parameter["unit"], dimension, f"{parameter_path}.unit")


def _scale_field(mapping: ConfigDict, key: str, factor: float, path: str) -> None:
    if key not in mapping:
        return
    value = mapping[key]
    if not _is_number(value):
        raise TypeError(f"{path}.{key} must be a number.")
    numeric = float(value)
    _require_finite(numeric, f"{path}.{key}")
    mapping[key] = numeric * factor


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
