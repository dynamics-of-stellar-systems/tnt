"""Unit systems and unit-aware configuration validation.

Dimension validation and conversion into orbit-integration units are
deliberately separate concerns. Declared units are validated for
dimensional correctness against `_REFERENCE_UNITS`, a fixed
per-physical-dimension reference that is independent of any run's chosen
unit system. Values keep their declared/source unit through construction; the only place
a shared, explicit unit system is genuinely needed is `Potential.to_galax`
and its callers (see `tnt.potential`), which pass it straight to `galax`'s
potential constructors.

`units.internal` requires exactly `length`, `time`, `mass`, and `angle` --
the dimensions `galax`'s potential types use that `unxt` cannot derive for
you. `unxt` builds `power`, `speed`, `frequency`, ... automatically from
mass/length/time, so declaring them is redundant; `angle` is dimensionally
independent (`unxt` can't decompose `rad` into the mechanical bases) and is
a real native parameter dimension for some `galax` types (e.g.
`LongMuraliBarPotential.alpha`), so it must be stated. `power` used to be
required too and was dropped: it is derivable, and requiring it let a
config declare a `power` unit inconsistent with the others that `galax`
silently ignores.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import unxt as u
from unxt import Quantity

from tnt.validation import (
    _mapping,
    _optional_mapping,
    _reject_unknown_keys,
    _require_keys,
)

_INTERNAL_DIMENSIONS = ("length", "time", "mass", "angle")
# `power` stays an accepted *display* override -- a user may still want
# luminosity presented in a specific unit -- even though it's no longer part
# of the required internal set (a unit system auto-derives it from its base
# units; see this module's docstring and `docs/source/units.md`).
_DISPLAY_DIMENSIONS = frozenset((*_INTERNAL_DIMENSIONS, "speed", "power"))
_REFERENCE_UNITS = {
    "length": "m",
    "time": "s",
    "mass": "kg",
    "angle": "rad",
    "power": "W",
    "speed": "m / s",
    "inverse_time": "1 / s",
    "mass_to_light": "kg / W",
    "light_surface_brightness": "W / rad2",
    "mass_surface_density": "kg / rad2",
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
    # Imported lazily: `tnt.mge`/`tnt.kinematics`/`tnt.spatial_binnings` import
    # this module for `validate_dimension`/`declared_quantity`, and
    # `tnt.potential` imports those -- a module-level import here would close
    # that cycle. Config validation, this function's only caller, always runs
    # well after every module is loaded.
    from tnt.potential import raw_parameter_dimensions

    potential_type = settings.get("type")
    parameterization = settings.get("parameterization")
    if not (
        isinstance(potential_type, str)
        and (parameterization is None or isinstance(parameterization, str))
    ):
        return {}
    return raw_parameter_dimensions(potential_type, parameterization)


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


def resolve_system_distance(system_attributes: Mapping[str, Any]) -> Quantity:
    """Convert a resolved configuration's declared `system_attributes.distance`.

    Preserved as `{value, unit}` by configuration preparation, same as
    `cosmological_parameters` (see `resolve_cosmological_parameters`).
    """
    declared = system_attributes["distance"]
    return Quantity(declared["value"], declared["unit"])


def reference_unit(dimension: str) -> Any:
    """The fixed reference unit for `dimension`, from `_REFERENCE_UNITS`.

    A single per-physical-dimension reference, independent of any run's
    chosen unit system (see this module's docstring). Callers that only need
    a pass/fail check should use `validate_dimension` instead; this is for
    the few that need the unit object itself (e.g. `tnt.mge.read_mge`
    inferring an MGE's kind from its `I` column).
    """
    return u.unit(_REFERENCE_UNITS[dimension])


def validate_dimension(unit: Any, dimension: str, path: str) -> None:
    """Raise if `unit` isn't dimensionally consistent with `dimension`.

    Checked against `_REFERENCE_UNITS`, a fixed physical-dimension
    reference -- independent of any run's chosen unit system, since
    dimension validation and conversion into orbit-integration units are
    deliberately separate concerns (see this module's docstring).
    """
    if unit is None:
        raise ValueError(f"{path} must declare a unit.")
    if not unit.is_equivalent(reference_unit(dimension)):
        raise ValueError(f"{path} must describe {dimension.replace('_', ' ')}.")


def declared_quantity(value: Any, dimension: str, path: str) -> Quantity:
    """Validate an explicit ``{value, unit}`` mapping, keeping its declared unit.

    The unconverted counterpart of the old eager normalization: returns a
    `Quantity` in the unit the configuration declares, leaving any
    conversion to whatever consumer later needs one.
    """
    numeric, source = _declared_quantity(value, dimension, path)
    return Quantity(numeric, source)


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
