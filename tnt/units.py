"""Unit systems and generic unit-aware primitives.

Dimension validation and conversion into orbit-integration units are
deliberately separate concerns. Declared units are validated for
dimensional correctness against `_REFERENCE_UNITS`, a fixed
per-physical-dimension reference that is independent of any run's chosen
unit system. Values keep their declared/source unit through construction; the
only place a shared, explicit unit system is genuinely needed is
`Potential.to_galax` and its callers (see `tnt.potential`), which pass it
straight to `galax`'s potential constructors.

`units.internal` requires exactly `length`, `time`, `mass`, and `angle` --
the dimensions `galax`'s potential types use that `unxt` cannot derive for
you. `unxt` builds `power`, `speed`, `frequency`, ... automatically from
mass/length/time, so derived dimensions must not be declared in
`units.internal`. `angle` is dimensionally independent (`unxt` can't
decompose `rad` into the mechanical bases) and is a real native parameter
dimension for some `galax` types (e.g. `LongMuraliBarPotential.alpha`), so it
must be stated.
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
    _reject_unknown_keys,
    _require_keys,
)

_INTERNAL_DIMENSIONS = ("length", "time", "mass", "angle")
# `power` is an accepted display override because a user may want luminosity
# presented in a specific unit; the internal system derives it from its base
# units (see this module's docstring and `docs/source/units.md`).
_DISPLAY_DIMENSIONS = frozenset((*_INTERNAL_DIMENSIONS, "speed", "power"))
_REFERENCE_UNITS = {
    "dimensionless": "",
    "length": "m",
    "time": "s",
    "mass": "kg",
    "angle": "rad",
    "power": "W",
    "speed": "m / s",
    "frequency": "1 / s",
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
        validate_declared_unit(
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
        validate_declared_unit(
            unit_name,
            dimension,
            f"{path}.display.{dimension}",
        )
        for dimension, unit_name in display_settings.items()
    ]
    display = u.unitsystem(internal, *display_units)
    return UnitSystems(internal=internal, display=display)


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

    Returns a `Quantity` in the unit the configuration declares, leaving any
    conversion to the runtime consumer that needs one.
    """
    numeric, source = _declared_quantity(value, dimension, path)
    return Quantity(numeric, source)


def declared_quantity_value(value: Any, dimension: str, path: str) -> float:
    """Validate an explicit quantity and return its unconverted numeric value."""
    numeric, _ = _declared_quantity(value, dimension, path)
    return numeric


def validate_position_angle(
    angle: Quantity, *, minimum_deg: float, maximum_deg: float, path: str
) -> None:
    """Reject a position angle that isn't a finite scalar in ``[lo, hi)`` degrees.

    A unit-aware peer of `validate_dimension`; the two owning modules supply
    the ``[minimum_deg, maximum_deg)`` domain (`tnt.mge.MAJOR_AXIS_PA_DOMAIN_DEG`,
    `tnt.spatial_binnings`'s y-axis domain). Out-of-domain values are
    rejected, not normalized: TNT preserves configuration declarations
    verbatim and compares them exactly for resume compatibility, so
    physically identical declarations (e.g. 0 and 360 degrees for a directed
    axis) must not both be accepted.
    """
    if getattr(angle, "unit", None) is None or getattr(angle, "shape", None) is None:
        raise ValueError(
            f"{path} must be an angular Quantity, got {type(angle).__name__}."
        )
    if angle.shape != ():
        raise ValueError(
            f"{path} must be a scalar angle, got shape {tuple(angle.shape)}."
        )
    if not angle.unit.is_equivalent(reference_unit("angle")):
        raise ValueError(f"{path} must describe angle, got unit '{angle.unit}'.")
    degrees = float(angle.ustrip("deg"))
    if not math.isfinite(degrees):
        raise ValueError(f"{path} must be finite, got {degrees}.")
    if not minimum_deg <= degrees < maximum_deg:
        raise ValueError(
            f"{path} must be in [{minimum_deg:g}, {maximum_deg:g}) degrees, "
            f"got {degrees:g}."
        )


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
    source = validate_declared_unit(explicit["unit"], dimension, f"{path}.unit")
    return numeric, source


def validate_declared_unit(value: Any, dimension: str, path: str) -> Any:
    """Parse a declared unit *string* and check its dimension, returning the unit.

    The string-input peer of `validate_dimension`, which takes an already
    parsed unit object and returns nothing. Use this at a configuration
    boundary where the unit arrives as a declared string; use
    `validate_dimension` once you hold a unit object.
    """
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
