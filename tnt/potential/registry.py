"""Curated `galax.potential` types and TNT's own potential-component registry.

`_SUPPORTED_GALAX_TYPES` is the source of truth `raw_parameter_dimensions`
reads from for any `type` without a registered `parameterization`; registered
non-native parameterizations (e.g. NFW's `concentration_m200`, see
`tnt.potential.nfw`) have their own hand-declared raw dimensions instead.
TNT's own composite types (the two MGE potentials, more planned alongside
them) are neither -- each such class registers itself via `register_component`
(see that function), reading its raw dimensions directly off the registered
class rather than a separately hand-maintained dict, so there is exactly one
place a new TNT component's `type`/parameter-dimensions/dispatch-target has
to be declared.

Normal `tnt.potential` initialization explicitly imports each concrete
component module so its decorators populate the registry. Configuration
preparation may read this static metadata; registration does not construct
component instances or load scientific input data.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, NamedTuple

from unxt import Quantity

if TYPE_CHECKING:
    from tnt.potential.components import AbstractPotentialComponent

ForwardConverter = Callable[
    [dict[str, Quantity], Mapping[str, Quantity]],
    dict[str, Quantity],
]
"""`(raw, cosmological_parameters) -> native galax constructor kwargs`.

No unit system: each result keeps whatever unit its arithmetic produces --
`to_galax()`'s native constructor converts again regardless (see
`tnt.potential`'s module docstring).
"""

InverseConverter = Callable[
    [dict[str, Quantity], Mapping[str, str], Mapping[str, Quantity]],
    dict[str, Quantity],
]
"""`(native, declared_units, cosmological_parameters) -> raw config parameters`.

`declared_units` maps each raw parameter name to the unit string its
configuration declares, so a reported value comes back in the parameterization
*and* the unit the config actually specifies.
"""


class Parameterization(NamedTuple):
    """A registered non-native parameterization, both directions.

    Bundled together so one can never be registered without the other --
    `AllModels` relies on `invert` existing for every `parameterization` a
    config can actually specify (see `tnt.potential.raw_potential_parameters`).
    """

    convert: ForwardConverter
    """Raw config parameters -> the type's native `galax` constructor kwargs."""
    invert: InverseConverter
    """Native `galax` constructor kwargs -> raw config parameters."""


class NativeParameter(NamedTuple):
    """A native constructor parameter's physical dimension and mass-rescale exponent."""

    dimension: str
    exponent: float


def _mass(exponent: float = 1.0) -> NativeParameter:
    return NativeParameter("mass", exponent)


def _length() -> NativeParameter:
    return NativeParameter("length", 0.0)


def _angle() -> NativeParameter:
    return NativeParameter("angle", 0.0)


def _dimensionless() -> NativeParameter:
    return NativeParameter("dimensionless", 0.0)


# `_SUPPORTED_GALAX_TYPES`: every galax.potential class TNT supports as
# `potential.<name>.type`, and each of its own native constructor
# parameters' physical dimension and mass-rescale exponent (`rescale` holds
# shape fixed while multiplying the total mass by `mass_scale`). This list
# excludes four kinds of galax.potential.AbstractPotential classes:
# (i) abstract/base classes;
# (ii) pre-packaged multi-component bundles with no free parameters of
#      their own (e.g. MilkyWayPotential, LM10Potential -- their
#      disk/bulge/halo/nucleus fields are themselves sub-potentials, not
#      `ParameterField`s; redundant with TNT's own multi-component
#      `potential:` section anyway);
# (iii) wrapper/transform decorators needing a required nested potential
#       object (e.g. TranslatedPotential, FlattenedInThePotential);
# (iv) classes needing a required non-`Quantity` hyperparameter (e.g.
#      MultipolePotential's `l_max: int`).
# Most parameters follow one pattern: mass=1.0 (linear),
# length/angle/dimensionless=0.0 (shape held fixed). The non-obvious
# exponents (`LogarithmicPotential`/`LMJ09LogarithmicPotential`'s `v_c`,
# `HarmonicOscillatorPotential`'s `omega`, `MonariEtAl2016BarPotential`'s
# `v0`/`alpha`/`Omega`) are each verified against galax's own potential
# formula by a dedicated test in tests/unit_tests/test_potential.py.
_SUPPORTED_GALAX_TYPES: dict[str, dict[str, NativeParameter]] = {
    "BurkertPotential": {"m": _mass(), "r_s": _length()},
    "HardCutoffNFWPotential": {"m": _mass(), "r_s": _length(), "r_t": _length()},
    "HarmonicOscillatorPotential": {"omega": NativeParameter("frequency", 0.5)},
    "HernquistPotential": {"m_tot": _mass(), "r_s": _length()},
    "IsochronePotential": {"m_tot": _mass(), "r_s": _length()},
    "JaffePotential": {"m_tot": _mass(), "r_s": _length()},
    "KeplerPotential": {"m_tot": _mass()},
    "KuzminPotential": {"m_tot": _mass(), "r_s": _length()},
    "LMJ09LogarithmicPotential": {
        "v_c": NativeParameter("speed", 0.5),
        "r_s": _length(),
        "q1": _dimensionless(),
        "q2": _dimensionless(),
        "q3": _dimensionless(),
        "phi": _angle(),
    },
    "LeeSutoTriaxialNFWPotential": {
        "m": _mass(),
        "r_s": _length(),
        "a1": _dimensionless(),
        "a2": _dimensionless(),
        "a3": _dimensionless(),
    },
    "LogarithmicPotential": {"v_c": NativeParameter("speed", 0.5), "r_s": _length()},
    "LongMuraliBarPotential": {
        "m_tot": _mass(),
        "a": _length(),
        "b": _length(),
        "c": _length(),
        "alpha": _angle(),
    },
    "MN3ExponentialPotential": {"m_tot": _mass(), "h_R": _length(), "h_z": _length()},
    "MN3Sech2Potential": {"m_tot": _mass(), "h_R": _length(), "h_z": _length()},
    "MiyamotoNagaiPotential": {"m_tot": _mass(), "a": _length(), "b": _length()},
    "MonariEtAl2016BarPotential": {
        "alpha": _dimensionless(),
        "R0": _length(),
        "v0": NativeParameter("speed", 0.5),
        "Rb": _length(),
        "phi_b": _angle(),
        "Omega": NativeParameter("frequency", 0.0),
    },
    "NFWPotential": {"m": _mass(), "r_s": _length()},
    "PlummerPotential": {"m_tot": _mass(), "r_s": _length()},
    "PowerLawCutoffPotential": {
        "m_tot": _mass(),
        "alpha": _dimensionless(),
        "r_c": _length(),
    },
    "SatohPotential": {"m_tot": _mass(), "a": _length(), "b": _length()},
    "StoneOstriker15Potential": {"m_tot": _mass(), "r_c": _length(), "r_h": _length()},
    "TriaxialHernquistPotential": {
        "m_tot": _mass(),
        "r_s": _length(),
        "q1": _dimensionless(),
        "q2": _dimensionless(),
    },
    "TriaxialNFWPotential": {
        "m": _mass(),
        "r_s": _length(),
        "q1": _dimensionless(),
        "q2": _dimensionless(),
    },
    "Vogelsberger08TriaxialNFWPotential": {
        "m": _mass(),
        "r_s": _length(),
        "q1": _dimensionless(),
        "a_r": _dimensionless(),
    },
    "gNFWPotential": {"m": _mass(), "r_s": _length(), "gamma": _dimensionless()},
}


# Shared by both TNT MGE composite types' own `_raw_dimensions` -- the
# global viewing angles both deproject against
# (`tnt.mge.AbstractMGE.deproject_triaxial`), required regardless of light
# vs. mass.
_VIEWING_ANGLES: dict[str, str] = {"theta": "angle", "phi": "angle", "psi": "angle"}

# TNT's own potential-component types (as opposed to native `galax` types,
# `_SUPPORTED_GALAX_TYPES` above), keyed by `_type`. Populated by
# `register_component`, applied directly to each concrete
# `AbstractPotentialComponent` subclass in its own defining module -- this
# dict is the single place both `AbstractPotentialComponent.resolve` (runtime
# dispatch) and `tnt.configuration.validation` (config-prep schema checks)
# read from; there is no second, independently-maintained list of TNT type
# names or dimensions to keep in sync with it.
_COMPONENT_REGISTRY: dict[str, type[AbstractPotentialComponent]] = {}


def register_component(
    cls: type[AbstractPotentialComponent],
) -> type[AbstractPotentialComponent]:
    """Register `cls` -- reading its own `_type`/`_raw_dimensions` -- for dispatch.

    Applied directly to a concrete `AbstractPotentialComponent` subclass's
    definition, e.g. `@register_component` above `class
    TriaxialLightMGEPotential(AbstractPotentialComponent): ...`. Only ever
    reads what the class already declares about itself (`_type`,
    `_raw_dimensions`) -- this decorator's job is registering that
    self-description, not being a second place either value gets typed in.

    Raises:
        ValueError: If another registered class already declared the same
            `_type`.
    """
    type_name = cls._type
    if type_name in _COMPONENT_REGISTRY:
        existing = _COMPONENT_REGISTRY[type_name].__name__
        raise ValueError(
            f"Duplicate potential type {type_name!r} on {existing} and {cls.__name__}."
        )
    _COMPONENT_REGISTRY[type_name] = cls
    return cls

# Raw parameter dimensions for registered non-native parameterizations,
# keyed by (type, parameterization). Populated alongside
# `tnt.potential.components._PARAMETERIZATIONS`.
PARAMETERIZATION_RAW_DIMENSIONS: dict[tuple[str, str], dict[str, str]] = {
    ("NFWPotential", "concentration_m200"): {
        "c": "dimensionless",
        "M_200": "mass",
    },
}


def raw_parameter_dimensions(kind: str, parameterization: str | None) -> dict[str, str]:
    """Each raw config parameter's physical dimension for one `type`/`parameterization`.

    Covers all three sources of truth this module knows about: a registered
    TNT component type's own `_raw_dimensions` (`_COMPONENT_REGISTRY`), a
    registered non-native parameterization's hand-declared raw dimensions,
    or -- the common case -- a curated galax class's native constructor
    kwargs, read directly from `_SUPPORTED_GALAX_TYPES`. Returns `{}` (every
    parameter treated as dimensionless) for anything unrecognized, deferring
    the "is this actually a valid type" question to
    `AbstractPotentialComponent.resolve`.
    """
    if parameterization is not None:
        return PARAMETERIZATION_RAW_DIMENSIONS.get((kind, parameterization), {})
    if kind in _COMPONENT_REGISTRY:
        return _COMPONENT_REGISTRY[kind]._raw_dimensions
    return {
        name: parameter.dimension
        for name, parameter in _SUPPORTED_GALAX_TYPES.get(kind, {}).items()
    }
