"""Curated `galax.potential` types and their native parameters' dimensions.

`_SUPPORTED_GALAX_TYPES` is the source of truth `raw_parameter_dimensions`
reads from for any `type` without a registered `parameterization`; the two
TNT MGE composite types and registered non-native parameterizations (e.g.
NFW's `concentration_m200`, see `tnt.potential.nfw`) each have their own
hand-declared raw dimensions instead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import NamedTuple

from unxt import AbstractUnitSystem, Quantity

ParameterizationConverter = Callable[
    [dict[str, Quantity], AbstractUnitSystem, Mapping[str, Quantity]],
    dict[str, Quantity],
]


class Parameterization(NamedTuple):
    """A registered non-native parameterization, both directions.

    Bundled together so one can never be registered without the other --
    `AllModels` relies on `invert` existing for every `parameterization` a
    config can actually specify (see `tnt.potential.raw_potential_parameters`).
    """

    convert: ParameterizationConverter
    """Raw config parameters -> the type's native `galax` constructor kwargs."""
    invert: ParameterizationConverter
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


# Hand-declared raw parameter dimensions for the two TNT MGE composite
# types. `TriaxialMassMGEPotential`'s own mass parameter is `mge_mass_scale`, a
# pure multiplicative scale factor -- dimensionless, but still given an
# explicit entry (not omitted) so `_validate_parameter_units` can reject a
# stray declared `unit` on it with a specific message. `ml` has no entry
# under `TriaxialMassMGEPotential` -- it's invalid there, not just dimensionless;
# rejected directly by `tnt.configuration.validation`'s `_validate_potential`,
# which runs before this module's dimension check and derives its own
# required/forbidden-parameter check from these same keys, so a config
# mistakenly declaring `ml` there gets that specific "invalid field for
# TriaxialMassMGEPotential" error rather than a generic dimension one.
_VIEWING_ANGLES: dict[str, str] = {"theta": "angle", "phi": "angle", "psi": "angle"}
_MGE_RAW_DIMENSIONS: dict[str, dict[str, str]] = {
    "TriaxialLightMGEPotential": {"ml": "mass_to_light", **_VIEWING_ANGLES},
    "TriaxialMassMGEPotential": {"mge_mass_scale": "dimensionless", **_VIEWING_ANGLES},
}

# The set of TNT-specific (non-native-galax) potential-component type names,
# derived from the same dict above rather than hand-maintained separately --
# `tnt.configuration.validation` imports this directly instead of keeping its
# own copy, since this module has no `galax`/`equinox` imports and stays safe
# for that "no scientific object construction" module to depend on.
MGE_POTENTIAL_TYPES: frozenset[str] = frozenset(_MGE_RAW_DIMENSIONS)

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

    Covers all three sources of truth in this module: a TNT MGE composite
    type's own hand-declared dimensions, a registered non-native
    parameterization's hand-declared raw dimensions, or -- the common case
    -- a curated galax class's native constructor kwargs, read directly from
    `_SUPPORTED_GALAX_TYPES`. Returns `{}` (every parameter treated as
    dimensionless) for anything unrecognized, deferring the "is this
    actually a valid type" question to `AbstractPotentialComponent.resolve`.
    """
    if parameterization is not None:
        return PARAMETERIZATION_RAW_DIMENSIONS.get((kind, parameterization), {})
    if kind in _MGE_RAW_DIMENSIONS:
        return _MGE_RAW_DIMENSIONS[kind]
    return {
        name: parameter.dimension
        for name, parameter in _SUPPORTED_GALAX_TYPES.get(kind, {}).items()
    }
