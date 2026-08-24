"""Galactic potentials, assembled from named, `galax`-backed components.

`potential.<name>.type` names either a curated `galax.potential` class (see
`_SUPPORTED_GALAX_TYPES`, e.g. `"NFWPotential"`, `"PlummerPotential"`) or one
of two TNT-specific MGE composite potentials, `"triaxial_light_mge"`/
`"triaxial_mass_mge"`, each built from a named MGE, and pairs every
included class with each native parameter's mass-rescale exponent.
`parameterization` is a separate, optional concern: when omitted,
`parameters` use the resolved type's own native constructor kwargs, with
physical dimensions read directly from `_SUPPORTED_GALAX_TYPES` (see
`raw_parameter_dimensions`). When given, `parameterization` names a
registered conversion from some other raw parameter convention into those
same native fields.

This module is filled in incrementally, one object at a time -- the same
approach already used for `ProjectedBinning`. `Potential.generate_orbit_library`
and the two MGE composite components' `to_galax` remain `NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar, NamedTuple, Self

import equinox as eqx
import galax.potential
import jax.numpy as jnp
import unxt as u
from unxt import AbstractUnitSystem, Quantity

from tnt.config_parsing import (
    _mapping,
    _number,
    _required,
    _required_mapping,
    _required_string,
    _resolve_typed_reference,
    _string,
)
from tnt.mge import LightMGE, MassMGE
from tnt.orbit_library import AbstractOrbitDithering, AbstractOrbitSampler, OrbitLibrary

ParameterizationConverter = Callable[
    [dict[str, Quantity], AbstractUnitSystem, Mapping[str, Quantity]],
    dict[str, Quantity],
]


class Parameterization(NamedTuple):
    """A registered non-native parameterization, both directions.

    Bundled together so one can never be registered without the other --
    `AllModels` relies on `invert` existing for every `parameterization` a
    config can actually specify (see `raw_potential_parameters`).
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
# types. `triaxial_mass_mge`'s own mass parameter is `mge_mass_scale`, a
# pure multiplicative scale factor with no physical unit of its own
# (dimensionless, so it has no entry here); it still declares `ml` so that
# a config mistakenly setting one on a mass-MGE component gets normalized
# as mass_to_light, letting `_validate_potential` raise the more specific
# "ml is invalid for a mass MGE potential" error.
_MGE_RAW_DIMENSIONS: dict[str, dict[str, str]] = {
    "triaxial_light_mge": {"ml": "mass_to_light"},
    "triaxial_mass_mge": {"ml": "mass_to_light"},
}

# Raw parameter dimensions for registered non-native parameterizations,
# keyed by (type, parameterization). Populated alongside `_PARAMETERIZATIONS`
# below.
PARAMETERIZATION_RAW_DIMENSIONS: dict[tuple[str, str], dict[str, str]] = {
    ("NFWPotential", "concentration_m200"): {"M_200": "mass"},  # c dimensionless
}

# Newton's gravitational constant, for parameterizations that need it (e.g.
# critical density). Not from galax's own `default_constants`, to keep this
# module's physics self-contained and independently verifiable rather than
# reaching into galax's private `_src` internals.
_G = Quantity(6.6743e-11, "m3 / (kg s2)")


def raw_parameter_dimensions(kind: str, parameterization: str | None) -> dict[str, str]:
    """Each raw config parameter's physical dimension for one `type`/`parameterization`.

    Covers all three sources of truth in this module: a TNT MGE composite
    type's own hand-declared dimensions, a registered non-native
    parameterization's hand-declared raw dimensions, or -- the common case
    -- a curated galax class's native constructor kwargs, read directly from
    `_SUPPORTED_GALAX_TYPES`. Returns `{}` (every parameter treated as
    dimensionless) for anything unrecognized, deferring the "is this
    actually a valid type" question to `AbstractPotentialComponent.from_settings`.
    """
    if parameterization is not None:
        return PARAMETERIZATION_RAW_DIMENSIONS.get((kind, parameterization), {})
    if kind in _MGE_RAW_DIMENSIONS:
        return _MGE_RAW_DIMENSIONS[kind]
    return {
        name: parameter.dimension
        for name, parameter in _SUPPORTED_GALAX_TYPES.get(kind, {}).items()
    }


def _resolve_unit(unit_system: AbstractUnitSystem, dimension: str) -> Any:
    """Resolve one of `raw_parameter_dimensions`' dimension names to a concrete unit.

    Mirrors `tnt.units._internal_unit`'s handling of `"mass_to_light"` -- a
    TNT pseudo-dimension (not a real astropy physical type) that appears
    here via the two MGE composite types' `ml` parameter.
    """
    if dimension == "mass_to_light":
        return unit_system[u.dimension("mass")] / unit_system[u.dimension("power")]
    return unit_system[u.dimension(dimension)]


def _nfw_concentration_m200(
    raw: dict[str, Quantity],
    unit_system: AbstractUnitSystem,
    cosmological_parameters: Mapping[str, Quantity],
) -> dict[str, Quantity]:
    """Convert NFW's `(c, M_200)` parameterization to native `(m, r_s)`.

    `M_200` uses the critical-density convention (M_200c): the mass
    enclosed within the radius `r_200` at which the mean density equals
    `200 * rho_crit`, where `rho_crit = 3 H0^2 / (8 pi G)`. Concentration is
    `c = r_200 / r_s`. Both `r_s` and the native characteristic mass `m`
    follow from `galax.potential.NFWPotential`'s own enclosed-mass formula,
    `M(<r) = m * (ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s))`, evaluated at
    `r = r_200` -- verified directly against galax's own `mass_enclosed` to
    float32 precision, and that the resulting `r_200` truly encloses a mean
    density of exactly `200 * rho_crit`.
    """
    c = raw["c"]
    m200 = raw["M_200"]
    h0 = cosmological_parameters["H0"]

    rho_crit = 3 * h0**2 / (8 * jnp.pi * _G)
    r200 = (3 * m200 / (4 * jnp.pi * 200 * rho_crit)) ** (1 / 3)
    r_s = r200 / c
    m = m200 / _nfw_g(c.ustrip(""))
    return {
        "m": m.to(unit_system[u.dimension("mass")]),
        "r_s": r_s.to(unit_system[u.dimension("length")]),
    }


def _nfw_g(c: Any) -> Any:
    """`ln(1 + c) - c / (1 + c)`, NFW's enclosed-mass shape function."""
    return jnp.log(1 + c) - c / (1 + c)


def _solve_nfw_concentration(target: Any) -> Any:
    """Solve `c**3 / _nfw_g(c) == target` for `c > 0`.

    No closed form. `h(c) = c**3 / _nfw_g(c)` is strictly monotonically
    increasing for `c > 0` (verified numerically across many orders of
    magnitude of `c`), so a fixed-iteration bisection over a wide,
    unit-independent bracket (`c` is dimensionless -- realistic halo
    concentrations are always well inside `[1e-6, 1e6]`) converges reliably
    regardless of `target`'s scale.
    """

    def h(c: Any) -> Any:
        return c**3 / _nfw_g(c)

    lower, upper = jnp.asarray(1e-6), jnp.asarray(1e6)
    for _ in range(80):
        mid = 0.5 * (lower + upper)
        too_low = h(mid) < target
        lower = jnp.where(too_low, mid, lower)
        upper = jnp.where(too_low, upper, mid)
    return 0.5 * (lower + upper)


def _nfw_concentration_m200_inverse(
    native: dict[str, Quantity],
    unit_system: AbstractUnitSystem,
    cosmological_parameters: Mapping[str, Quantity],
) -> dict[str, Quantity]:
    """Convert NFW's native `(m, r_s)` back to `(c, M_200)`.

    The inverse of `_nfw_concentration_m200`. Substituting
    `r_200 = c * r_s` into that function's `r_200`/`m` relations leaves one
    equation in `c` alone, `c**3 / _nfw_g(c) = m / ((4 pi 200 rho_crit / 3) * r_s**3)`,
    solved numerically by `_solve_nfw_concentration` since it has no closed
    form. `M_200` then follows directly from `c` via the forward relation
    `m = M_200 / _nfw_g(c)`.

    This matters after `GalaxPotentialComponent.rescale()`, which scales
    `m` while holding `r_s` fixed (see `_SUPPORTED_GALAX_TYPES`): that is
    *not* the same as holding `c` fixed and scaling `M_200`, so the rescaled
    `(c, M_200)` genuinely differs from the original and must be recomputed
    here, not just carried through unchanged.
    """
    m = native["m"]
    r_s = native["r_s"]
    h0 = cosmological_parameters["H0"]

    rho_crit = 3 * h0**2 / (8 * jnp.pi * _G)
    target = (m / (4 * jnp.pi * 200 * rho_crit / 3 * r_s**3)).ustrip("")
    c = _solve_nfw_concentration(target)
    m200 = m * _nfw_g(c)
    return {"c": Quantity(c, ""), "M_200": m200.to(unit_system[u.dimension("mass")])}


_PARAMETERIZATIONS: dict[str, dict[str, Parameterization]] = {
    "NFWPotential": {
        "concentration_m200": Parameterization(
            _nfw_concentration_m200, _nfw_concentration_m200_inverse
        ),
    },
}


class AbstractPotentialComponent(eqx.Module):
    """One named term of the total potential (e.g. a halo, a light MGE).

    `rescale` holds shape fixed while re-normalizing a component's overall
    mass: for `GalaxPotentialComponent`, every native parameter scales by
    its own exponent, curated per class (see `_SUPPORTED_GALAX_TYPES`) --
    e.g. `LogarithmicPotential`'s `v_c` scales alongside a true mass
    parameter like Plummer's `m_tot`, each by its own confirmed exponent.
    For the two MGE composite types,
    `rescale` multiplies their one TNT-defined mass-normalization parameter
    (`ml`/`mge_mass_scale`) directly. `parameters` always holds canonical,
    parameterization-independent fields: for a native galax type these are
    exactly that class's own constructor kwarg names; for the two MGE
    composite types, TNT's own `ml`/`mge_mass_scale`.
    """

    parameters: dict[str, Quantity]

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        mges: Mapping[str, LightMGE | MassMGE],
        unit_system: AbstractUnitSystem,
        cosmological_parameters: Mapping[str, Quantity],
        *,
        path: str = "potential.<component>",
    ) -> AbstractPotentialComponent:
        """Build one potential component from its resolved config entry.

        Args:
            settings: One resolved `potential.<name>` entry: `type`, an
                optional `parameterization`, and `parameters`.
            mges: Named MGEs, e.g. from `tnt.mge.build_mges` -- used only by
                the two MGE composite types.
            unit_system: The unit system `parameters` values are already
                expressed in (post `tnt.units` normalization) and that any
                resulting `galax` potential will be constructed with.
            cosmological_parameters: A resolved configuration's
                `cosmological_parameters` section -- used only by
                parameterizations that need it, e.g. NFW's
                `concentration_m200`.
            path: This entry's location in the configuration, used in error
                messages.

        Returns:
            The resolved component, either a `GalaxPotentialComponent` (for
            any real `galax.potential` class name) or one of the two MGE
            composite components.

        Raises:
            ValueError: If `type` names neither a supported
                `galax.potential` class (see `_SUPPORTED_GALAX_TYPES`) nor
                one of the two MGE composite type names.
            NotImplementedError: If an explicit `parameterization` isn't
                registered for this `type`.
        """
        kind = _required_string(settings, "type", path)
        component_cls = _MGE_COMPOSITE_CLASSES.get(kind, GalaxPotentialComponent)
        unsupported = (
            component_cls is GalaxPotentialComponent
            and kind not in _SUPPORTED_GALAX_TYPES
        )
        if unsupported:
            allowed = ", ".join(sorted(_MGE_COMPOSITE_CLASSES))
            raise ValueError(
                f"Unsupported {path}.type {kind!r}; expected a supported "
                "galax.potential class name (see "
                "tnt.potential._SUPPORTED_GALAX_TYPES) or one of: "
                f"{allowed}."
            )

        parameterization_name = settings.get("parameterization")
        convert: ParameterizationConverter | None = None
        if parameterization_name is not None:
            _string(parameterization_name, f"{path}.parameterization")
            converters = _PARAMETERIZATIONS.get(kind, {})
            try:
                convert = converters[parameterization_name].convert
            except KeyError as error:
                allowed = ", ".join(sorted(converters)) or "(none implemented yet)"
                raise NotImplementedError(
                    f"{path}.parameterization {parameterization_name!r} is not "
                    f"implemented for type {kind!r}; implemented: {allowed}."
                ) from error

        raw_dimensions = raw_parameter_dimensions(kind, parameterization_name)
        raw_settings = _required_mapping(settings, "parameters", path)
        raw: dict[str, Quantity] = {}
        for name, parameter_value in raw_settings.items():
            parameter_path = f"{path}.parameters.{name}"
            parameter = _mapping(parameter_value, parameter_path)
            value = _number(
                _required(parameter, "value", parameter_path), f"{parameter_path}.value"
            )
            dimension = raw_dimensions.get(name)
            unit = (
                _resolve_unit(unit_system, dimension) if dimension is not None else ""
            )
            raw[name] = Quantity(value, unit)

        canonical = (
            convert(raw, unit_system, cosmological_parameters)
            if convert is not None
            else raw
        )
        extra = component_cls._extra_fields(kind, settings, mges, path=path)
        return component_cls(parameters=canonical, **extra)

    @classmethod
    def _extra_fields(
        cls,
        kind: str,
        settings: Mapping[str, Any],
        mges: Mapping[str, LightMGE | MassMGE],
        *,
        path: str,
    ) -> dict[str, Any]:
        """Extra constructor kwargs beyond `parameters` (e.g. `galax_type`, `mge`)."""
        return {}

    def to_galax(
        self, unit_system: AbstractUnitSystem
    ) -> galax.potential.AbstractPotential:
        """This component as a `galax` potential."""
        raise NotImplementedError

    def rescale(self, mass_scale: float) -> Self:
        """Re-normalize this component to a different overall mass scale.

        Used for cheap re-exploration of nearby mass scales without
        re-integrating orbits (`parameter_space_settings.potential_rescalings`
        via `ModelIterator`). This applies even when that parameter is
        `fixed`: `fixed` only stops `ParameterGenerator` from proposing
        independent values for it across shape points -- it doesn't exempt
        it from this uniform rescale, which every component must undergo
        together, or the potential's shape (and the orbit library
        integrated in it) would silently no longer match.
        """
        raise NotImplementedError

    def raw_parameters(
        self,
        parameterization: str | None,
        unit_system: AbstractUnitSystem,
        cosmological_parameters: Mapping[str, Quantity],
    ) -> dict[str, Quantity]:
        """This component's parameters in the resolved config's own parameterization.

        The inverse of `from_settings`'s conversion, so `AllModels` can
        report every component the way its configuration actually
        specified it, regardless of `rescale`. Identity by default:
        `parameters` already *is* the raw, parameterization-independent
        representation for anything without a registered non-native
        parameterization -- both MGE composite types (which don't support
        one at all) and a native `galax` type with `parameterization`
        omitted.
        """
        del parameterization, unit_system, cosmological_parameters
        return self.parameters


class GalaxPotentialComponent(AbstractPotentialComponent):
    """A component built directly from a named `galax.potential` class."""

    galax_type: str

    @classmethod
    def _extra_fields(
        cls,
        kind: str,
        settings: Mapping[str, Any],
        mges: Mapping[str, LightMGE | MassMGE],
        *,
        path: str,
    ) -> dict[str, Any]:
        del settings, mges, path
        return {"galax_type": kind}

    def to_galax(
        self, unit_system: AbstractUnitSystem
    ) -> galax.potential.AbstractPotential:
        potential_cls = getattr(galax.potential, self.galax_type)
        return potential_cls(**self.parameters, units=unit_system)

    def rescale(self, mass_scale: float) -> Self:
        try:
            exponents = _SUPPORTED_GALAX_TYPES[self.galax_type]
        except KeyError as error:
            raise NotImplementedError(
                f"{self.galax_type} is not a supported potential type (see "
                "tnt.potential._SUPPORTED_GALAX_TYPES); rescale() doesn't know "
                "its parameters' mass-rescale exponents."
            ) from error
        rescaled = {}
        for name, value in self.parameters.items():
            try:
                exponent = exponents[name].exponent
            except KeyError as error:
                raise NotImplementedError(
                    f"{self.galax_type}.{name} has no confirmed mass-rescale exponent."
                ) from error
            rescaled[name] = value * mass_scale**exponent
        return eqx.tree_at(lambda c: c.parameters, self, rescaled)

    def raw_parameters(
        self,
        parameterization: str | None,
        unit_system: AbstractUnitSystem,
        cosmological_parameters: Mapping[str, Quantity],
    ) -> dict[str, Quantity]:
        if parameterization is None:
            return self.parameters
        invert = _PARAMETERIZATIONS[self.galax_type][parameterization].invert
        return invert(self.parameters, unit_system, cosmological_parameters)


class TriaxialLightMGEComponent(AbstractPotentialComponent):
    """A triaxial potential from a light MGE, via its `ml` parameter.

    Not yet implemented: no native `galax.potential` class exists for a
    sum-of-triaxial-Gaussians potential; building one needs a custom
    `galax.potential.AbstractPotential` subclass, the same difficulty tier
    as `AbstractMGE.get_projected_mass`'s from-scratch Cappellari-2002
    implementation -- a separate, larger effort.
    """

    _type: ClassVar[str] = "triaxial_light_mge"
    mge: LightMGE

    @classmethod
    def _extra_fields(
        cls,
        kind: str,
        settings: Mapping[str, Any],
        mges: Mapping[str, LightMGE | MassMGE],
        *,
        path: str,
    ) -> dict[str, Any]:
        del kind
        mge_name = _required_string(settings, "mge", path)
        return {
            "mge": _resolve_typed_reference(
                mges, mge_name, f"{path}.mge", "MGEs", LightMGE
            )
        }

    def to_galax(
        self, unit_system: AbstractUnitSystem
    ) -> galax.potential.AbstractPotential:
        raise NotImplementedError

    def rescale(self, mass_scale: float) -> Self:
        rescaled = dict(self.parameters)
        rescaled["ml"] = rescaled["ml"] * mass_scale
        return eqx.tree_at(lambda c: c.parameters, self, rescaled)


class TriaxialMassMGEComponent(AbstractPotentialComponent):
    """A triaxial potential from an already-mass-calibrated MGE.

    `mge_mass_scale` is the analogue of a light MGE's `ml` for a component
    whose shape template is already in mass units: a normalization on top
    of an otherwise-fixed mass map, typically left `fixed` (see `rescale`'s
    docstring for why it can still move regardless). Not yet implemented,
    for the same reason as `TriaxialLightMGEComponent`.
    """

    _type: ClassVar[str] = "triaxial_mass_mge"
    mge: MassMGE

    @classmethod
    def _extra_fields(
        cls,
        kind: str,
        settings: Mapping[str, Any],
        mges: Mapping[str, LightMGE | MassMGE],
        *,
        path: str,
    ) -> dict[str, Any]:
        del kind
        mge_name = _required_string(settings, "mge", path)
        return {
            "mge": _resolve_typed_reference(
                mges, mge_name, f"{path}.mge", "MGEs", MassMGE
            )
        }

    def to_galax(
        self, unit_system: AbstractUnitSystem
    ) -> galax.potential.AbstractPotential:
        raise NotImplementedError

    def rescale(self, mass_scale: float) -> Self:
        rescaled = dict(self.parameters)
        rescaled["mge_mass_scale"] = rescaled["mge_mass_scale"] * mass_scale
        return eqx.tree_at(lambda c: c.parameters, self, rescaled)


def _mge_composite_registry() -> dict[str, type[AbstractPotentialComponent]]:
    return {
        TriaxialLightMGEComponent._type: TriaxialLightMGEComponent,
        TriaxialMassMGEComponent._type: TriaxialMassMGEComponent,
    }


_MGE_COMPOSITE_CLASSES = _mge_composite_registry()


class Potential(eqx.Module):
    """The sum of included potential components, at one point in parameter space."""

    components: dict[str, AbstractPotentialComponent]

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Mapping[str, Any]],
        mges: Mapping[str, LightMGE | MassMGE],
        unit_system: AbstractUnitSystem,
        cosmological_parameters: Mapping[str, Quantity],
    ) -> Self:
        """Build a `Potential` from a resolved configuration's `potential` section."""
        components: dict[str, AbstractPotentialComponent] = {}
        for name, component_value in settings.items():
            path = f"potential.{name}"
            component_settings = _mapping(component_value, path)
            if not component_settings.get("include", True):
                continue
            components[name] = AbstractPotentialComponent.from_settings(
                component_settings,
                mges,
                unit_system,
                cosmological_parameters,
                path=path,
            )
        if not components:
            raise ValueError("potential must contain at least one included component.")
        return cls(components=components)

    def to_galax(
        self, unit_system: AbstractUnitSystem
    ) -> galax.potential.AbstractPotential:
        """This potential's included components, composed into one `galax` potential."""
        return galax.potential.CompositePotential(
            {
                name: component.to_galax(unit_system)
                for name, component in self.components.items()
            },
            units=unit_system,
        )

    def generate_orbit_library(
        self,
        orbit_library_settings: Mapping[str, Any],
        orbit_sampler: AbstractOrbitSampler,
        orbit_dithering: AbstractOrbitDithering,
    ) -> OrbitLibrary:
        """Integrate this potential's `OrbitLibrary`.

        Args:
            orbit_library_settings: A resolved configuration's
                `orbit_library_settings` section.
            orbit_sampler: Determines the number of orbit bundles
                (`OrbitLibrary.orbits`' leading axis), e.g. from
                `tnt.orbit_library.build_orbit_sampler`.
            orbit_dithering: Determines the number of dithered orbits per
                bundle (`OrbitLibrary.orbits`' second axis), e.g. from
                `tnt.orbit_library.build_orbit_dithering`.
        """
        raise NotImplementedError

    def rescale(self, mass_scale: float) -> Self:
        """Multiply every component's mass parameter by `mass_scale`.

        See `AbstractPotentialComponent.rescale` for why every component,
        `fixed` or not, must be rescaled together.
        """
        return type(self)(
            components={
                name: component.rescale(mass_scale)
                for name, component in self.components.items()
            }
        )


def build_potential(
    potential: Mapping[str, Mapping[str, Any]],
    mges: Mapping[str, LightMGE | MassMGE],
    unit_system: AbstractUnitSystem,
    cosmological_parameters: Mapping[str, Quantity],
) -> Potential:
    """Build the `Potential` from a resolved configuration's `potential` section.

    Args:
        potential: A resolved configuration's `potential` section.
        mges: Named MGEs, e.g. from `tnt.mge.build_mges`.
        unit_system: The unit system `potential`'s parameter values are
            already expressed in, and that the resulting `galax` potential
            will be constructed with.
        cosmological_parameters: A resolved configuration's
            `cosmological_parameters` section -- used only by
            parameterizations that need it, e.g. NFW's `concentration_m200`.

    Returns:
        A `Potential` assembled from every included component.
    """
    return Potential.from_settings(
        potential, mges, unit_system, cosmological_parameters
    )


def raw_potential_parameters(
    potential_settings: Mapping[str, Mapping[str, Any]],
    potential: Potential,
    unit_system: AbstractUnitSystem,
    cosmological_parameters: Mapping[str, Quantity],
) -> dict[str, dict[str, Quantity]]:
    """Every included component's parameters, in the config's own parameterization.

    The inverse of `build_potential`/`Potential.from_settings`: where those
    convert each raw config parameter into `galax`'s native constructor
    kwargs, this converts back, e.g. NFW's `concentration_m200`'s native
    `(m, r_s)` back to `(c, M_200)`. `AllModels` uses this to report every
    model in the parameterization its configuration actually specifies,
    regardless of `Potential.rescale`, which only knows how to scale
    native parameters (see `GalaxPotentialComponent.raw_parameters`).

    `potential_settings` is the source of "which parameterization was
    configured" for each component -- `potential` itself doesn't carry that,
    since `AbstractPotentialComponent.parameters` is deliberately
    parameterization-independent.

    Args:
        potential_settings: A resolved configuration's `potential` section
            (e.g. `ModelIterator.potential_settings`) -- only each
            component's `parameterization` is used.
        potential: The resolved `Potential` to report, e.g. from
            `build_potential`, possibly after `Potential.rescale`.
        unit_system: The unit system `potential`'s parameters are expressed in.
        cosmological_parameters: A resolved configuration's
            `cosmological_parameters` section -- used only by
            parameterizations that need it, e.g. NFW's `concentration_m200`.

    Returns:
        A mapping from each included component's name to its raw
        parameters, keyed exactly as its configuration's `parameters` are.
    """
    return {
        name: component.raw_parameters(
            _mapping(potential_settings.get(name, {}), f"potential.{name}").get(
                "parameterization"
            ),
            unit_system,
            cosmological_parameters,
        )
        for name, component in potential.components.items()
    }
