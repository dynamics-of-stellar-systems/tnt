"""Galactic potentials, assembled from named, `galax`-backed components.

`potential.<name>.type` names either a `galax.potential` class directly
(e.g. `"NFWPotential"`, `"PlummerPotential"`) or one of two TNT-specific MGE
composite potentials (`"triaxial_light_mge"`, `"triaxial_mass_mge"`) that
aren't single galax classes. `parameterization` is a separate, optional
concern: when omitted, `parameters` must use the resolved type's own native
constructor kwargs -- both their physical dimensions and, for `rescale`, the
mass-normalization parameter are derived directly from galax's own
`ParameterField` metadata (see `raw_parameter_dimensions`/
`native_parameter_dimensions`), not hand-maintained per type. When given,
`parameterization` names a registered conversion from some other raw
parameter convention (e.g. NFW's `concentration_mass_ratio`, not yet
implemented -- the `(c, f) -> (m, r_s)` formula needs a confirmed reference)
into those same native fields.

This module is filled in incrementally, one object at a time -- the same
approach already used for `ProjectedBinning`. `Potential.generate_orbit_library`
and the two MGE composite components' `to_galax` remain `NotImplementedError`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Self

import equinox as eqx
import galax.potential
import unxt as u
from galax.potential.params import ParameterField
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
    [dict[str, Quantity], AbstractUnitSystem], dict[str, Quantity]
]


def native_parameter_dimensions(galax_type: str) -> dict[str, str] | None:
    """Each of `galax_type`'s native constructor parameters' physical dimension.

    Derived from galax's own `ParameterField(dimensions=...)` metadata, so
    this works for any `galax.potential` class -- not a hand-maintained
    table grown one type at a time.

    Args:
        galax_type: A name from `galax.potential`, e.g. `"NFWPotential"`.

    Returns:
        A mapping from each native constructor parameter name to its
        physical dimension (e.g. `"mass"`, `"length"`, `"dimensionless"`),
        or `None` if `galax_type` doesn't name a real
        `galax.potential.AbstractPotential` subclass.
    """
    cls = getattr(galax.potential, galax_type, None)
    if not (
        isinstance(cls, type) and issubclass(cls, galax.potential.AbstractPotential)
    ):
        return None
    return _native_parameter_dimensions(cls)


def _native_parameter_dimensions(
    galax_cls: type[galax.potential.AbstractPotential],
) -> dict[str, str]:
    return {
        f.name: str(raw.dimensions)
        for f in dataclasses.fields(galax_cls)
        if isinstance(raw := galax_cls.__dict__.get(f.name), ParameterField)
    }


def _native_mass_parameter(galax_cls: type[galax.potential.AbstractPotential]) -> str:
    """The one native parameter with dimension `"mass"`.

    Every potential component has exactly one mass-normalization parameter
    (see `AbstractPotentialComponent.rescale`'s docstring); for a native
    galax type, that's whichever of its own constructor parameters has
    dimension `"mass"`.
    """
    candidates = [
        name
        for name, dimension in _native_parameter_dimensions(galax_cls).items()
        if dimension == "mass"
    ]
    if len(candidates) != 1:
        raise NotImplementedError(
            f"{galax_cls.__name__} has {len(candidates)} mass-dimensioned "
            "parameter(s) (expected exactly 1); rescale() needs a single "
            "mass-normalization parameter to scale."
        )
    return candidates[0]


# Raw parameter dimensions for the two TNT MGE composite types, which have
# no native galax class to introspect. `triaxial_mass_mge` has no entry for
# its `mge_mass_scale` parameter: like `ml`, it's a pure multiplicative
# scale factor with no physical unit of its own (dimensionless).
# `triaxial_mass_mge` still declares `ml` here (even though `ml` is invalid
# for it) so that a config mistakenly declaring one gets normalized as
# mass_to_light rather than rejected here as "dimensionless" -- letting
# `_validate_potential` raise the more specific "ml is invalid for a mass
# MGE potential" error instead.
_MGE_RAW_DIMENSIONS: dict[str, dict[str, str]] = {
    "triaxial_light_mge": {"ml": "mass_to_light"},
    "triaxial_mass_mge": {"ml": "mass_to_light"},
}

# Raw parameter dimensions for registered non-native parameterizations,
# keyed by (type, parameterization). Populated alongside `_PARAMETERIZATIONS`
# below.
PARAMETERIZATION_RAW_DIMENSIONS: dict[tuple[str, str], dict[str, str]] = {
    ("NFWPotential", "concentration_mass_ratio"): {},  # c, f both dimensionless
}


def raw_parameter_dimensions(kind: str, parameterization: str | None) -> dict[str, str]:
    """Each raw config parameter's physical dimension for one `type`/`parameterization`.

    Covers all three sources of truth in this module: a TNT MGE composite
    type's own hand-declared dimensions, a registered non-native
    parameterization's hand-declared raw dimensions, or -- the common case
    -- a real galax class's native constructor kwargs, derived dynamically
    via `native_parameter_dimensions`. Returns `{}` (every parameter treated
    as dimensionless) for anything unrecognized, deferring the "is this
    actually a valid type" question to `AbstractPotentialComponent.from_settings`.
    """
    if parameterization is not None:
        return PARAMETERIZATION_RAW_DIMENSIONS.get((kind, parameterization), {})
    if kind in _MGE_RAW_DIMENSIONS:
        return _MGE_RAW_DIMENSIONS[kind]
    return native_parameter_dimensions(kind) or {}


def _resolve_unit(unit_system: AbstractUnitSystem, dimension: str) -> Any:
    """Resolve one of `raw_parameter_dimensions`' dimension names to a concrete unit.

    Mirrors `tnt.units._internal_unit`'s handling of `"mass_to_light"` -- a
    TNT pseudo-dimension (not a real astropy physical type) that appears
    here via the two MGE composite types' `ml` parameter.
    """
    if dimension == "mass_to_light":
        return unit_system[u.dimension("mass")] / unit_system[u.dimension("power")]
    return unit_system[u.dimension(dimension)]


def _nfw_concentration_mass_ratio(
    raw: dict[str, Quantity], unit_system: AbstractUnitSystem
) -> dict[str, Quantity]:
    """Convert NFW's `(c, f)` parameterization to native `(m, r_s)`.

    Not yet implemented: converting concentration `c` and the mass-ratio
    parameter `f` into galax's native `(m, r_s)` requires a formula from the
    triaxial-Schwarzschild-modeling / DYNAMITE-successor literature that
    hasn't been confirmed yet.
    """
    del raw, unit_system
    raise NotImplementedError(
        "NFWPotential's 'concentration_mass_ratio' parameterization "
        "((c, f) -> (m, r_s)) is not yet implemented: the conversion "
        "formula has not been confirmed."
    )


_PARAMETERIZATIONS: dict[str, dict[str, ParameterizationConverter]] = {
    "NFWPotential": {"concentration_mass_ratio": _nfw_concentration_mass_ratio},
}


class AbstractPotentialComponent(eqx.Module):
    """One named term of the total potential (e.g. a halo, a light MGE).

    Every component has exactly one mass-normalization parameter on top of
    an otherwise-fixed shape; `rescale` multiplies that parameter, holding
    shape fixed. `parameters` always holds canonical, parameterization
    -independent fields: for a native galax type these are exactly that
    class's own constructor kwarg names; for the two MGE composite types,
    TNT's own `ml`/`mge_mass_scale`.
    """

    parameters: dict[str, Quantity]

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        mges: Mapping[str, LightMGE | MassMGE],
        unit_system: AbstractUnitSystem,
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
            path: This entry's location in the configuration, used in error
                messages.

        Returns:
            The resolved component, either a `GalaxPotentialComponent` (for
            any real `galax.potential` class name) or one of the two MGE
            composite components.

        Raises:
            ValueError: If `type` names neither a real
                `galax.potential.AbstractPotential` subclass nor one of the
                two MGE composite type names.
            NotImplementedError: If an explicit `parameterization` isn't
                registered for this `type`.
        """
        kind = _required_string(settings, "type", path)
        component_cls = _MGE_COMPOSITE_CLASSES.get(kind, GalaxPotentialComponent)
        if component_cls is GalaxPotentialComponent:
            galax_cls = getattr(galax.potential, kind, None)
            if not (
                isinstance(galax_cls, type)
                and issubclass(galax_cls, galax.potential.AbstractPotential)
            ):
                allowed = ", ".join(sorted(_MGE_COMPOSITE_CLASSES))
                raise ValueError(
                    f"Unsupported {path}.type {kind!r}; expected a "
                    f"galax.potential class name or one of: {allowed}."
                )

        parameterization_name = settings.get("parameterization")
        convert: ParameterizationConverter | None = None
        if parameterization_name is not None:
            _string(parameterization_name, f"{path}.parameterization")
            converters = _PARAMETERIZATIONS.get(kind, {})
            try:
                convert = converters[parameterization_name]
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

        canonical = convert(raw, unit_system) if convert is not None else raw
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
        """Multiply this component's mass-normalization parameter.

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
        potential_cls = getattr(galax.potential, self.galax_type)
        key = _native_mass_parameter(potential_cls)
        rescaled = dict(self.parameters)
        rescaled[key] = rescaled[key] * mass_scale
        return eqx.tree_at(lambda c: c.parameters, self, rescaled)


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
    ) -> Self:
        """Build a `Potential` from a resolved configuration's `potential` section."""
        components: dict[str, AbstractPotentialComponent] = {}
        for name, component_value in settings.items():
            path = f"potential.{name}"
            component_settings = _mapping(component_value, path)
            if not component_settings.get("include", True):
                continue
            components[name] = AbstractPotentialComponent.from_settings(
                component_settings, mges, unit_system, path=path
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
) -> Potential:
    """Build the `Potential` from a resolved configuration's `potential` section.

    Args:
        potential: A resolved configuration's `potential` section.
        mges: Named MGEs, e.g. from `tnt.mge.build_mges`.
        unit_system: The unit system `potential`'s parameter values are
            already expressed in, and that the resulting `galax` potential
            will be constructed with.

    Returns:
        A `Potential` assembled from every included component.
    """
    return Potential.from_settings(potential, mges, unit_system)
