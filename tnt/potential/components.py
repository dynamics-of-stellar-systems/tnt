"""One named term of a `Potential`: resolving its static structure and building it.

`AbstractPotentialComponent` (here) is the base every concrete component
subclasses -- `GalaxPotentialComponent` (also here, built directly from a
curated `galax.potential` class) and the MGE-backed composite types (one
module per deprojection convention: `tnt.potential.triaxial_mge` and
`tnt.potential.oblate_mge`). `resolve` dispatches to whichever subclass
matches a config entry's `type` via
the explicit registry in `tnt.potential.registry` -- a concrete subclass
participates by applying `tnt.potential.registry.register_component`
directly to its own definition, the moment its module is imported;
`tnt/potential/__init__.py`'s own explicit imports of every concrete
implementation module (`triaxial_mge`, `oblate_mge`) are what make that
happen in the first place -- a module that's never imported never
participates.
`ResolvedPotentialComponent` is a component's static structure
(`type`/`parameterization`/`mge`), resolved once from its config entry and
reused across every proposed point in parameter space; see
`AbstractPotentialComponent.resolve`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple, Self

import equinox as eqx
import galax.potential
from unxt import AbstractUnitSystem, Quantity

from tnt.mge import LightMGE, MassMGE
from tnt.potential.registry import (
    _SUPPORTED_GALAX_TYPES,
    ForwardConverter,
    component_type_names,
    get_component_class,
    get_parameterization,
    parameterization_names,
    raw_parameter_dimensions,
)
from tnt.validation import _required_string, _string


class ResolvedPotentialComponent(NamedTuple):
    """One potential component's static structure, resolved once from its config entry.

    Everything needed to build the component except the current parameter
    values -- fixed for a whole run, independent of any proposed point in
    parameter space. Computed once by `AbstractPotentialComponent.resolve`;
    reused across every call to `build` for the same component.
    """

    component_cls: type[AbstractPotentialComponent]
    raw_dimensions: dict[str, str]
    convert: ForwardConverter | None
    extra_fields: dict[str, Any]

    def build(
        self,
        parameter_values: Mapping[str, Quantity],
        cosmological_parameters: Mapping[str, Quantity],
    ) -> AbstractPotentialComponent:
        """Build this component from one proposed point in parameter space.

        `parameter_values` should have this component's raw parameter names
        (native, or under a `parameterization`), each already a `Quantity`
        in whatever unit it was declared/proposed in -- no unit-system
        conversion happens here (see `tnt.potential`'s module docstring for
        why). Passed straight through as-is -- a missing or otherwise wrong
        parameter surfaces at construction (a native `galax` constructor
        error, or `AbstractMGE.deproject_triaxial`/`deproject_oblate` for the
        four MGE composite types, via `AbstractPotentialComponent._build`) or a
        registered
        `parameterization` converter, not here. Exact-name parameter-schema
        validation -- for TNT's own registered component types, curated native
        `galax` types, and registered parameterizations alike -- already
        happens earlier, at configuration-prep time
        (`tnt.configuration.validation._validate_potential`). Value/domain
        validation (positivity, physical bounds, ...) isn't implemented
        anywhere yet (GitHub issue #30).

        Args:
            parameter_values: This component's current values, e.g. one
                entry of a `tnt.parameter_generator.ParameterSet`.
            cosmological_parameters: Passed through to a registered
                `parameterization` converter that needs it, e.g. NFW's
                `concentration_m200` via `H`.
        """
        raw = dict(parameter_values)
        canonical = (
            self.convert(raw, cosmological_parameters)
            if self.convert is not None
            else raw
        )
        return self.component_cls._build(
            canonical, cosmological_parameters, self.extra_fields
        )


class AbstractPotentialComponent(eqx.Module):
    """One named term of the total potential (e.g. a halo, a light MGE).

    `rescale` holds shape fixed while re-normalizing a component's overall
    mass: for `GalaxPotentialComponent`, every native parameter scales by
    its own exponent, curated per class (see
    `tnt.potential.registry._SUPPORTED_GALAX_TYPES`) -- e.g.
    `LogarithmicPotential`'s `v_c` scales alongside a true mass parameter
    like Plummer's `m_tot`, each by its own confirmed exponent. For the four
    MGE composite types, `rescale` multiplies their one TNT-defined
    mass-normalization parameter (`ml`/`mge_mass_scale`) directly.
    `parameters` always holds canonical, parameterization-independent
    fields: for a native galax type these are exactly that class's own
    constructor kwarg names; for the four MGE composite types, TNT's own
    `ml`/`mge_mass_scale`.
    """

    parameters: dict[str, Quantity]

    @classmethod
    def resolve(
        cls,
        settings: Mapping[str, Any],
        mges: Mapping[str, LightMGE | MassMGE],
        *,
        path: str = "potential.<component>",
    ) -> ResolvedPotentialComponent:
        """Resolve one potential component's static structure from its config entry.

        Everything here is fixed for a whole run, independent of any
        proposed point in parameter space -- a caller building many
        `Potential`s from the same configuration (e.g. `ModelIterator`,
        once per proposed `ParameterSet`) should call this once and reuse
        the result via `ResolvedPotentialComponent.build`, rather than
        re-deriving it every time. `Potential.from_settings` calls this
        internally for one-shot construction.

        Args:
            settings: One resolved `potential.<name>` entry: `type`, an
                optional `parameterization`, and `parameters`.
            mges: Named MGEs, e.g. from `tnt.mge.build_mges` -- used only by
                the four MGE composite types.
            path: This entry's location in the configuration, used in error
                messages.

        Returns:
            This component's resolved static structure.

        Raises:
            ValueError: If `type` names neither a supported
                `galax.potential` class (see
                `tnt.potential.registry._SUPPORTED_GALAX_TYPES`) nor a
                registered composite type's `_type`.
            NotImplementedError: If an explicit `parameterization` isn't
                registered for this `type`.
        """
        kind = _required_string(settings, "type", path)
        # Every non-native concrete subclass (e.g. the MGE composite types
        # in tnt.potential.triaxial_mge / tnt.potential.oblate_mge) registers
        # itself via tnt.potential.registry.register_component;
        # GalaxPotentialComponent doesn't, and stays the default.
        component_cls = get_component_class(kind) or GalaxPotentialComponent
        unsupported = (
            component_cls is GalaxPotentialComponent
            and kind not in _SUPPORTED_GALAX_TYPES
        )
        if unsupported:
            allowed = ", ".join(sorted(component_type_names()))
            raise ValueError(
                f"Unsupported {path}.type {kind!r}; expected a supported "
                "galax.potential class name (see "
                "tnt.potential.registry._SUPPORTED_GALAX_TYPES) or one of: "
                f"{allowed}."
            )

        parameterization_name = settings.get("parameterization")
        convert: ForwardConverter | None = None
        if parameterization_name is not None:
            _string(parameterization_name, f"{path}.parameterization")
            spec = get_parameterization(kind, parameterization_name)
            if spec is None:
                allowed = (
                    ", ".join(parameterization_names(kind)) or "(none implemented yet)"
                )
                raise NotImplementedError(
                    f"{path}.parameterization {parameterization_name!r} is not "
                    f"implemented for type {kind!r}; implemented: {allowed}."
                )
            convert = spec.convert

        return ResolvedPotentialComponent(
            component_cls=component_cls,
            raw_dimensions=raw_parameter_dimensions(kind, parameterization_name),
            convert=convert,
            extra_fields=component_cls._extra_fields(kind, settings, mges, path=path),
        )

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

    @classmethod
    def _build(
        cls,
        parameters: dict[str, Quantity],
        cosmological_parameters: Mapping[str, Quantity],
        extra_fields: dict[str, Any],
    ) -> Self:
        """Construct this component from its canonical `parameters` and static fields.

        The default just constructs directly -- the four MGE composite types
        (`tnt.potential.triaxial_mge`, `tnt.potential.oblate_mge`) override
        this to deproject and validate their MGE eagerly, here, rather than
        lazily inside `to_galax()`: this is
        the point where the proposed parameter values are turned into a concrete
        potential, so it's the appropriate place for that potential to fail if
        it's invalid (e.g. `tnt.mge.MGEDeprojectionError`), before anything
        downstream (like orbit integration) is attempted.

        Args:
            parameters: This component's canonical, parameterization-independent
                parameter values (post-`ResolvedPotentialComponent.build`'s
                `convert` step).
            cosmological_parameters: Passed through for subclasses that need it;
                unused by the default implementation.
            extra_fields: This component's resolved static structure beyond
                `parameters`, e.g. `galax_type` or `mge` -- see `_extra_fields`.
        """
        del cosmological_parameters
        return cls(parameters=parameters, **extra_fields)

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
        declared_units: Mapping[str, str],
        cosmological_parameters: Mapping[str, Quantity],
    ) -> dict[str, Quantity]:
        """This component's parameters in the resolved config's own parameterization.

        The inverse of `ResolvedPotentialComponent.build`'s conversion, so
        `AllModels` can report every component the way its configuration
        actually specifies it, regardless of `rescale`. Identity by default:
        `parameters` already *is* the raw, parameterization-independent
        representation for anything without a registered non-native
        parameterization -- the four MGE composite types (which don't support
        one at all) and a native `galax` type with `parameterization`
        omitted.

        Args:
            parameterization: The registered non-native parameterization
                the configuration specified, or `None`.
            declared_units: Each raw parameter's declared unit string, from
                the component's `potential.<name>.parameters` section --
                passed to a registered `invert` converter so a reported value
                comes back in the configured unit.
            cosmological_parameters: Passed through to an `invert` converter
                that needs it, e.g. NFW's `concentration_m200` via `H`.
        """
        del parameterization, declared_units, cosmological_parameters
        return self.parameters


class GalaxPotentialComponent(AbstractPotentialComponent):
    """A component built directly from a named `galax.potential` class."""

    # static: a structural type identifier, not a value JAX transforms
    # should trace -- without this, `jax.jit` directly over a component
    # sees `galax_type` as a dynamic string leaf and fails (though
    # `eqx.filter_jit`, which already excludes non-array leaves, works
    # either way).
    galax_type: str = eqx.field(static=True)

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
                "tnt.potential.registry._SUPPORTED_GALAX_TYPES); rescale() "
                "doesn't know its parameters' mass-rescale exponents."
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
        declared_units: Mapping[str, str],
        cosmological_parameters: Mapping[str, Quantity],
    ) -> dict[str, Quantity]:
        if parameterization is None:
            return self.parameters
        spec = get_parameterization(self.galax_type, parameterization)
        if spec is None:  # unreachable: resolve() already validated it
            raise NotImplementedError(
                f"{self.galax_type}.{parameterization!r} is not a registered "
                "parameterization."
            )
        return spec.invert(self.parameters, declared_units, cosmological_parameters)
