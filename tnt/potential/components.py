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
from typing import Any, ClassVar, NamedTuple, Self

import equinox as eqx
import galax.potential
import jax
import jax.numpy as jnp
from unxt import AbstractUnitSystem, Quantity

from tnt.mge import Deprojected3DMGE, LightMGE, MassMGE
from tnt.potential.registry import (
    _SUPPORTED_GALAX_TYPES,
    ForwardConverter,
    InvalidPotentialParametersError,
    ParameterConstraint,
    component_type_names,
    get_component_class,
    get_parameterization,
    parameter_constraints,
    parameterization_names,
    raw_parameter_dimensions,
)
from tnt.units import validate_dimension
from tnt.validation import _mapping, _required_string, _string


class ResolvedPotentialComponent(NamedTuple):
    """One potential component's static structure, resolved once from its config entry.

    Everything needed to build the component except the current parameter
    values -- fixed for a whole run, independent of any proposed point in
    parameter space. Computed once by `AbstractPotentialComponent.resolve`;
    reused across every call to `build` for the same component.
    """

    component_cls: type[AbstractPotentialComponent]
    raw_dimensions: dict[str, str]
    raw_constraints: dict[str, ParameterConstraint]
    canonical_dimensions: dict[str, str]
    canonical_constraints: dict[str, ParameterConstraint]
    convert: ForwardConverter | None
    extra_fields: dict[str, Any]
    path: str

    def _raw_parameters_valid(
        self, parameter_values: Mapping[str, Quantity]
    ) -> jax.Array:
        """JAX boolean for raw numerical values after static contract checks."""
        _mapping(parameter_values, f"{self.path}.parameters")
        _check_parameter_set_structure(
            parameter_values, self.raw_dimensions, path=self.path, stage="raw"
        )
        return _parameter_values_valid(parameter_values, self.raw_constraints)

    def build_with_validity(
        self,
        parameter_values: Mapping[str, Quantity],
        cosmological_parameters: Mapping[str, Quantity],
    ) -> tuple[AbstractPotentialComponent, jax.Array]:
        """Build a component and return its scalar JAX validity flag.

        Galax native parameters/traceable conversions and native MGE parameters
        are supported. MGE parameter conversions are deferred.
        Conversion first probes native validity without differentiation, then
        runs differentiably only for valid proposals. Invalid converted proposals
        contain zero placeholders in the converter's output units, never a
        usable physical model.
        Callers must condition numerical use of any component on the flag.
        """
        raw = dict(_mapping(parameter_values, f"{self.path}.parameters"))
        valid = self._raw_parameters_valid(raw)
        if self.component_cls is not GalaxPotentialComponent:
            if not self.component_cls._native_mge:
                raise NotImplementedError(
                    f"Traced construction is not implemented for {self.path}."
                )
            if self.convert is not None:
                raise NotImplementedError(
                    "Traced MGE parameter conversions are not implemented "
                    f"for {self.path}."
                )
            return _build_mge_with_validity(
                self.component_cls, raw, self.extra_fields, valid
            )
        if self.convert is None:
            canonical = raw
        else:

            def convert(values: dict[str, Quantity]) -> dict[str, Quantity]:
                canonical = self.convert(
                    values, cosmological_parameters, self.extra_fields.get("mge")
                )
                _check_parameter_set_structure(
                    canonical,
                    self.canonical_dimensions,
                    path=self.path,
                    stage="converted",
                )
                return canonical

            # Abstract evaluation establishes units/dtypes without numerical
            # conversion. Both conditional branches need identical structure.
            shape = jax.eval_shape(convert, raw)
            placeholder = jax.tree.map(
                lambda value: jnp.zeros(value.shape, dtype=value.dtype), shape
            )
            probe = jax.lax.stop_gradient(
                jax.lax.cond(valid, convert, lambda _: placeholder, raw)
            )
            valid = valid & _parameter_values_valid(probe, self.canonical_constraints)
            # A zero cotangent through an invalid conversion can still produce
            # NaN gradients. Keep that conversion outside the differentiated
            # path rather than masking its result after it has run.
            canonical = jax.lax.cond(valid, convert, lambda _: placeholder, raw)
        component = self.component_cls._build(
            canonical, cosmological_parameters, self.extra_fields
        )
        return component, valid

    def build(
        self,
        parameter_values: Mapping[str, Quantity],
        cosmological_parameters: Mapping[str, Quantity],
    ) -> AbstractPotentialComponent:
        """Build this component from one proposed point in parameter space.

        `parameter_values` must have this component's exact raw parameter names
        (native, or under a `parameterization`), each as a scalar, finite
        `Quantity` in whatever unit it was declared/proposed in. Physical-domain
        constraints are checked before a registered converter runs, then the
        converter's canonical output is checked again against the native
        component constraints. No unit-system normalization happens here (see
        `tnt.potential`'s module docstring); constraints convert only as needed
        for a comparison. Native MGE construction shares its numerical checks
        with `build_with_validity`, including geometry and derivative checks.
        Validation uses Python boolean checks, so `build` is an
        eager runtime boundary and must remain outside `jax.jit`/`jax.vmap`
        traces; only the resulting potential enters compiled numerical work.

        Args:
            parameter_values: This component's current values, e.g. one
                entry of a `tnt.parameter_generator.ParameterSet`.
            cosmological_parameters: Passed through to a registered
                `parameterization` converter that needs it, e.g. NFW's
                `concentration_m200` via `H`.
        """
        raw = dict(_mapping(parameter_values, f"{self.path}.parameters"))
        _check_parameter_set_contract(
            raw, self.raw_dimensions, path=self.path, stage="raw"
        )
        _validate_parameter_constraints(
            raw, self.raw_constraints, path=self.path, stage="raw"
        )
        if self.convert is None:
            canonical = raw
        else:
            canonical = self.convert(
                raw, cosmological_parameters, self.extra_fields.get("mge")
            )
            _check_parameter_set_contract(
                canonical,
                self.canonical_dimensions,
                path=self.path,
                stage="converted",
            )
            _validate_parameter_constraints(
                canonical,
                self.canonical_constraints,
                path=self.path,
                stage="converted",
            )
        if self.component_cls._native_mge and self.convert is None:
            component, valid = _build_mge_with_validity(
                self.component_cls, canonical, self.extra_fields, jnp.asarray(True)
            )
            if not bool(valid):
                # Preserve the eager geometry diagnostics from the same checks.
                self.component_cls._build(
                    canonical, cosmological_parameters, self.extra_fields
                )
                raise InvalidPotentialParametersError(
                    f"Invalid construction for {self.path}: converted values or "
                    "derivatives are not representable at this numerical precision."
                )
            return component
        return self.component_cls._build(
            canonical, cosmological_parameters, self.extra_fields
        )


def _build_mge_with_validity(
    component_cls: type[AbstractPotentialComponent],
    parameters: dict[str, Quantity],
    extra_fields: dict[str, Any],
    raw_valid: jax.Array,
) -> tuple[AbstractPotentialComponent, jax.Array]:
    """Shared native construction for light/mass and oblate/triaxial MGEs."""
    mge = extra_fields["mge"]

    def candidate(
        values: dict[str, Quantity],
    ) -> tuple[Deprojected3DMGE, jax.Array]:
        mass = (
            mge.to_mass(values["ml"])
            if isinstance(mge, LightMGE)
            else mge.rescaled(values["mge_mass_scale"])
        )
        if "inclination" in values:
            mass._check_deprojection_structure(values["inclination"])
            return mass._oblate_candidate(values["inclination"])
        mass._check_deprojection_structure(
            values["theta"], values["phi"], values["psi"]
        )
        return mass._triaxial_candidate(values["theta"], values["phi"], values["psi"])

    def numerical_outputs(numbers: dict[str, jax.Array]) -> tuple[jax.Array, ...]:
        values = {
            key: Quantity(number, parameters[key].unit)
            for key, number in numbers.items()
        }
        model, _ = candidate(values)
        total = model.component_masses
        return tuple(
            value.ustrip(value.unit)
            for value in (model.I, model.sigma, model.p, model.q, total)
        ) + (
            model.I.ustrip("Msun / kpc3"),
            model.sigma.ustrip("kpc"),
            total.ustrip("Msun"),
        )

    def probe(values: dict[str, Quantity]) -> jax.Array:
        _, valid = candidate(values)
        numbers = {
            key: jnp.asarray(
                value.ustrip(value.unit),
                dtype=jnp.result_type(value.ustrip(value.unit), float),
            )
            for key, value in values.items()
        }
        outputs = numerical_outputs(numbers)
        derivatives = jax.jacfwd(numerical_outputs)(numbers)
        for leaf in jax.tree.leaves((outputs, derivatives)):
            valid = valid & jnp.all(jnp.isfinite(leaf))
        for output in outputs[-3:]:
            valid = valid & jnp.all(output > 0)
        return valid

    shape = jax.eval_shape(candidate, parameters)[0]
    placeholder = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), shape)
    if not any(
        isinstance(x, jax.core.Tracer)
        for x in jax.tree.leaves((parameters, mge, raw_valid))
    ):
        valid = raw_valid & probe(parameters) if bool(raw_valid) else jnp.asarray(False)
        model = candidate(parameters)[0] if bool(valid) else placeholder
        return component_cls(parameters=parameters, mge=mge, deprojected=model), valid
    # Invalid normalization/angles must never enter differentiated scaling or
    # deprojection, including 0 * inf in a reverse-mode cotangent.
    geometry_valid = jax.lax.cond(
        raw_valid,
        probe,
        lambda _: jnp.asarray(False),
        jax.tree.map(jax.lax.stop_gradient, parameters),
    )
    valid = raw_valid & geometry_valid
    deprojected = jax.lax.cond(
        valid, lambda values: candidate(values)[0], lambda _: placeholder, parameters
    )
    return component_cls(parameters=parameters, mge=mge, deprojected=deprojected), valid


def _parameter_values_valid(
    values: Mapping[str, Quantity], constraints: Mapping[str, ParameterConstraint]
) -> jax.Array:
    """Combine finiteness and registered bounds for structurally checked values."""
    valid = jnp.asarray(True)
    for value in values.values():
        valid = valid & jnp.isfinite(value.ustrip(value.unit))
    for name, constraint in constraints.items():
        valid = valid & constraint.valid(values[name], values)
    return valid


def _check_parameter_set_contract(
    values: Mapping[str, Quantity],
    dimensions: Mapping[str, str],
    *,
    path: str,
    stage: str,
) -> None:
    """Check static structure and eager finiteness of one parameter mapping."""
    _check_parameter_set_structure(values, dimensions, path=path, stage=stage)
    for name, value in values.items():
        if not bool(jnp.isfinite(value.ustrip(value.unit))):
            raise InvalidPotentialParametersError(
                f"Invalid {stage} value for {path}.parameters.{name}: "
                f"{value} must be finite."
            )


def _check_parameter_set_structure(
    values: Mapping[str, Quantity],
    dimensions: Mapping[str, str],
    *,
    path: str,
    stage: str,
) -> None:
    """Check names, Quantity types, dimensions, and scalar shapes before tracing."""
    expected = set(dimensions)
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise InvalidPotentialParametersError(
            f"Invalid {stage} parameters for {path}: {', '.join(details)}."
        )

    for name, value in values.items():
        label = f"{path}.parameters.{name}"
        if not isinstance(value, Quantity):
            raise TypeError(
                f"Invalid {stage} value for {label}: expected a Quantity, "
                f"got {type(value).__name__}."
            )
        try:
            validate_dimension(value.unit, dimensions[name], label)
        except ValueError as error:
            raise InvalidPotentialParametersError(
                f"Invalid {stage} value: {error}"
            ) from error
        stripped = value.ustrip(value.unit)
        if not (
            jnp.issubdtype(stripped.dtype, jnp.integer)
            or jnp.issubdtype(stripped.dtype, jnp.floating)
        ):
            raise TypeError(
                f"Invalid {stage} value for {label}: expected a real number."
            )
        if getattr(stripped, "shape", ()) != ():
            raise InvalidPotentialParametersError(
                f"Invalid {stage} value for {label}: expected a scalar, "
                f"got shape {getattr(stripped, 'shape', None)}."
            )


def _validate_parameter_constraints(
    values: Mapping[str, Quantity],
    constraints: Mapping[str, ParameterConstraint],
    *,
    path: str,
    stage: str,
) -> None:
    """Apply registered physical-domain constraints to checked parameters."""
    for name, constraint in constraints.items():
        violation = constraint.violation(values[name], values)
        if violation is not None:
            raise InvalidPotentialParametersError(
                f"Invalid {stage} value for {path}.parameters.{name}: {violation}"
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

    _constraints: ClassVar[dict[str, ParameterConstraint]] = {}
    _native_mge: ClassVar[bool] = False

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
            raw_constraints=parameter_constraints(kind, parameterization_name),
            canonical_dimensions=raw_parameter_dimensions(kind, None),
            canonical_constraints=parameter_constraints(kind, None),
            convert=convert,
            extra_fields=component_cls._extra_fields(kind, settings, mges, path=path),
            path=path,
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

    def _registry_type_name(self) -> str:
        """The key this component looks itself up under in the parameterization
        registry (`tnt.potential.registry.get_parameterization`).

        Every `tnt.potential.registry.register_component`-registered subclass
        already declares this as its own `_type` `ClassVar`, so that's the
        default. `GalaxPotentialComponent` -- the only concrete subclass that
        isn't registered that way -- overrides this to `self.galax_type`.
        """
        return self._type

    def raw_parameters(
        self,
        parameterization: str | None,
        declared_units: Mapping[str, str],
        cosmological_parameters: Mapping[str, Quantity],
    ) -> dict[str, Quantity]:
        """This component's parameters in the resolved config's own parameterization.

        The inverse of `ResolvedPotentialComponent.build`'s conversion, so
        `AllModels` can report every component the way its configuration
        actually specifies it, regardless of `rescale`. `parameters` already
        *is* the raw, parameterization-independent representation when no
        `parameterization` was configured; otherwise this looks up and runs
        the registered `invert` converter itself, via `_registry_type_name`,
        so every component type gets correct dispatch without its own
        override. The converter's trailing `mge` argument is read directly
        off `self` (every MGE composite type stores its MGE in a field named
        `mge`; `None` for a curated native `galax` type, which never carries
        one) -- `ForwardConverter`'s matching argument comes from
        `ResolvedPotentialComponent.build`'s `extra_fields`, before a
        component exists to read it from.

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
        if parameterization is None:
            return self.parameters
        spec = get_parameterization(self._registry_type_name(), parameterization)
        if spec is None:  # unreachable: resolve() already validated it
            raise NotImplementedError(
                f"{self._registry_type_name()}.{parameterization!r} is not a "
                "registered parameterization."
            )
        return spec.invert(
            self.parameters,
            declared_units,
            cosmological_parameters,
            getattr(self, "mge", None),
        )


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

    def _registry_type_name(self) -> str:
        """`galax_type`, not `_type` -- see `AbstractPotentialComponent`'s docstring."""
        return self.galax_type
