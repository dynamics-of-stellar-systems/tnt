"""Curated `galax.potential` types and TNT's own potential-component registry.

`raw_parameter_dimensions` draws each config parameter's dimension from one
of three registries:

- `_SUPPORTED_GALAX_TYPES` -- a curated galax class's native constructor
  kwargs, used for any `type` without a registered `parameterization`;
- `_COMPONENT_REGISTRY` -- TNT's own composite types (the four MGE
  potentials), each registered by `register_component`; schema access reads
  its `_raw_dimensions` from the registered class;
- `_PARAMETERIZATION_REGISTRY` -- non-native parameterizations (e.g. NFW's
  `concentration_m200`), each registered by `register_parameterization`,
  which bundles the converters, raw schema, and raw domain constraints in
  one call.

Each has exactly one declaration site per entry. Normal `tnt.potential`
initialization explicitly imports each concrete component and parameterization
module so its `register_*` calls run. Configuration preparation may read this
static metadata; registration does not construct component instances or load
scientific input data.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal, NamedTuple

from unxt import Quantity

from tnt.registry import register_typed_class

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


class InvalidPotentialParametersError(ValueError):
    """A proposed parameter set cannot define a physically valid potential."""


_RELATION_OPERATORS: dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


class ParameterConstraint(NamedTuple):
    """Bounds and an optional same-component parameter relationship.

    Numeric bounds are interpreted in ``unit`` when supplied, or in the
    parameter value's own unit otherwise. Runtime validation independently
    requires every potential parameter to be scalar and finite, including
    parameters with no additional constraint entry. When both
    ``other_parameter`` and ``relation`` are supplied, the rule is interpreted
    as ``this parameter relation other_parameter`` after compatible-unit
    conversion.
    """

    minimum: float | None = None
    minimum_inclusive: bool = True
    maximum: float | None = None
    maximum_inclusive: bool = True
    unit: str | None = None
    other_parameter: str | None = None
    relation: Literal[">", ">=", "<", "<="] | None = None

    def violation(
        self,
        value: Quantity,
        siblings: Mapping[str, Quantity],
    ) -> str | None:
        """Return why ``value`` violates this constraint, or ``None``.

        Constraint metadata has already been checked by registration, and the
        caller must first verify the parameter mapping's names, types,
        dimensions, scalar shapes, and finite values.
        """
        unit = self.unit or value.unit
        try:
            number = float(value.ustrip(unit))
        except (TypeError, ValueError):
            return f"{value} cannot be compared in constraint unit {unit!s}."

        if self.minimum is not None:
            valid = (
                number >= self.minimum
                if self.minimum_inclusive
                else number > self.minimum
            )
            if not valid:
                relation = "at least" if self.minimum_inclusive else "greater than"
                suffix = f" {unit}" if str(unit) else ""
                return f"{value} must be {relation} {self.minimum}{suffix}."

        if self.maximum is not None:
            valid = (
                number <= self.maximum
                if self.maximum_inclusive
                else number < self.maximum
            )
            if not valid:
                relation = "at most" if self.maximum_inclusive else "less than"
                suffix = f" {unit}" if str(unit) else ""
                return f"{value} must be {relation} {self.maximum}{suffix}."

        if self.other_parameter is None:
            return None
        other_value = siblings[self.other_parameter]
        try:
            other = float(other_value.ustrip(unit))
        except (TypeError, ValueError):
            return (
                f"{value} and parameter {self.other_parameter!r} "
                f"({other_value}) do not have compatible units."
            )
        if not _RELATION_OPERATORS[self.relation](number, other):
            return (
                f"{value} must be {self.relation} parameter "
                f"{self.other_parameter!r} ({other_value})."
            )
        return None


class ParameterizationSpec(NamedTuple):
    """A non-native parameterization's converters, raw schema, and constraints.

    One `register_parameterization` call bundles all of this metadata, so the
    convert/invert functions, config parameter schema, and runtime domain rules
    `tnt.configuration.validation._validate_potential` checks against can
    never be registered apart or drift out of step. `AllModels` relies on
    `invert` existing for every `parameterization` a config can specify (see
    `tnt.potential.raw_potential_parameters`).
    """

    convert: ForwardConverter
    """Raw config parameters -> the type's native `galax` constructor kwargs."""
    invert: InverseConverter
    """Native `galax` constructor kwargs -> raw config parameters."""
    raw_dimensions: dict[str, str]
    """Each raw config parameter's physical dimension, for schema validation."""
    raw_constraints: dict[str, ParameterConstraint]
    """Physical-domain constraints on raw configuration parameters."""


class NativeParameter(NamedTuple):
    """A native parameter's dimension, mass-rescale exponent, and domain."""

    dimension: str
    exponent: float
    constraint: ParameterConstraint | None = None


_POSITIVE = ParameterConstraint(minimum=0.0, minimum_inclusive=False)


def _mass(exponent: float = 1.0) -> NativeParameter:
    return NativeParameter("mass", exponent, _POSITIVE)


def _length() -> NativeParameter:
    return NativeParameter("length", 0.0, _POSITIVE)


def _angle(constraint: ParameterConstraint | None = None) -> NativeParameter:
    return NativeParameter("angle", 0.0, constraint)


def _dimensionless(
    constraint: ParameterConstraint | None = None,
) -> NativeParameter:
    return NativeParameter("dimensionless", 0.0, constraint)


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
    "HarmonicOscillatorPotential": {
        "omega": NativeParameter("frequency", 0.5, _POSITIVE)
    },
    "HernquistPotential": {"m_tot": _mass(), "r_s": _length()},
    "IsochronePotential": {"m_tot": _mass(), "r_s": _length()},
    "JaffePotential": {"m_tot": _mass(), "r_s": _length()},
    "KeplerPotential": {"m_tot": _mass()},
    "KuzminPotential": {"m_tot": _mass(), "r_s": _length()},
    "LMJ09LogarithmicPotential": {
        "v_c": NativeParameter("speed", 0.5, _POSITIVE),
        "r_s": _length(),
        "q1": _dimensionless(_POSITIVE),
        "q2": _dimensionless(_POSITIVE),
        "q3": _dimensionless(_POSITIVE),
        "phi": _angle(),
    },
    "LeeSutoTriaxialNFWPotential": {
        "m": _mass(),
        "r_s": _length(),
        "a1": _dimensionless(
            ParameterConstraint(
                minimum=0.0,
                minimum_inclusive=False,
                other_parameter="a2",
                relation=">=",
            )
        ),
        "a2": _dimensionless(
            ParameterConstraint(
                minimum=0.0,
                minimum_inclusive=False,
                other_parameter="a3",
                relation=">=",
            )
        ),
        "a3": _dimensionless(_POSITIVE),
    },
    "LogarithmicPotential": {
        "v_c": NativeParameter("speed", 0.5, _POSITIVE),
        "r_s": _length(),
    },
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
        "v0": NativeParameter("speed", 0.5, _POSITIVE),
        "Rb": _length(),
        "phi_b": _angle(),
        "Omega": NativeParameter("frequency", 0.0),
    },
    "NFWPotential": {"m": _mass(), "r_s": _length()},
    "PlummerPotential": {"m_tot": _mass(), "r_s": _length()},
    "PowerLawCutoffPotential": {
        "m_tot": _mass(),
        "alpha": _dimensionless(
            ParameterConstraint(
                minimum=0.0,
                maximum=3.0,
                maximum_inclusive=False,
            )
        ),
        "r_c": _length(),
    },
    "SatohPotential": {"m_tot": _mass(), "a": _length(), "b": _length()},
    "StoneOstriker15Potential": {
        "m_tot": _mass(),
        "r_c": _length(),
        "r_h": NativeParameter(
            "length",
            0.0,
            ParameterConstraint(
                minimum=0.0,
                minimum_inclusive=False,
                other_parameter="r_c",
                relation=">",
            ),
        ),
    },
    "TriaxialHernquistPotential": {
        "m_tot": _mass(),
        "r_s": _length(),
        "q1": _dimensionless(_POSITIVE),
        "q2": _dimensionless(_POSITIVE),
    },
    "TriaxialNFWPotential": {
        "m": _mass(),
        "r_s": _length(),
        "q1": _dimensionless(_POSITIVE),
        "q2": _dimensionless(_POSITIVE),
    },
    "Vogelsberger08TriaxialNFWPotential": {
        "m": _mass(),
        "r_s": _length(),
        "q1": _dimensionless(
            ParameterConstraint(
                minimum=0.0,
                minimum_inclusive=False,
                maximum=3**0.5,
                maximum_inclusive=False,
            )
        ),
        "a_r": _dimensionless(_POSITIVE),
    },
    "gNFWPotential": {
        "m": _mass(),
        "r_s": _length(),
        "gamma": _dimensionless(
            ParameterConstraint(
                minimum=0.0,
                maximum=2.0,
                maximum_inclusive=False,
            )
        ),
    },
}


# Shared by the two *triaxial* MGE composite types' own `_raw_dimensions` --
# the global viewing angles they deproject against
# (`tnt.mge.AbstractMGE.deproject_triaxial`), required regardless of light
# vs. mass. The oblate pair uses a single `inclination` instead (see
# `tnt.potential.oblate_mge`).
_VIEWING_ANGLES: dict[str, str] = {"theta": "angle", "phi": "angle", "psi": "angle"}

# TNT's own potential-component types (as opposed to native `galax` types,
# `_SUPPORTED_GALAX_TYPES` above), keyed by `_type`. Populated by
# `register_component`, applied directly to each concrete
# `AbstractPotentialComponent` subclass in its own defining module -- this
# dict is the single source used through lookup, type-predicate, and schema
# accessors by runtime dispatch and configuration preparation; there is no
# second, independently maintained list of TNT type names or dimensions.
_COMPONENT_REGISTRY: dict[str, type[AbstractPotentialComponent]] = {}


def _validate_constraint_metadata(
    constraints: Mapping[str, ParameterConstraint],
    parameter_names: set[str],
    *,
    context: str,
) -> None:
    """Reject constraint metadata that disagrees with its parameter schema."""
    unknown = set(constraints) - parameter_names
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{context}: constraint(s) not present in schema: {names}.")
    for name, constraint in constraints.items():
        if not isinstance(constraint, ParameterConstraint):
            raise TypeError(
                f"{context}: constraint for {name!r} must be a "
                "ParameterConstraint."
            )
        if constraint.relation not in {None, ">", ">=", "<", "<="}:
            raise ValueError(
                f"{context}: constraint for {name!r} has unsupported relation "
                f"{constraint.relation!r}."
            )
        relationship_is_complete = (
            constraint.other_parameter is None
        ) == (constraint.relation is None)
        if not relationship_is_complete:
            raise ValueError(
                f"{context}: constraint for {name!r} must specify both "
                "other_parameter and relation."
            )
        if constraint.other_parameter is not None:
            if constraint.other_parameter not in parameter_names:
                raise ValueError(
                    f"{context}: constraint for {name!r} refers to unknown "
                    f"parameter {constraint.other_parameter!r}."
                )
            if constraint.other_parameter == name:
                raise ValueError(
                    f"{context}: constraint for {name!r} cannot compare the "
                    "parameter with itself."
                )
        if (
            constraint.minimum is not None
            and constraint.maximum is not None
            and (
                constraint.minimum > constraint.maximum
                or (
                    constraint.minimum == constraint.maximum
                    and not (
                        constraint.minimum_inclusive
                        and constraint.maximum_inclusive
                    )
                )
            )
        ):
            raise ValueError(f"{context}: constraint for {name!r} has empty bounds.")


for _native_type, _native_parameters in _SUPPORTED_GALAX_TYPES.items():
    _validate_constraint_metadata(
        {
            name: parameter.constraint
            for name, parameter in _native_parameters.items()
            if parameter.constraint is not None
        },
        set(_native_parameters),
        context=f"Invalid native potential metadata for {_native_type!r}",
    )


def register_component(
    cls: type[AbstractPotentialComponent],
) -> type[AbstractPotentialComponent]:
    """Register `cls` under its own explicitly declared `_type` for dispatch.

    Applied directly to a concrete `AbstractPotentialComponent` subclass's
    definition, e.g. `@register_component` above `class
    TriaxialLightMGEPotential(AbstractPotentialComponent): ...`. The registry
    stores the class itself; schema/domain access subsequently reads that
    class's `_raw_dimensions` and `_constraints`, so the type name, dimensions,
    and physical rules are not repeated in a separate registry.

    Raises:
        TypeError: If the class does not declare its own non-empty string
            `_type`.
        ValueError: If another registered class already declared the same
            `_type`, or if its constraint metadata disagrees with its schema.
    """
    constraints = getattr(cls, "_constraints", {})
    dimensions = getattr(cls, "_raw_dimensions", {})
    _validate_constraint_metadata(
        constraints,
        set(dimensions),
        context=f"Cannot register potential component {getattr(cls, '_type', None)!r}",
    )
    return register_typed_class(_COMPONENT_REGISTRY, cls, family="potential")


def get_component_class(
    type_name: str,
) -> type[AbstractPotentialComponent] | None:
    """Return TNT's registered component class for ``type_name``, if any."""
    return _COMPONENT_REGISTRY.get(type_name)


def component_type_names() -> frozenset[str]:
    """Return every explicitly registered TNT component type name."""
    return frozenset(_COMPONENT_REGISTRY)


def is_registered_component_type(type_name: str) -> bool:
    """Whether ``type_name`` identifies a registered TNT component class."""
    return type_name in _COMPONENT_REGISTRY


# TNT's non-native parameterizations, keyed by `(type, parameterization)`.
# Populated by `register_parameterization`, called at import from the module
# that owns each parameterization's numerics (e.g. `tnt.potential.nfw`) --
# `tnt/potential/__init__.py`'s explicit imports are what run those, exactly
# as for `_COMPONENT_REGISTRY`. One entry per parameterization is the single
# place its converters, config parameter schema, and raw constraints are
# declared.
_PARAMETERIZATION_REGISTRY: dict[tuple[str, str], ParameterizationSpec] = {}


def register_parameterization(
    *,
    type_name: str,
    name: str,
    convert: ForwardConverter,
    invert: InverseConverter,
    raw_dimensions: Mapping[str, str],
    raw_constraints: Mapping[str, ParameterConstraint],
) -> None:
    """Register a non-native `parameterization` for `type_name` under `name`.

    Bundles the forward/inverse converters with the raw config parameter
    schema and its domain constraints so config validation
    (`_validate_potential`) and runtime resolution can never disagree on which
    parameterizations exist, what parameters they take, or which raw domains
    they accept.

    `type_name` must be a curated native `galax` class
    (`_SUPPORTED_GALAX_TYPES`). A parameterization converts a raw config
    convention into a component's native `galax` constructor kwargs, and only
    `GalaxPotentialComponent` runs the inverse converter that reports a model
    back in its configured parameterization (`AllModels`); a TNT MGE composite
    type would silently round-trip through its canonical parameters instead.
    Supporting composite types needs the inverse dispatch moved to a
    type-independent layer first.

    Constraint names must be a subset of ``raw_dimensions`` so schema and
    domain metadata cannot silently disagree.

    Raises:
        ValueError: If `type_name` is not a curated native `galax` type, or if
            `(type_name, name)` is already registered, or a constraint names
            an unknown raw parameter.
    """
    if type_name not in _SUPPORTED_GALAX_TYPES:
        raise ValueError(
            f"Cannot register parameterization {name!r}: {type_name!r} is not a "
            "curated native galax type. Parameterizations are only supported for "
            "native galax component types."
        )
    _validate_constraint_metadata(
        raw_constraints,
        set(raw_dimensions),
        context=f"Cannot register parameterization {name!r}",
    )
    key = (type_name, name)
    if key in _PARAMETERIZATION_REGISTRY:
        raise ValueError(f"Duplicate parameterization {name!r} for type {type_name!r}.")
    _PARAMETERIZATION_REGISTRY[key] = ParameterizationSpec(
        convert,
        invert,
        dict(raw_dimensions),
        dict(raw_constraints),
    )


def get_parameterization(type_name: str, name: str) -> ParameterizationSpec | None:
    """The registered `ParameterizationSpec` for `(type_name, name)`, or `None`."""
    return _PARAMETERIZATION_REGISTRY.get((type_name, name))


def parameterization_names(type_name: str) -> list[str]:
    """Every registered parameterization name for `type_name`, sorted."""
    return sorted(
        name for (kind, name) in _PARAMETERIZATION_REGISTRY if kind == type_name
    )


def raw_parameter_dimensions(kind: str, parameterization: str | None) -> dict[str, str]:
    """Each raw config parameter's physical dimension for one `type`/`parameterization`.

    Covers all three sources of truth this module knows about: a registered
    TNT component type's own `_raw_dimensions` (`_COMPONENT_REGISTRY`), a
    registered non-native parameterization's raw schema
    (`_PARAMETERIZATION_REGISTRY`), or -- the common case -- a curated galax
    class's native constructor kwargs, read directly from
    `_SUPPORTED_GALAX_TYPES`. Returns `{}` (every parameter treated as
    dimensionless) for anything unrecognized; `parameter_schema_is_known`
    tells that apart from a genuinely empty schema.
    """
    if parameterization is not None:
        spec = _PARAMETERIZATION_REGISTRY.get((kind, parameterization))
        return dict(spec.raw_dimensions) if spec is not None else {}
    component_cls = get_component_class(kind)
    if component_cls is not None:
        return component_cls._raw_dimensions
    return {
        name: parameter.dimension
        for name, parameter in _SUPPORTED_GALAX_TYPES.get(kind, {}).items()
    }


def parameter_constraints(
    kind: str, parameterization: str | None
) -> dict[str, ParameterConstraint]:
    """Physical-domain constraints for one raw type/parameterization schema."""
    if parameterization is not None:
        spec = _PARAMETERIZATION_REGISTRY.get((kind, parameterization))
        return dict(spec.raw_constraints) if spec is not None else {}
    component_cls = get_component_class(kind)
    if component_cls is not None:
        return dict(component_cls._constraints)
    return {
        name: parameter.constraint
        for name, parameter in _SUPPORTED_GALAX_TYPES.get(kind, {}).items()
        if parameter.constraint is not None
    }


def parameter_schema_is_known(kind: str, parameterization: str | None) -> bool:
    """Whether `raw_parameter_dimensions(kind, parameterization)` is authoritative.

    `raw_parameter_dimensions` returns `{}` both for a recognized type/
    parameterization with a genuinely empty parameter schema and for anything
    unrecognized. `_validate_potential` uses this predicate to tell the two
    apart: it can run exact-name completeness checks on a component's declared
    `parameters` only when the schema is known, and must otherwise defer the
    whole question to `AbstractPotentialComponent.resolve`, which raises its
    own clearer error for an unsupported `type` or an unimplemented
    `parameterization`.
    """
    if parameterization is not None:
        return (kind, parameterization) in _PARAMETERIZATION_REGISTRY
    return is_registered_component_type(kind) or kind in _SUPPORTED_GALAX_TYPES
