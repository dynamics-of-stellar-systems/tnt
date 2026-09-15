"""Oblate axisymmetric MGE-backed potential components.

`OblateLightMGEPotential`/`OblateMassMGEPotential` build a potential from a
named Multi-Gaussian Expansion, deprojected under a single `inclination`
angle (`AbstractMGE.deproject_oblate`) -- the oblate-axisymmetric sibling of
`tnt.potential.triaxial_mge`. TNT's axisymmetric deprojection is the oblate
convention only (`p = B/A = 1`, `q = C/A <= 1`); a prolate spheroid, whose
long axis is the symmetry axis, needs a different relation and would get its
own `Prolate...` types (tracked as a non-urgent follow-up). Deprojection
happens once, in `_build` -- not lazily inside `to_galax()` -- so an invalid
inclination (`tnt.mge.MGEDeprojectionError`, or a `ValueError` for an
inclination outside `(0, 90]` deg), or an MGE with nonzero `PA_twist`
(`ValueError` -- an axisymmetric system has no isophote twist), surfaces
right there, before anything downstream is attempted. Same rationale and
shape as `triaxial_mge`; kept separate so each module is about one
deprojection convention.

Both types also accept `parameterization: "q_min"`, replacing `inclination`
with the intrinsic axial ratio of the flattest observed (anchor) Gaussian
component, `q_min <= q' = min(component q)`. `_qmin_to_inclination` /
`_inclination_to_qmin` are thin adapters over
`AbstractMGE.inclination_from_q_min` / `q_min_from_inclination`, which own
the conversion and its inverse -- the oblate counterpart of
`triaxial_mge`'s `pqu`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import equinox as eqx
import galax.potential
import jax.numpy as jnp
from unxt import AbstractUnitSystem, Quantity

from tnt.mge import Deprojected3DMGE, LightMGE, MassMGE, MGEDeprojectionError
from tnt.potential.components import AbstractPotentialComponent
from tnt.potential.registry import (
    InvalidPotentialParametersError,
    ParameterConstraint,
    register_component,
    register_parameterization,
)
from tnt.validation import _required_string, _resolve_typed_reference

# The single viewing angle an oblate axisymmetric MGE is deprojected under
# (`tnt.mge.AbstractMGE.deproject_oblate`), in place of the triaxial module's
# `theta`/`phi`/`psi` -- required regardless of light vs. mass.
_INCLINATION: dict[str, str] = {"inclination": "angle"}
_INCLINATION_CONSTRAINT = ParameterConstraint(
    minimum=0.0,
    minimum_inclusive=False,
    maximum=90.0,
    unit="deg",
)


@register_component
class OblateLightMGEPotential(AbstractPotentialComponent):
    """An oblate axisymmetric potential from a light MGE, via its `ml` parameter.

    `_build` converts the light MGE to mass via `ml`, deprojects it under
    the single `inclination` angle (`AbstractMGE.deproject_oblate`), and
    stores the result as `deprojected`; `to_galax` sums one
    `galax.potential.AxisymmetricGaussianPotential` per Gaussian component
    from it (see `_galax_potential_from_oblate_deprojected`). `inclination`
    is this component's native viewing-geometry parameter.
    """

    _type: ClassVar[str] = "OblateLightMGEPotential"
    _raw_dimensions: ClassVar[dict[str, str]] = {
        "ml": "mass_to_light",
        **_INCLINATION,
    }
    _constraints: ClassVar[dict[str, ParameterConstraint]] = {
        "ml": ParameterConstraint(minimum=0.0, minimum_inclusive=False),
        "inclination": _INCLINATION_CONSTRAINT,
    }
    mge: LightMGE
    deprojected: Deprojected3DMGE

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

    @classmethod
    def _build(
        cls,
        parameters: dict[str, Quantity],
        cosmological_parameters: Mapping[str, Quantity],
        extra_fields: dict[str, Any],
    ) -> Self:
        del cosmological_parameters
        mge = extra_fields["mge"]
        mass_mge = mge.to_mass(parameters["ml"])
        deprojected = mass_mge.deproject_oblate(parameters["inclination"])
        return cls(parameters=parameters, mge=mge, deprojected=deprojected)

    def to_galax(
        self, unit_system: AbstractUnitSystem
    ) -> galax.potential.AbstractPotential:
        return _galax_potential_from_oblate_deprojected(self.deprojected, unit_system)

    def rescale(self, mass_scale: float) -> Self:
        rescaled_parameters = dict(self.parameters)
        rescaled_parameters["ml"] = rescaled_parameters["ml"] * mass_scale
        rescaled_deprojected = eqx.tree_at(
            lambda d: d.I, self.deprojected, self.deprojected.I * mass_scale
        )
        return eqx.tree_at(
            lambda c: (c.parameters, c.deprojected),
            self,
            (rescaled_parameters, rescaled_deprojected),
        )


@register_component
class OblateMassMGEPotential(AbstractPotentialComponent):
    """An oblate axisymmetric potential from an already-mass-calibrated MGE.

    `mge_mass_scale` is the analogue of a light MGE's `ml` for a component
    whose shape template is already in mass units (see
    `tnt.potential.triaxial_mge.TriaxialMassMGEPotential` for why it can
    still move under `rescale` even when `fixed`). `_build`/`to_galax`
    otherwise follow the same path as `OblateLightMGEPotential` -- see its
    docstring.
    """

    _type: ClassVar[str] = "OblateMassMGEPotential"
    _raw_dimensions: ClassVar[dict[str, str]] = {
        "mge_mass_scale": "dimensionless",
        **_INCLINATION,
    }
    _constraints: ClassVar[dict[str, ParameterConstraint]] = {
        "mge_mass_scale": ParameterConstraint(minimum=0.0, minimum_inclusive=False),
        "inclination": _INCLINATION_CONSTRAINT,
    }
    mge: MassMGE
    deprojected: Deprojected3DMGE

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

    @classmethod
    def _build(
        cls,
        parameters: dict[str, Quantity],
        cosmological_parameters: Mapping[str, Quantity],
        extra_fields: dict[str, Any],
    ) -> Self:
        del cosmological_parameters
        mge = extra_fields["mge"]
        mass_mge = mge.rescaled(parameters["mge_mass_scale"])
        deprojected = mass_mge.deproject_oblate(parameters["inclination"])
        return cls(parameters=parameters, mge=mge, deprojected=deprojected)

    def to_galax(
        self, unit_system: AbstractUnitSystem
    ) -> galax.potential.AbstractPotential:
        return _galax_potential_from_oblate_deprojected(self.deprojected, unit_system)

    def rescale(self, mass_scale: float) -> Self:
        rescaled_parameters = dict(self.parameters)
        rescaled_parameters["mge_mass_scale"] = (
            rescaled_parameters["mge_mass_scale"] * mass_scale
        )
        rescaled_deprojected = eqx.tree_at(
            lambda d: d.I, self.deprojected, self.deprojected.I * mass_scale
        )
        return eqx.tree_at(
            lambda c: (c.parameters, c.deprojected),
            self,
            (rescaled_parameters, rescaled_deprojected),
        )


def _galax_potential_from_oblate_deprojected(
    deprojected: Deprojected3DMGE,
    unit_system: AbstractUnitSystem,
) -> galax.potential.AbstractPotential:
    """Sum one `galax.potential.AxisymmetricGaussianPotential` per Gaussian component.

    An oblate axisymmetric deprojection (`tnt.mge.AbstractMGE.deproject_oblate`,
    called once in `_build`, before this) always gives intrinsic `p = 1`, so
    each Gaussian needs only `q2` -- `galax.potential.AxisymmetricGaussianPotential`
    is exactly the `q1 = 1` special case of the `TriaxialGaussianPotential`
    that `tnt.potential.triaxial_mge._galax_potential_from_deprojected` uses
    (`r_s <-> sigma`, `q2 <-> q`). Equating central densities gives
    `m_tot = I * q * (2*pi)**1.5 * sigma**3` (`p == 1` drops out of the
    triaxial `m_tot = I * p * q * (2*pi)**1.5 * sigma**3`).
    """
    n_components = deprojected.I.shape[0]
    components = {
        str(i): galax.potential.AxisymmetricGaussianPotential(
            m_tot=deprojected.I[i]
            * deprojected.q[i]
            * (2 * jnp.pi) ** 1.5
            * deprojected.sigma[i] ** 3,
            r_s=deprojected.sigma[i],
            q2=deprojected.q[i],
            units=unit_system,
        )
        for i in range(n_components)
    }
    return galax.potential.CompositePotential(components, units=unit_system)


# ==========================================================================
# The `q_min` parameterization: the anchor Gaussian's intrinsic axial ratio
# <-> the single global inclination, the oblate counterpart of
# `triaxial_mge`'s `pqu`. Both directions live on `AbstractMGE`
# (`inclination_from_q_min` / `q_min_from_inclination`); these converters are
# just the parameterization-registry adapters.

_QMIN_CONSTRAINT = ParameterConstraint(
    minimum=0.0, minimum_inclusive=False, maximum=1.0
)


def _mass_parameter_name(parameters: Mapping[str, Any]) -> str:
    """`"ml"` for the light type, `"mge_mass_scale"` for the mass type."""
    return "ml" if "ml" in parameters else "mge_mass_scale"


def _qmin_to_inclination(
    raw: dict[str, Quantity],
    cosmological_parameters: Mapping[str, Quantity],
    mge: LightMGE | MassMGE | None,
) -> dict[str, Quantity]:
    """Adapt ``q_min`` -> ``inclination`` via `AbstractMGE.inclination_from_q_min`.

    The data-independent `q_min` bound (`0 < q_min <= 1`) is enforced by the
    parameterization's `raw_constraints` before this runs; the MGE-dependent
    domain and every singular geometry are `inclination_from_q_min`'s
    business, re-raised here as `InvalidPotentialParametersError` so
    `ModelIterator` records an invalid model rather than crashing.
    """
    del cosmological_parameters
    if mge is None:  # unreachable: only the MGE composite types register this
        raise InvalidPotentialParametersError(
            "The 'q_min' parameterization requires an MGE component."
        )
    try:
        inclination = mge.inclination_from_q_min(float(raw["q_min"].ustrip("")))
    except MGEDeprojectionError as error:
        raise InvalidPotentialParametersError(str(error)) from error

    mass = _mass_parameter_name(raw)
    return {mass: raw[mass], "inclination": inclination}


def _inclination_to_qmin(
    native: dict[str, Quantity],
    declared_units: Mapping[str, str],
    cosmological_parameters: Mapping[str, Quantity],
    mge: LightMGE | MassMGE | None,
) -> dict[str, Quantity]:
    """Report native ``inclination`` back as ``q_min`` for `AllModels`.

    `AbstractMGE.q_min_from_inclination` -- the numerical inverse of
    `inclination_from_q_min`.
    """
    del cosmological_parameters
    if mge is None:  # unreachable: only the MGE composite types register this
        raise InvalidPotentialParametersError(
            "The 'q_min' parameterization requires an MGE component."
        )
    q_min = mge.q_min_from_inclination(native["inclination"])
    mass = _mass_parameter_name(native)
    mass_value = native[mass]
    if mass in declared_units:
        mass_value = mass_value.to(declared_units[mass])
    return {mass: mass_value, "q_min": Quantity(q_min, "")}


def _register_q_min(type_name: str, mass_name: str, mass_dimension: str) -> None:
    register_parameterization(
        type_name=type_name,
        name="q_min",
        convert=_qmin_to_inclination,
        invert=_inclination_to_qmin,
        raw_dimensions={mass_name: mass_dimension, "q_min": "dimensionless"},
        raw_constraints={
            mass_name: ParameterConstraint(minimum=0.0, minimum_inclusive=False),
            "q_min": _QMIN_CONSTRAINT,
        },
    )


_register_q_min("OblateLightMGEPotential", "ml", "mass_to_light")
_register_q_min("OblateMassMGEPotential", "mge_mass_scale", "dimensionless")
