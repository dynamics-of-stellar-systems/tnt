"""Triaxial MGE-backed potential components.

`TriaxialLightMGEPotential`/`TriaxialMassMGEPotential` build a potential
from a named Multi-Gaussian Expansion, deprojected under global viewing
angles `theta`/`phi`/`psi` (see `_galax_potential_from_deprojected`).
Deprojection happens once, in `_build`, when the component is constructed
from a proposed point in parameter space -- not lazily inside `to_galax()`
-- so an invalid viewing geometry (`tnt.mge.MGEDeprojectionError`) surfaces
right there, before anything downstream (like orbit integration) is
attempted. The oblate axisymmetric counterparts, built on
`galax.potential.AxisymmetricGaussianPotential`, live in the sibling
`tnt.potential.oblate_mge` module, so this one stays specifically about the
triaxial case.

Both types also accept `parameterization: "pqu"`, replacing `theta/phi/psi`
with the intrinsic axis ratios `p = B/A`, `q = C/A` and the compression `u`
of the triaxial-Schwarzschild / DYNAMITE-successor literature (van den Bosch
et al. 2008, MNRAS 385, 647). `_pqu_to_tpp` converts `(p, q, u)` to the
native viewing angles at `q' = min(component q)` and zero twist -- the
standard anchor; `_tpp_to_pqu` reports back the same way for `AllModels`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar, Self

import equinox as eqx
import galax.potential
import jax.numpy as jnp
from unxt import AbstractUnitSystem, Quantity

from tnt.mge import (
    Deprojected3DMGE,
    LightMGE,
    MassMGE,
    _triaxial_intrinsic_axis_ratios,
)
from tnt.potential.components import AbstractPotentialComponent
from tnt.potential.registry import (
    _VIEWING_ANGLES,
    InvalidPotentialParametersError,
    ParameterConstraint,
    get_parameterization,
    register_component,
    register_parameterization,
)
from tnt.validation import _required_string, _resolve_typed_reference


@register_component
class TriaxialLightMGEPotential(AbstractPotentialComponent):
    """A triaxial potential from a light MGE, via its `ml` parameter.

    `_build` converts the light MGE to mass via `ml`, deprojects it under
    the shared, global viewing angles `theta`/`phi`/`psi`
    (`AbstractMGE.deproject_triaxial`), and stores the result as
    `deprojected`; `to_galax` sums one
    `galax.potential.TriaxialGaussianPotential` per Gaussian component from
    it (see `_galax_potential_from_deprojected`). `(theta, phi, psi)` are
    this component's native viewing-geometry parameters;
    `parameterization: "pqu"` accepts the intrinsic shape/compression
    `(p, q, u)` instead (see this module's docstring and
    `docs/source/potential.md`).
    """

    _type: ClassVar[str] = "TriaxialLightMGEPotential"
    _raw_dimensions: ClassVar[dict[str, str]] = {
        "ml": "mass_to_light",
        **_VIEWING_ANGLES,
    }
    _constraints: ClassVar[dict[str, ParameterConstraint]] = {
        "ml": ParameterConstraint(minimum=0.0, minimum_inclusive=False)
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
        deprojected = mass_mge.deproject_triaxial(
            parameters["theta"], parameters["phi"], parameters["psi"]
        )
        return cls(parameters=parameters, mge=mge, deprojected=deprojected)

    def to_galax(
        self, unit_system: AbstractUnitSystem
    ) -> galax.potential.AbstractPotential:
        return _galax_potential_from_deprojected(self.deprojected, unit_system)

    def raw_parameters(
        self,
        parameterization: str | None,
        declared_units: Mapping[str, str],
        cosmological_parameters: Mapping[str, Quantity],
    ) -> dict[str, Quantity]:
        return _mge_raw_parameters(
            self, parameterization, declared_units, cosmological_parameters
        )

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
class TriaxialMassMGEPotential(AbstractPotentialComponent):
    """A triaxial potential from an already-mass-calibrated MGE.

    `mge_mass_scale` is the analogue of a light MGE's `ml` for a component
    whose shape template is already in mass units: a normalization on top
    of an otherwise-fixed mass map, typically left `fixed` (see `rescale`'s
    docstring for why it can still move regardless). `_build`/`to_galax`,
    and `parameterization: "pqu"`, otherwise follow the same path as
    `TriaxialLightMGEPotential` -- see its docstring.
    """

    _type: ClassVar[str] = "TriaxialMassMGEPotential"
    _raw_dimensions: ClassVar[dict[str, str]] = {
        "mge_mass_scale": "dimensionless",
        **_VIEWING_ANGLES,
    }
    _constraints: ClassVar[dict[str, ParameterConstraint]] = {
        "mge_mass_scale": ParameterConstraint(minimum=0.0, minimum_inclusive=False)
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
        deprojected = mass_mge.deproject_triaxial(
            parameters["theta"], parameters["phi"], parameters["psi"]
        )
        return cls(parameters=parameters, mge=mge, deprojected=deprojected)

    def to_galax(
        self, unit_system: AbstractUnitSystem
    ) -> galax.potential.AbstractPotential:
        return _galax_potential_from_deprojected(self.deprojected, unit_system)

    def raw_parameters(
        self,
        parameterization: str | None,
        declared_units: Mapping[str, str],
        cosmological_parameters: Mapping[str, Quantity],
    ) -> dict[str, Quantity]:
        return _mge_raw_parameters(
            self, parameterization, declared_units, cosmological_parameters
        )

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


def _galax_potential_from_deprojected(
    deprojected: Deprojected3DMGE,
    unit_system: AbstractUnitSystem,
) -> galax.potential.AbstractPotential:
    """Sum one `galax.potential.TriaxialGaussianPotential` per Gaussian component.

    Converts an already-deprojected intrinsic MGE
    (`tnt.mge.AbstractMGE.deproject_triaxial`, called once in `_build`,
    before this) directly into a `galax` composite potential -- its
    density, `rho = rho_0 * exp(-xi^2/2)`, `xi^2 = x^2/r_s^2 + y^2/(q1
    r_s)^2 + z^2/(q2 r_s)^2`, matches `Deprojected3DMGE`'s own exactly (`r_s
    <-> sigma`, `q1 <-> p`, `q2 <-> q`), so equating the two central
    densities gives `m_tot = I * p * q * (2*pi)**1.5 * sigma**3` directly --
    no new potential formula, `galax` already owns that math.
    """
    n_components = deprojected.I.shape[0]
    components = {
        str(i): galax.potential.TriaxialGaussianPotential(
            m_tot=deprojected.I[i]
            * deprojected.p[i]
            * deprojected.q[i]
            * (2 * jnp.pi) ** 1.5
            * deprojected.sigma[i] ** 3,
            r_s=deprojected.sigma[i],
            q1=deprojected.p[i],
            q2=deprojected.q[i],
            units=unit_system,
        )
        for i in range(n_components)
    }
    return galax.potential.CompositePotential(components, units=unit_system)


# ==========================================================================
# The `pqu` parameterization: intrinsic axis ratios `(p, q, u)` <-> viewing
# angles `(theta, phi, psi)`. `p = B/A`, `q = C/A`, `u` the scale-length
# compression; anchored at `q' = min(component q)` with zero twist, matching
# DYNAMITE `triax_pqu2tpp` (van den Bosch et al. 2008, MNRAS 385, 647). The
# reverse direction is the same van den Bosch math `deproject_triaxial`
# already uses, factored into `tnt.mge._triaxial_intrinsic_axis_ratios`.

_PQU_SHAPE_CONSTRAINTS: dict[str, ParameterConstraint] = {
    "p": ParameterConstraint(minimum=0.0, minimum_inclusive=False, maximum=1.0),
    # 0 < q <= p
    "q": ParameterConstraint(
        minimum=0.0, minimum_inclusive=False, other_parameter="p", relation="<="
    ),
    # p < u <= 1 (data-independent floor of DYNAMITE's
    # max(q/q', p) < u <= min(p/q', 1); the q'-dependent parts are checked
    # against the MGE in `_pqu_to_tpp`)
    "u": ParameterConstraint(maximum=1.0, other_parameter="p", relation=">"),
}


def _mass_parameter_name(parameters: Mapping[str, Any]) -> str:
    """`"ml"` for the light type, `"mge_mass_scale"` for the mass type."""
    return "ml" if "ml" in parameters else "mge_mass_scale"


def _mge_min_observed_q(mge: LightMGE | MassMGE) -> float:
    """`q' = min` component observed axial ratio -- the `(p, q, u)` anchor."""
    return float(jnp.min(mge.q.ustrip("")))


def _pqu_to_tpp(
    raw: dict[str, Quantity],
    cosmological_parameters: Mapping[str, Quantity],
    mge: LightMGE | MassMGE | None,
) -> dict[str, Quantity]:
    """Convert `(p, q, u)` to the native viewing angles `(theta, phi, psi)`.

    van den Bosch et al. 2008 (MNRAS 385, 647) / DYNAMITE `triax_pqu2tpp`,
    with `q' = min(component q)` and zero twist. `p, q, u` bounds that don't
    involve the MGE (`0 < q <= p <= 1`, `p < u <= 1`) are already enforced by
    the parameterization's `raw_constraints` before this runs. Checked here,
    raising `InvalidPotentialParametersError`: a circular MGE (`q' = 1`), the
    prolate limit `q == p`, the `q'`-dependent domain
    `max(q/q', p) < u <= min(p/q', 1)`, and any degenerate weight. `u == 1`
    (intrinsic major axis exactly in the sky plane) is valid -- handled by its
    analytic limit `phi = psi = pi/2`, where DYNAMITE nudges `u` down one ULP.
    """
    del cosmological_parameters
    if mge is None:  # unreachable: only the MGE composite types register `pqu`
        raise InvalidPotentialParametersError(
            "The 'pqu' parameterization requires an MGE component."
        )
    p = float(raw["p"].ustrip(""))
    q = float(raw["q"].ustrip(""))
    u = float(raw["u"].ustrip(""))
    q_obs = _mge_min_observed_q(mge)
    if not q_obs < 1.0:
        raise InvalidPotentialParametersError(
            "The 'pqu' parameterization needs a genuinely flattened MGE "
            f"(min observed q' = {q_obs:g}); a circular MGE has no triaxial "
            "viewing geometry to solve for."
        )
    if q >= p:  # prolate limit: no unique triaxial viewing geometry
        raise InvalidPotentialParametersError(
            f"(p, q, u) = ({p:g}, {q:g}, {u:g}): q == p (prolate) has no unique "
            "triaxial viewing geometry."
        )

    lo = max(q / q_obs, p)
    hi = min(p / q_obs, 1.0)
    if not lo < u <= hi:
        raise InvalidPotentialParametersError(
            f"(p, q, u) = ({p:g}, {q:g}, {u:g}) has no triaxial deprojection for "
            f"this MGE (min observed q' = {q_obs:g}): requires "
            f"max(q/q', p) = {lo:g} < u <= min(p/q', 1) = {hi:g}."
        )

    p2, q2, u2, o2 = p * p, q * q, u * u, q_obs * q_obs
    w1 = (u2 - q2) * (o2 * u2 - q2) / ((1.0 - q2) * (p2 - q2))
    if not 0.0 <= w1 <= 1.0:
        raise InvalidPotentialParametersError(
            f"(p, q, u) = ({p:g}, {q:g}, {u:g}), q' = {q_obs:g}: degenerate "
            f"viewing geometry (theta weight w1 = {w1:g} outside [0, 1])."
        )
    theta = math.acos(math.sqrt(w1))

    if u == 1.0:
        # Intrinsic major axis exactly in the sky plane: the (1 - u^2)
        # denominators in w2/w3 vanish, and the analytic u -> 1 limit is
        # phi = psi = pi/2 (DYNAMITE nudges u down one ULP to the same end).
        phi = psi = math.pi / 2
    else:
        w2 = (
            (u2 - p2)
            * (p2 - o2 * u2)
            * (1.0 - q2)
            / ((1.0 - u2) * (1.0 - o2 * u2) * (p2 - q2))
        )
        w3 = (
            (1.0 - o2 * u2)
            * (p2 - o2 * u2)
            * (u2 - q2)
            / ((1.0 - u2) * (u2 - p2) * (o2 * u2 - q2))
        )
        if not (w2 >= 0.0 and w3 >= 0.0):
            raise InvalidPotentialParametersError(
                f"(p, q, u) = ({p:g}, {q:g}, {u:g}), q' = {q_obs:g}: degenerate "
                f"viewing geometry (phi/psi weights w2 = {w2:g}, w3 = {w3:g})."
            )
        phi = math.atan(math.sqrt(w2))
        psi = math.pi - math.atan(math.sqrt(w3))

    mass = _mass_parameter_name(raw)
    return {
        mass: raw[mass],
        "theta": Quantity(theta, "rad"),
        "phi": Quantity(phi, "rad"),
        "psi": Quantity(psi, "rad"),
    }


def _tpp_to_pqu(
    native: dict[str, Quantity],
    declared_units: Mapping[str, str],
    cosmological_parameters: Mapping[str, Quantity],
    mge: LightMGE | MassMGE | None,
) -> dict[str, Quantity]:
    """Report native `(theta, phi, psi)` back as `(p, q, u)` for `AllModels`.

    The same van den Bosch 2008 relation `deproject_triaxial` uses
    (`tnt.mge._triaxial_intrinsic_axis_ratios`), evaluated at the anchor
    `q' = min(component q)` with zero twist -- the exact inverse of
    `_pqu_to_tpp`.
    """
    del cosmological_parameters
    if mge is None:  # unreachable: only the MGE composite types register `pqu`
        raise InvalidPotentialParametersError(
            "The 'pqu' parameterization requires an MGE component."
        )
    q_obs = _mge_min_observed_q(mge)
    p, q, u = _triaxial_intrinsic_axis_ratios(
        native["theta"].ustrip("rad"),
        native["phi"].ustrip("rad"),
        native["psi"].ustrip("rad"),
        q_obs,
    )
    mass = _mass_parameter_name(native)
    mass_value = native[mass]
    if mass in declared_units:
        mass_value = mass_value.to(declared_units[mass])
    return {
        mass: mass_value,
        "p": Quantity(p, ""),
        "q": Quantity(q, ""),
        "u": Quantity(u, ""),
    }


def _mge_raw_parameters(
    component: AbstractPotentialComponent,
    parameterization: str | None,
    declared_units: Mapping[str, str],
    cosmological_parameters: Mapping[str, Quantity],
) -> dict[str, Quantity]:
    """Shared `raw_parameters` body for both triaxial MGE composite types."""
    if parameterization is None:
        return component.parameters
    spec = get_parameterization(component._type, parameterization)
    if spec is None:  # unreachable: resolve() already validated it
        raise NotImplementedError(
            f"{component._type}.{parameterization!r} is not a registered "
            "parameterization."
        )
    return spec.invert(
        component.parameters,
        declared_units,
        cosmological_parameters,
        component.mge,
    )


def _register_pqu(type_name: str, mass_name: str, mass_dimension: str) -> None:
    register_parameterization(
        type_name=type_name,
        name="pqu",
        convert=_pqu_to_tpp,
        invert=_tpp_to_pqu,
        raw_dimensions={
            mass_name: mass_dimension,
            "p": "dimensionless",
            "q": "dimensionless",
            "u": "dimensionless",
        },
        raw_constraints={
            mass_name: ParameterConstraint(minimum=0.0, minimum_inclusive=False),
            **_PQU_SHAPE_CONSTRAINTS,
        },
    )


_register_pqu("TriaxialLightMGEPotential", "ml", "mass_to_light")
_register_pqu("TriaxialMassMGEPotential", "mge_mass_scale", "dimensionless")
