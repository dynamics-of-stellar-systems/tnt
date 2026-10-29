"""Triaxial MGE-backed potential components.

`TriaxialLightMGEPotential`/`TriaxialMassMGEPotential` build a potential
from a named Multi-Gaussian Expansion, deprojected under global viewing
angles `theta`/`phi`/`psi` (see `_galax_potential_from_deprojected`).
The shared `build_with_validity` builder deprojects and checks the MGE
before any numerical use. Invalid proposals return a false flag and a zero
intrinsic placeholder. `to_galax` must only be called for valid proposals.

Both types also accept `parameterization: "pqu"`, replacing `theta/phi/psi`
with the intrinsic axis ratios `p = B/A`, `q = C/A` and the compression `u`
of the triaxial-Schwarzschild / DYNAMITE-successor literature (van den Bosch
et al. 2008, MNRAS 385, 647). `_pqu_to_tpp` shares its forward numerical
candidate with `AbstractMGE.triaxial_viewing_angles`; `_tpp_to_pqu` uses
`triaxial_intrinsic_shape_with_validity` for the inverse. The conversion is anchored at
`q' = min(component q)`.

`parameterization: "T_maj_min"` is a second option, over the same
`(p, q, u)` anchor but reparameterized as `(T, T_maj, T_min) in [0, 1]^3`
for more uniform shape/viewing-geometry sampling (Quenneville, Liepold & Ma
2022, ApJ 926:30, sec. 3). `_tmajmin_to_tpp` / `_tpp_to_tmajmin` are the
matching adapters, sharing the forward numerical candidate with
`AbstractMGE.viewing_angles_from_T_Tmaj_Tmin` and using
`T_Tmaj_Tmin_from_viewing_angles_with_validity` for the inverse. Both directions convert
through `(p, q, u)` and share its viewing-geometry rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import equinox as eqx
import galax.potential
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from unxt import AbstractUnitSystem, Quantity

from tnt.mge import (
    Deprojected3DMGE,
    LightMGE,
    MassMGE,
    _conversion_valid,
    _inverse_reporting_values,
)
from tnt.potential.components import AbstractMGEPotentialComponent
from tnt.potential.registry import (
    _VIEWING_ANGLES,
    InvalidPotentialParametersError,
    ParameterConstraint,
    register_component,
    register_default_prior,
    register_parameterization,
)


@register_component
class TriaxialLightMGEPotential(AbstractMGEPotentialComponent):
    """A triaxial potential from a light MGE, via its `ml` parameter.

    The shared builder converts the light MGE to mass via `ml`, deprojects it under
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

    _mge_type: ClassVar[type[LightMGE]] = LightMGE
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

    def to_galax(
        self, unit_system: AbstractUnitSystem
    ) -> galax.potential.AbstractPotential:
        return _galax_potential_from_deprojected(
            self.deprojected, unit_system, self.integration_order
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
class TriaxialMassMGEPotential(AbstractMGEPotentialComponent):
    """A triaxial potential from an already-mass-calibrated MGE.

    `mge_mass_scale` is the analogue of a light MGE's `ml` for a component
    whose shape template is already in mass units: a normalization on top
    of an otherwise-fixed mass map, typically left `fixed` (see `rescale`'s
    docstring for why it can still move regardless). `build_with_validity`/`to_galax`,
    and `parameterization: "pqu"`, otherwise follow the same path as
    `TriaxialLightMGEPotential` -- see its docstring.
    """

    _mge_type: ClassVar[type[MassMGE]] = MassMGE
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

    def to_galax(
        self, unit_system: AbstractUnitSystem
    ) -> galax.potential.AbstractPotential:
        return _galax_potential_from_deprojected(
            self.deprojected, unit_system, self.integration_order
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
    integration_order: int,
) -> galax.potential.AbstractPotential:
    """Sum one `galax.potential.TriaxialGaussianPotential` per Gaussian component.

    Converts an already-deprojected intrinsic MGE
    (`tnt.mge.AbstractMGE.deproject_triaxial`, called by the shared builder,
    before this) directly into a `galax` composite potential -- its
    density, `rho = rho_0 * exp(-xi^2/2)`, `xi^2 = x^2/r_s^2 + y^2/(q1
    r_s)^2 + z^2/(q2 r_s)^2`, matches `Deprojected3DMGE`'s own exactly (`r_s
    <-> sigma`, `q1 <-> p`, `q2 <-> q`), so equating the two central
    densities gives `m_tot = I * p * q * (2*pi)**1.5 * sigma**3` directly --
    no new potential formula, `galax` already owns that math.
    """
    n_components = deprojected.I.shape[0]
    masses = deprojected.component_masses
    widths = deprojected.gaussian_widths
    components = {
        str(i): galax.potential.TriaxialGaussianPotential(
            m_tot=masses[i],
            r_s=widths[i],
            q1=deprojected.p[i],
            q2=deprojected.q[i],
            units=unit_system,
            integration_order=integration_order,
        )
        for i in range(n_components)
    }
    return galax.potential.CompositePotential(components, units=unit_system)


# ==========================================================================
# The `pqu` parameterization: intrinsic axis ratios `(p, q, u)` of the MGE's
# flattest Gaussian <-> global viewing angles `(theta, phi, psi)`, matching
# DYNAMITE `triax_pqu2tpp` (van den Bosch et al. 2008, MNRAS 385, 647). Both
# directions -- including the anchor-twist bookkeeping -- live on
# `AbstractMGE` (`triaxial_viewing_angles` / `triaxial_intrinsic_shape` and
# their `with_validity` counterparts); the registry adapters use their shared
# numerical candidates and the inverse validity interface.

_PQU_SHAPE_CONSTRAINTS: dict[str, ParameterConstraint] = {
    "p": ParameterConstraint(
        minimum=0.0, minimum_inclusive=False, maximum=1.0, unit=""
    ),
    # 0 < q <= p
    "q": ParameterConstraint(
        minimum=0.0,
        minimum_inclusive=False,
        unit="",
        other_parameter="p",
        relation="<=",
    ),
    # p < u <= 1 (data-independent floor of DYNAMITE's
    # max(q/q', p) < u <= min(p/q', 1); the q'-dependent parts are checked
    # against the MGE in `AbstractMGE.triaxial_viewing_angles`)
    "u": ParameterConstraint(maximum=1.0, unit="", other_parameter="p", relation=">"),
}


def _mass_parameter_name(parameters: Mapping[str, Any]) -> str:
    """`"ml"` for the light type, `"mge_mass_scale"` for the mass type."""
    return "ml" if "ml" in parameters else "mge_mass_scale"


def _pqu_to_tpp(
    raw: dict[str, Quantity],
    cosmological_parameters: Mapping[str, Quantity],
    mge: LightMGE | MassMGE | None,
) -> dict[str, Quantity]:
    """Adapt `(p, q, u)` -> `(theta, phi, psi)` via the shared angle candidate.

    The data-independent `(p, q, u)` bounds (`0 < q <= p <= 1`, `p < u <= 1`)
    are enforced by the parameterization's `raw_constraints` before this runs;
    the MGE-dependent domain and singular geometries use the same predicates
    as `triaxial_viewing_angles`. Failed numerical conversions return nonfinite
    angles for the shared builder to reject before differentiable construction.
    """
    del cosmological_parameters
    if mge is None:  # unreachable: only the MGE composite types register `pqu`
        raise InvalidPotentialParametersError(
            "The 'pqu' parameterization requires an MGE component."
        )
    angles, checks = mge._viewing_angles_candidate(
        raw["p"].ustrip(""), raw["q"].ustrip(""), raw["u"].ustrip("")
    )
    theta, phi, psi = (
        Quantity(
            jnp.where(_conversion_valid(checks), angle.ustrip("rad"), jnp.nan), "rad"
        )
        for angle in angles
    )

    mass = _mass_parameter_name(raw)
    return {mass: raw[mass], "theta": theta, "phi": phi, "psi": psi}


def _tpp_to_pqu(
    native: dict[str, Quantity],
    declared_units: Mapping[str, str],
    cosmological_parameters: Mapping[str, Quantity],
    mge: LightMGE | MassMGE | None,
) -> dict[str, Quantity]:
    """Report native `(theta, phi, psi)` back as `(p, q, u)` for `AllModels`.

    Uses the shared traceable inverse of `triaxial_viewing_angles` (a boundary
    `u` comes back as the nudged value). Invalid eager reporting raises;
    invalid traced shapes are nonfinite.
    """
    del cosmological_parameters
    if mge is None:  # unreachable: only the MGE composite types register `pqu`
        raise InvalidPotentialParametersError(
            "The 'pqu' parameterization requires an MGE component."
        )
    shape, valid = mge.triaxial_intrinsic_shape_with_validity(
        native["theta"], native["phi"], native["psi"]
    )
    p, q, u = _inverse_reporting_values(shape, valid)
    mass = _mass_parameter_name(native)
    mass_value = native[mass]
    if mass in declared_units:
        mass_value = mass_value.to(declared_units[mass])
    return {
        mass: mass_value,
        "p": Quantity(p, "").to(declared_units.get("p", "")),
        "q": Quantity(q, "").to(declared_units.get("q", "")),
        "u": Quantity(u, "").to(declared_units.get("u", "")),
    }


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


def _pqu_default_prior_sample(
    site_prefix: str, known: dict[str, jax.Array], mge: LightMGE | MassMGE | None
) -> dict[str, jax.Array]:
    """`pqu`'s whole generative default prior, as one joint draw.

    The joint domain (ignoring precision margins) is `0 < q < p <= 1` and
    `max(q/q', p) < u <= min(p/q', 1)`, `q'` this MGE's own anchor axis
    ratio (`AbstractMGE._triaxial_anchor`). Sampled in `p`, `q`, `u` order,
    each one conditioned on *every* already-`known` sibling -- fixed or
    not, and regardless of which side of it in this order -- not just the
    ones sampled earlier in this call:

    - `p`'s lower bound is `q` if `q` is already known (`0` otherwise),
      additionally raised to `u * q'` if `u` is already known too; its
      upper bound is `u` if `u` is already known (`1` otherwise) -- all of
      which follow directly from the joint domain above (`p > q`;
      `u <= p/q'` i.e. `p >= u * q'`; `p < u` from the `max(...) < u` side).
    - `q`'s upper bound is `min(p, u * q')` if `u` is already known, else
      `min(p, q')` -- the latter is `pqu`'s own existence condition for
      *some* `u` to exist at all (bounding `q` by `q'` as well as by `p`
      is what keeps `u`'s domain reachable in the all-free case; working
      through the joint domain's four terms, every one but `q < q'`
      reduces to something already true by construction here -- `q < p`,
      `p < 1`, `q' < 1`); the former is the same condition once `u` is
      already fixed, with `q' ` replaced by the now-tighter `u * q'`.
    - `u`'s distribution is unchanged: `Uniform(u_lo, safe_hi)` from
      `AbstractMGE._triaxial_u_domain(p, q)`, which already only ever
      reads the final, fully-resolved `p`/`q` -- fixed or freshly
      sampled above, it was never order-sensitive to begin with.

    A degenerate fixed combination (e.g. a fixed `q` and `u` whose own
    `q/q' < u` already fails) can still make `p`'s own bounds invert; that
    draw is invalid regardless of what `p` turns out to be, so -- the same
    policy `u`'s own fallback already uses -- the upper bound is only
    widened enough to keep `Uniform` constructible, not corrected into a
    valid one. The one case this can't absorb, `u_lo >= u_hi` once `p`/`q`
    are resolved -- the domain too narrow to deproject reliably at this
    precision for this `(p, q)` and the MGE's own anchor (now only ever a
    genuine floating-point edge case, not the structural `q >= q'` case
    `q`'s own bound already rules out when `u` isn't fixed) -- isn't
    guarded here with a factor of its own either: that's caught downstream
    the same way any other invalid draw already is
    (`Potential.build_with_validity`'s own validity factor when a plugin
    touches `build_potential()`, or `Model.valid_potential` otherwise) --
    not a new gap, and keeping this prior pure `numpyro.sample` statements
    (no factor of its own) is what keeps the `Predictive` (no-factors)
    path usable for a component using only this default prior.
    """
    if mge is None:  # unreachable: only the MGE composite types register `pqu`
        raise InvalidPotentialParametersError(
            "The 'pqu' parameterization requires an MGE component."
        )
    q_obs, _ = mge._triaxial_anchor()
    eps = jnp.finfo(jnp.result_type(q_obs, float)).eps
    margin = 4 * jnp.sqrt(eps)

    p = known.get("p")
    if p is None:
        p_lower = known.get("q", 0.0)
        if "u" in known:
            p_lower = jnp.maximum(p_lower, known["u"] * q_obs)
            p_upper = known["u"]
        else:
            p_upper = 1.0
        p_upper = jnp.maximum(p_upper, p_lower + margin)
        p = numpyro.sample(f"{site_prefix}.p", dist.Uniform(p_lower, p_upper))

    q = known.get("q")
    if q is None:
        q_cap = known["u"] * q_obs if "u" in known else q_obs
        q_upper = jnp.maximum(jnp.minimum(p, q_cap), margin)
        q = numpyro.sample(f"{site_prefix}.q", dist.Uniform(0.0, q_upper))

    u = known.get("u")
    if u is None:
        _lo, _hi, u_lo, u_hi, u_margin = mge._triaxial_u_domain(p, q)
        safe_hi = jnp.maximum(u_hi, u_lo + u_margin)
        u = numpyro.sample(f"{site_prefix}.u", dist.Uniform(u_lo, safe_hi))
    return {"p": p, "q": q, "u": u}


def _register_pqu_default_prior(type_name: str) -> None:
    register_default_prior(
        type_name=type_name,
        parameterization="pqu",
        parameter_names=("p", "q", "u"),
        sample=_pqu_default_prior_sample,
    )


_register_pqu_default_prior("TriaxialLightMGEPotential")
_register_pqu_default_prior("TriaxialMassMGEPotential")


# ==========================================================================
# The `T_maj_min` parameterization: `(T, T_maj, T_min)`, a reparameterization
# of `pqu`'s own `(p, q, u)` chosen for more uniform sampling of shape and
# viewing geometry (Quenneville, Liepold & Ma 2022, ApJ 926:30, sec. 3), not
# a different deprojection -- same anchor, same underlying geometry. Both
# directions live on `AbstractMGE` (`viewing_angles_from_T_Tmaj_Tmin` /
# `T_Tmaj_Tmin_from_viewing_angles` and their `with_validity` counterparts).
# Beyond the `(T, T_maj, T_min)` <-> `(p, q, u)` algebra, they share the same
# viewing-angle and inverse-anchor candidates as the `pqu` methods.

_TMAJMIN_SHAPE_CONSTRAINTS: dict[str, ParameterConstraint] = {
    name: ParameterConstraint(minimum=0.0, maximum=1.0, unit="")
    for name in ("T", "T_maj", "T_min")
}


def _tmajmin_to_tpp(
    raw: dict[str, Quantity],
    cosmological_parameters: Mapping[str, Quantity],
    mge: LightMGE | MassMGE | None,
) -> dict[str, Quantity]:
    """Adapt ``(T, T_maj, T_min)`` -> ``(theta, phi, psi)`` via `AbstractMGE`.

    The data-independent `(T, T_maj, T_min)` bounds (each in `[0, 1]`) are
    enforced by the parameterization's `raw_constraints` before this runs;
    the MGE-dependent domain and every singular geometry are
    the shared numerical candidate's business. Failed numerical conversions return
    nonfinite angles for the shared builder to reject.
    """
    del cosmological_parameters
    if mge is None:  # unreachable: only the MGE composite types register this
        raise InvalidPotentialParametersError(
            "The 'T_maj_min' parameterization requires an MGE component."
        )
    angles, checks = mge._T_viewing_angles_candidate(
        raw["T"].ustrip(""), raw["T_maj"].ustrip(""), raw["T_min"].ustrip("")
    )
    theta, phi, psi = (
        Quantity(
            jnp.where(_conversion_valid(checks), angle.ustrip("rad"), jnp.nan), "rad"
        )
        for angle in angles
    )

    mass = _mass_parameter_name(raw)
    return {mass: raw[mass], "theta": theta, "phi": phi, "psi": psi}


def _tpp_to_tmajmin(
    native: dict[str, Quantity],
    declared_units: Mapping[str, str],
    cosmological_parameters: Mapping[str, Quantity],
    mge: LightMGE | MassMGE | None,
) -> dict[str, Quantity]:
    """Report native ``(theta, phi, psi)`` back as ``(T, T_maj, T_min)`` for
    `AllModels`.

    Uses the shared traceable inverse of `viewing_angles_from_T_Tmaj_Tmin`
    (a boundary value comes back nudged, as for `pqu`). Invalid eager reporting
    raises; invalid traced shapes are nonfinite.
    """
    del cosmological_parameters
    if mge is None:  # unreachable: only the MGE composite types register this
        raise InvalidPotentialParametersError(
            "The 'T_maj_min' parameterization requires an MGE component."
        )
    shape, valid = mge.T_Tmaj_Tmin_from_viewing_angles_with_validity(
        native["theta"], native["phi"], native["psi"]
    )
    T, T_maj, T_min = _inverse_reporting_values(shape, valid)
    mass = _mass_parameter_name(native)
    mass_value = native[mass]
    if mass in declared_units:
        mass_value = mass_value.to(declared_units[mass])
    return {
        mass: mass_value,
        "T": Quantity(T, "").to(declared_units.get("T", "")),
        "T_maj": Quantity(T_maj, "").to(declared_units.get("T_maj", "")),
        "T_min": Quantity(T_min, "").to(declared_units.get("T_min", "")),
    }


def _register_tmajmin(type_name: str, mass_name: str, mass_dimension: str) -> None:
    register_parameterization(
        type_name=type_name,
        name="T_maj_min",
        convert=_tmajmin_to_tpp,
        invert=_tpp_to_tmajmin,
        raw_dimensions={
            mass_name: mass_dimension,
            "T": "dimensionless",
            "T_maj": "dimensionless",
            "T_min": "dimensionless",
        },
        raw_constraints={
            mass_name: ParameterConstraint(minimum=0.0, minimum_inclusive=False),
            **_TMAJMIN_SHAPE_CONSTRAINTS,
        },
    )


_register_tmajmin("TriaxialLightMGEPotential", "ml", "mass_to_light")
_register_tmajmin("TriaxialMassMGEPotential", "mge_mass_scale", "dimensionless")
