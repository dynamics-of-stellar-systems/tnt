"""Triaxial MGE-backed potential components.

`TriaxialLightMGEPotential`/`TriaxialMassMGEPotential` build a potential
from a named Multi-Gaussian Expansion, deprojected under global viewing
angles `theta`/`phi`/`psi` (see `_galax_potential_from_deprojected`).
Deprojection happens once, in `_build`, when the component is constructed
from a proposed point in parameter space -- not lazily inside `to_galax()`
-- so an invalid viewing geometry (`tnt.mge.MGEDeprojectionError`) surfaces
right there, before anything downstream (like orbit integration) is
attempted. Axisymmetric counterparts, built on
`galax.potential.AxisymmetricGaussianPotential`, are a planned addition --
kept in their own module rather than this one once they land, so this one
stays specifically about the triaxial case.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import equinox as eqx
import galax.potential
import jax.numpy as jnp
from unxt import AbstractUnitSystem, Quantity

from tnt.mge import Deprojected3DMGE, LightMGE, MassMGE
from tnt.potential.components import AbstractPotentialComponent
from tnt.potential.registry import _VIEWING_ANGLES, register_component
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
    this component's native viewing-geometry parameters; the
    shape/compression `(p, q, u)` parameterization closer to the
    triaxial-Schwarzschild-modeling literature isn't registered yet --
    converting it to `(theta, phi, psi)` needs a formula that hasn't been
    confirmed (see `docs/source/potential.md`).
    """

    _type: ClassVar[str] = "TriaxialLightMGEPotential"
    _raw_dimensions: ClassVar[dict[str, str]] = {
        "ml": "mass_to_light",
        **_VIEWING_ANGLES,
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
        unit_system: AbstractUnitSystem,
        cosmological_parameters: Mapping[str, Quantity],
        extra_fields: dict[str, Any],
    ) -> Self:
        del unit_system, cosmological_parameters
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
    docstring for why it can still move regardless). `_build`/`to_galax`
    otherwise follow the same path as `TriaxialLightMGEPotential` -- see its
    docstring.
    """

    _type: ClassVar[str] = "TriaxialMassMGEPotential"
    _raw_dimensions: ClassVar[dict[str, str]] = {
        "mge_mass_scale": "dimensionless",
        **_VIEWING_ANGLES,
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
        unit_system: AbstractUnitSystem,
        cosmological_parameters: Mapping[str, Quantity],
        extra_fields: dict[str, Any],
    ) -> Self:
        del unit_system, cosmological_parameters
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
