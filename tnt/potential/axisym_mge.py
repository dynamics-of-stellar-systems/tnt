"""Axisymmetric MGE-backed potential components.

`AxisymmetricLightMGEPotential`/`AxisymmetricMassMGEPotential` build a
potential from a named Multi-Gaussian Expansion, deprojected under a single
`inclination` angle (`AbstractMGE.deproject_axisymmetric`) -- the
axisymmetric sibling of `tnt.potential.triaxial_mge`, for the common
oblate/prolate case where a full triaxial viewing-angle deprojection isn't
needed (or, per `AbstractMGE.deproject_axisymmetric`'s own requirement, an
MGE with zero `PA_twist` on every component, which a genuinely triaxial
system's isophote twist would rule out).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import equinox as eqx
import galax.potential
import jax.numpy as jnp
from unxt import AbstractUnitSystem, Quantity

from tnt.mge import LightMGE, MassMGE
from tnt.potential.components import AbstractPotentialComponent
from tnt.validation import _required_string, _resolve_typed_reference


class AxisymmetricLightMGEPotential(AbstractPotentialComponent):
    """An axisymmetric potential from a light MGE, via its `ml` parameter.

    `to_galax` converts the light MGE to mass via `ml`, deprojects it under
    the single `inclination` angle (`AbstractMGE.deproject_axisymmetric`),
    and sums one `galax.potential.AxisymmetricGaussianPotential` per
    Gaussian component (see `_axisym_mge_potential`). `inclination` is this
    component's native viewing-geometry parameter.
    """

    _type: ClassVar[str] = "AxisymmetricLightMGEPotential"
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
        mass_mge = self.mge.to_mass(self.parameters["ml"])
        return _axisym_mge_potential(mass_mge, self.parameters, unit_system)

    def rescale(self, mass_scale: float) -> Self:
        rescaled = dict(self.parameters)
        rescaled["ml"] = rescaled["ml"] * mass_scale
        return eqx.tree_at(lambda c: c.parameters, self, rescaled)


class AxisymmetricMassMGEPotential(AbstractPotentialComponent):
    """An axisymmetric potential from an already-mass-calibrated MGE.

    `mge_mass_scale` is the analogue of a light MGE's `ml` for a component
    whose shape template is already in mass units (see
    `tnt.potential.triaxial_mge.TriaxialMassMGEPotential`'s docstring for
    why it can still move under `rescale` even when `fixed`). `to_galax`
    otherwise follows the same path as `AxisymmetricLightMGEPotential` --
    see its docstring.
    """

    _type: ClassVar[str] = "AxisymmetricMassMGEPotential"
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
        mass_mge = self.mge.rescaled(self.parameters["mge_mass_scale"])
        return _axisym_mge_potential(mass_mge, self.parameters, unit_system)

    def rescale(self, mass_scale: float) -> Self:
        rescaled = dict(self.parameters)
        rescaled["mge_mass_scale"] = rescaled["mge_mass_scale"] * mass_scale
        return eqx.tree_at(lambda c: c.parameters, self, rescaled)


def _axisym_mge_potential(
    mass_mge: MassMGE,
    parameters: Mapping[str, Quantity],
    unit_system: AbstractUnitSystem,
) -> galax.potential.AbstractPotential:
    """Sum one `galax.potential.AxisymmetricGaussianPotential` per Gaussian component.

    Deprojects `mass_mge` under a single `inclination` angle
    (`tnt.mge.AbstractMGE.deproject_axisymmetric`), which always gives an
    intrinsic `p = 1` -- so unlike
    `tnt.potential.triaxial_mge._triaxial_mge_potential`, each resulting
    Gaussian needs only `q2` (not `q1`), matching
    `galax.potential.AxisymmetricGaussianPotential` exactly (`r_s <->
    sigma`, `q2 <-> q`); the same central-density equivalence as the
    triaxial case gives `m_tot = I * q * (2*pi)**1.5 * sigma**3` (`p == 1`
    drops out of the triaxial `m_tot = I * p * q * (2*pi)**1.5 * sigma**3`).
    Verified to agree with `TriaxialGaussianPotential(q1=1, q2=q, ...)` to
    float32 precision.
    """
    deprojected = mass_mge.deproject_axisymmetric(parameters["inclination"])
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
