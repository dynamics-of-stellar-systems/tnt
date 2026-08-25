"""Triaxial MGE-backed potential components.

`TriaxialLightMGEPotential`/`TriaxialMassMGEPotential` build a potential
from a named Multi-Gaussian Expansion, deprojected under global viewing
angles `theta`/`phi`/`psi` (see `_triaxial_mge_potential`). Axisymmetric
counterparts, built on `galax.potential.AxisymmetricGaussianPotential`, are
a planned addition -- kept in their own module rather than this one once
they land, so this one stays specifically about the triaxial case.
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


class TriaxialLightMGEPotential(AbstractPotentialComponent):
    """A triaxial potential from a light MGE, via its `ml` parameter.

    `to_galax` converts the light MGE to mass via `ml`, deprojects it under
    the shared, global viewing angles `theta`/`phi`/`psi`
    (`AbstractMGE.deproject_triaxial`), and sums one
    `galax.potential.TriaxialGaussianPotential` per Gaussian component (see
    `_triaxial_mge_potential`). `(theta, phi, psi)` are this component's
    native viewing-geometry parameters; the shape/compression `(p, q, u)`
    parameterization closer to the triaxial-Schwarzschild-modeling
    literature isn't registered yet -- converting it to `(theta, phi, psi)`
    needs a formula that hasn't been confirmed (see
    `docs/source/potential.md`).
    """

    _type: ClassVar[str] = "TriaxialLightMGEPotential"
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
        return _triaxial_mge_potential(mass_mge, self.parameters, unit_system)

    def rescale(self, mass_scale: float) -> Self:
        rescaled = dict(self.parameters)
        rescaled["ml"] = rescaled["ml"] * mass_scale
        return eqx.tree_at(lambda c: c.parameters, self, rescaled)


class TriaxialMassMGEPotential(AbstractPotentialComponent):
    """A triaxial potential from an already-mass-calibrated MGE.

    `mge_mass_scale` is the analogue of a light MGE's `ml` for a component
    whose shape template is already in mass units: a normalization on top
    of an otherwise-fixed mass map, typically left `fixed` (see `rescale`'s
    docstring for why it can still move regardless). `to_galax` otherwise
    follows the same path as `TriaxialLightMGEPotential` -- see its
    docstring.
    """

    _type: ClassVar[str] = "TriaxialMassMGEPotential"
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
        return _triaxial_mge_potential(mass_mge, self.parameters, unit_system)

    def rescale(self, mass_scale: float) -> Self:
        rescaled = dict(self.parameters)
        rescaled["mge_mass_scale"] = rescaled["mge_mass_scale"] * mass_scale
        return eqx.tree_at(lambda c: c.parameters, self, rescaled)


def _triaxial_mge_potential(
    mass_mge: MassMGE,
    parameters: Mapping[str, Quantity],
    unit_system: AbstractUnitSystem,
) -> galax.potential.AbstractPotential:
    """Sum one `galax.potential.TriaxialGaussianPotential` per Gaussian component.

    Deprojects `mass_mge` under the shared, global viewing angles
    `theta`/`phi`/`psi` (`tnt.mge.AbstractMGE.deproject_triaxial`), then
    converts each resulting intrinsic Gaussian directly into a
    `galax.potential.TriaxialGaussianPotential` -- its density,
    `rho = rho_0 * exp(-xi^2/2)`, `xi^2 = x^2/r_s^2 + y^2/(q1 r_s)^2 +
    z^2/(q2 r_s)^2`, matches `Deprojected3DMGE`'s own exactly (`r_s <->
    sigma`, `q1 <-> p`, `q2 <-> q`), so equating the two central densities
    gives `m_tot = I * p * q * (2*pi)**1.5 * sigma**3` directly -- no new
    potential formula, `galax` already owns that math.
    """
    deprojected = mass_mge.deproject_triaxial(
        parameters["theta"], parameters["phi"], parameters["psi"]
    )
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
