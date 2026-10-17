"""Force-accuracy regressions through TNT's resolved MGE potential builders."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unxt import Quantity, unitsystem

from tnt.mge import LightMGE, MassMGE
from tnt.potential import Potential

UNITS = unitsystem("kpc", "Myr", "Msun", "rad")
KINDS = [
    "TriaxialLightMGEPotential",
    "TriaxialMassMGEPotential",
    "OblateLightMGEPotential",
    "OblateMassMGEPotential",
]


def _component(kind: str, order: int | None = None, q: float = 1.0):
    light = "Light" in kind
    mge = (LightMGE if light else MassMGE)(
        I=Quantity([2.0, 1.0], "Lsun / pc2" if light else "Msun / pc2"),
        sigma=Quantity([1.0, 2.0], "rad"),
        q=Quantity([q, q], ""),
        PA_twist=Quantity([0.0, 0.0], "rad"),
        major_axis_pa=Quantity(0.0, "rad"),
    ).angular_to_physical(Quantity(1.0, "kpc"))
    viewing = (
        {"inclination": Quantity(90.0, "deg")}
        if kind.startswith("Oblate")
        else {
            "theta": Quantity(1.1, "rad"),
            "phi": Quantity(0.6, "rad"),
            "psi": Quantity(-0.4, "rad"),
        }
    )
    values = {
        "ml" if light else "mge_mass_scale": Quantity(
            2.0, "Msun / Lsun" if light else ""
        ),
        **viewing,
    }
    resolved = Potential.resolve(
        {"stars": {"type": kind, "mge": "m"}},
        {"m": mge},
        **({} if order is None else {"gaussian_potential_quad_order": order}),
    )
    potential, valid = Potential.build_with_validity(resolved, {"stars": values}, {})
    assert bool(valid)
    return potential.components["stars"]


@pytest.mark.parametrize("kind", KINDS)
def test_default_gaussian_forces_match_spherical_far_field_and_mass_scaling(kind):
    component = _component(kind)
    xyz = Quantity([10000.0, 0.0, 0.0], "kpc")
    native = component.to_galax(UNITS)
    gravity = float(native.constants["G"].ustrip("kpc3/(Msun Myr2)"))
    mass = float(jnp.sum(component.deprojected.component_masses.ustrip("Msun")))
    # At >=5000 sigma, the spherical Gaussian's enclosed mass equals its
    # total mass to floating-point precision: Newton's shell theorem.
    expected = np.array([-gravity * mass / 10000**2, 0.0, 0.0])
    actual = native.acceleration(xyz, Quantity(0.0, "Myr")).ustrip("kpc/Myr2")
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=0)

    rescaled = component.rescale(4.0)
    assert rescaled.integration_order == component.integration_order
    for gaussian in rescaled.to_galax(UNITS).values():
        assert gaussian.integration_order == component.integration_order
    force = (
        rescaled.to_galax(UNITS)
        .acceleration(xyz, Quantity(0.0, "Myr"))
        .ustrip("kpc/Myr2")
    )
    np.testing.assert_allclose(force, 4 * actual, rtol=1e-12, atol=0)


def test_explicit_quadrature_override_changes_the_actual_force_evaluator():
    accurate = _component("OblateMassMGEPotential")
    low_order = _component("OblateMassMGEPotential", order=50)
    xyz, time = Quantity([10000.0, 0.0, 0.0], "kpc"), Quantity(0.0, "Myr")
    high = accurate.to_galax(UNITS).acceleration(xyz, time).ustrip("kpc/Myr2")
    low = low_order.to_galax(UNITS).acceleration(xyz, time).ustrip("kpc/Myr2")
    assert float(jnp.linalg.norm(low - high) / jnp.linalg.norm(high)) > 0.1
    assert all(g.integration_order == 50 for g in low_order.to_galax(UNITS).values())
    assert "integration_order" not in low_order.parameters


def test_flattened_central_force_matches_independent_adaptive_reference():
    component = _component("OblateMassMGEPotential", q=0.01)
    native = component.to_galax(UNITS)
    # Scalar adaptive force integrals in benchmarks/gaussian_forces.py,
    # evaluated at r/sigma=1e-4 along the diagonal, G=M=sigma=1.
    # The reference is integrated independently of Galax autodiff.
    point = jnp.ones(3) * (1e-4 / jnp.sqrt(3))
    from scipy.integrate import quad

    def reference_component(axis: int) -> float:
        def integrand(s: float) -> float:
            d = np.array([1, 1, 1 - s * s + 0.0001 * s * s])
            return float(
                s
                * s
                * np.exp(-0.5 * s * s * np.sum(np.asarray(point) ** 2 / d))
                / (d[axis] * np.sqrt(np.prod(d)))
            )

        value, _ = quad(integrand, 0, 1, epsabs=1e-13, epsrel=2e-12, limit=300)
        return float(-np.sqrt(2 / np.pi) * point[axis] * value)

    reference = np.array([reference_component(axis) for axis in range(3)])
    gaussian = native["0"]
    gravity = float(gaussian.constants["G"].ustrip("kpc3/(Msun Myr2)"))
    mass = float(gaussian.m_tot(Quantity(0.0, "Myr")).ustrip("Msun"))
    actual = gaussian.acceleration(Quantity(point, "kpc"), Quantity(0.0, "Myr")).ustrip(
        "kpc/Myr2"
    )
    np.testing.assert_allclose(actual / (gravity * mass), reference, rtol=4e-9)


def test_configured_static_quadrature_survives_jit_and_mass_gradient():
    component = _component("TriaxialLightMGEPotential", order=400)
    xyz, time = Quantity([1.0, 0.2, -0.3], "kpc"), Quantity(0.0, "Myr")

    @eqx.filter_jit
    def force(scale: jax.Array) -> jax.Array:
        return (
            component.rescale(scale)
            .to_galax(UNITS)
            .acceleration(xyz, time)
            .ustrip("kpc/Myr2")[0]
        )

    value, gradient = jax.value_and_grad(force)(jnp.asarray(2.0))
    np.testing.assert_allclose(gradient, value / 2, rtol=1e-11)


@pytest.mark.parametrize("bad_order", [0, -1, 2.5, True, "200"])
def test_programmatic_quadrature_rejects_invalid_orders(bad_order):
    with pytest.raises((TypeError, ValueError), match="quad_order.*integer"):
        _component("OblateMassMGEPotential", order=bad_order)
