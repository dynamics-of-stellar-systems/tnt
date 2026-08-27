"""Unit tests for `tnt.potential.fused_triaxial_gaussian_composite`.

Deliberately doesn't import anything from `tnt` -- the module under test has
zero TNT imports by design (see its own module docstring), and these tests
mirror that: every fixture is built directly from `galax.potential`, the
same style `test_potential.py`'s MGE-`to_galax` tests already use.
"""

from __future__ import annotations

import galax.potential as gp
import jax.numpy as jnp
import numpy as np
import pytest
import unxt as u

from tnt.potential.fused_triaxial_gaussian_composite import (
    FusedTriaxialGaussianCompositePotential,
)


def _unit_system() -> u.AbstractUnitSystem:
    return u.unitsystem("galactic")


def _children(
    unit_system: u.AbstractUnitSystem, *, integration_order: int = 50
) -> dict[str, gp.TriaxialGaussianPotential]:
    # Genuinely triaxial (q1 != q2 != 1) so the ellipsoid math is actually
    # exercised, not just the spherical special case.
    raw = [
        (1.0e8, 1.0, 0.9, 0.7),
        (5.0e7, 2.5, 0.6, 0.4),
        (2.0e8, 0.3, 0.8, 0.75),
        (3.0e7, 4.0, 0.95, 0.5),
    ]
    return {
        str(i): gp.TriaxialGaussianPotential(
            m_tot=u.Quantity(m_tot, "Msun"),
            r_s=u.Quantity(r_s, "kpc"),
            q1=q1,
            q2=q2,
            units=unit_system,
            integration_order=integration_order,
        )
        for i, (m_tot, r_s, q1, q2) in enumerate(raw)
    }


@pytest.fixture
def fused_and_composite() -> tuple[
    FusedTriaxialGaussianCompositePotential, gp.CompositePotential
]:
    unit_system = _unit_system()
    children = _children(unit_system)
    fused = FusedTriaxialGaussianCompositePotential(children)
    composite = gp.CompositePotential(children, units=unit_system)
    return fused, composite


# ---------------------------------------------------------------------------
# Fused potential/gradient match the shipped composite.
# ---------------------------------------------------------------------------


def test_fused_potential_matches_composite_potential(
    fused_and_composite: tuple[
        FusedTriaxialGaussianCompositePotential, gp.CompositePotential
    ],
) -> None:
    fused, composite = fused_and_composite
    xyz = u.Quantity(jnp.array([3.0, -1.5, 0.7]), "kpc")
    t = u.Quantity(0.0, "Myr")

    assert fused.potential(xyz, t).ustrip(
        composite.units["specific energy"]
    ) == pytest.approx(
        composite.potential(xyz, t).ustrip(composite.units["specific energy"]),
        rel=1e-5,
    )


def test_fused_gradient_matches_composite_gradient(
    fused_and_composite: tuple[
        FusedTriaxialGaussianCompositePotential, gp.CompositePotential
    ],
) -> None:
    fused, composite = fused_and_composite
    xyz = u.Quantity(jnp.array([3.0, -1.5, 0.7]), "kpc")
    t = u.Quantity(0.0, "Myr")

    assert np.allclose(
        np.asarray(fused.gradient(xyz, t)),
        np.asarray(composite.gradient(xyz, t)),
        rtol=1e-4,
    )


@pytest.mark.parametrize("batch_shape", [(), (7,), (3, 4)])
def test_fused_potential_and_gradient_match_composite_across_batch_shapes(
    fused_and_composite: tuple[
        FusedTriaxialGaussianCompositePotential, gp.CompositePotential
    ],
    batch_shape: tuple[int, ...],
) -> None:
    fused, composite = fused_and_composite
    rng = np.random.default_rng(0)
    xyz = jnp.asarray(rng.uniform(-5.0, 5.0, size=(*batch_shape, 3)))
    t = 0.0

    potential_fused = fused._potential(xyz, t)
    potential_composite = composite._potential(xyz, t)
    assert potential_fused.shape == batch_shape
    assert np.allclose(potential_fused, potential_composite, rtol=1e-5)

    gradient_fused = fused.gradient(xyz, t)
    gradient_composite = composite.gradient(xyz, t)
    assert gradient_fused.shape == (*batch_shape, 3)
    assert np.allclose(gradient_fused, gradient_composite, rtol=1e-4)


# ---------------------------------------------------------------------------
# Unfused methods stay correct via inherited AbstractCompositePotential
# behavior -- same code path, so these should match exactly, not just
# approximately.
# ---------------------------------------------------------------------------


def test_fused_density_matches_composite(
    fused_and_composite: tuple[
        FusedTriaxialGaussianCompositePotential, gp.CompositePotential
    ],
) -> None:
    fused, composite = fused_and_composite
    xyz = u.Quantity(jnp.array([3.0, -1.5, 0.7]), "kpc")
    t = u.Quantity(0.0, "Myr")

    assert fused.density(xyz, t) == composite.density(xyz, t)


def test_fused_laplacian_matches_composite(
    fused_and_composite: tuple[
        FusedTriaxialGaussianCompositePotential, gp.CompositePotential
    ],
) -> None:
    fused, composite = fused_and_composite
    xyz = u.Quantity(jnp.array([3.0, -1.5, 0.7]), "kpc")
    t = u.Quantity(0.0, "Myr")

    assert fused.laplacian(xyz, t) == composite.laplacian(xyz, t)


# ---------------------------------------------------------------------------
# Mapping protocol / combination parity with CompositePotential.
# ---------------------------------------------------------------------------


def test_fused_mapping_protocol_matches_composite(
    fused_and_composite: tuple[
        FusedTriaxialGaussianCompositePotential, gp.CompositePotential
    ],
) -> None:
    fused, composite = fused_and_composite
    assert list(fused.keys()) == list(composite.keys())
    assert list(fused.values()) == list(composite.values())
    assert list(fused.items()) == list(composite.items())
    assert len(fused) == len(composite)
    for key in composite:
        assert key in fused
        assert fused[key] is composite[key] or fused[key] == composite[key]
    assert "not-a-real-key" not in fused


def test_fused_or_another_potential_returns_plain_composite_potential(
    fused_and_composite: tuple[
        FusedTriaxialGaussianCompositePotential, gp.CompositePotential
    ],
) -> None:
    fused, _composite = fused_and_composite
    unit_system = _unit_system()
    other = gp.HernquistPotential(
        m_tot=u.Quantity(1.0e10, "Msun"), r_s=u.Quantity(3.0, "kpc"), units=unit_system
    )

    combined = fused | other

    assert type(combined) is gp.CompositePotential
    xyz = u.Quantity(jnp.array([3.0, -1.5, 0.7]), "kpc")
    t = u.Quantity(0.0, "Myr")
    expected = fused.potential(xyz, t) + other.potential(xyz, t)
    assert combined.potential(xyz, t).ustrip(
        unit_system["specific energy"]
    ) == pytest.approx(
        expected.ustrip(unit_system["specific energy"]),
        rel=1e-5,
    )


# ---------------------------------------------------------------------------
# Construction validation.
# ---------------------------------------------------------------------------


def test_construction_rejects_non_triaxial_gaussian_component() -> None:
    unit_system = _unit_system()
    children = _children(unit_system)
    children["not_triaxial"] = gp.HernquistPotential(
        m_tot=u.Quantity(1.0e10, "Msun"), r_s=u.Quantity(3.0, "kpc"), units=unit_system
    )

    with pytest.raises(TypeError, match="not a TriaxialGaussianPotential"):
        FusedTriaxialGaussianCompositePotential(children)


def test_construction_rejects_mismatched_integration_order() -> None:
    unit_system = _unit_system()
    children = _children(unit_system)
    children["different_order"] = gp.TriaxialGaussianPotential(
        m_tot=u.Quantity(1.0e8, "Msun"),
        r_s=u.Quantity(1.0, "kpc"),
        units=unit_system,
        integration_order=40,
    )

    with pytest.raises(ValueError, match="same integration_order"):
        FusedTriaxialGaussianCompositePotential(children)


def test_construction_rejects_mismatched_unit_systems() -> None:
    children = _children(_unit_system())
    children["different_units"] = gp.TriaxialGaussianPotential(
        m_tot=u.Quantity(1.0e8, "Msun"),
        r_s=u.Quantity(1.0, "kpc"),
        units="solarsystem",
    )

    with pytest.raises(ValueError, match="same unit system"):
        FusedTriaxialGaussianCompositePotential(children)


def test_construction_units_default_to_first_child() -> None:
    unit_system = _unit_system()
    children = _children(unit_system)

    fused = FusedTriaxialGaussianCompositePotential(children)

    assert fused.units == unit_system
