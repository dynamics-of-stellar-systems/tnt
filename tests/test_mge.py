import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import unxt as u

from tnt.mge import LightMGE, MassMGE

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _internal_unit_system() -> u.AbstractUnitSystem:
    return u.unitsystem("kpc", "Myr", "Msun", "rad", "Lsun")


def test_read_converts_light_columns_to_unit_system():
    unit_system = _internal_unit_system()

    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", unit_system)

    assert mge.I.unit == u.unit("Lsun / rad2")
    assert mge.sigma.unit == u.unit("rad")
    assert mge.q.unit == u.unit("")
    assert mge.PA_twist.unit == u.unit("rad")
    assert jnp.allclose(
        mge.q.ustrip(""),
        jnp.array([0.89541, 0.79093, 0.9999, 0.55097, 0.9999, 0.55097]),
    )


def test_read_converts_mass_columns_to_unit_system():
    unit_system = _internal_unit_system()

    mge = MassMGE.read(FIXTURES_DIR / "mge_mass.ecsv", unit_system)

    assert mge.I.unit == u.unit("Msun / rad2")
    assert mge.sigma.unit == u.unit("rad")
    assert mge.q.unit == u.unit("")
    assert mge.PA_twist.unit == u.unit("rad")
    assert jnp.allclose(
        mge.q.ustrip(""),
        jnp.array([0.91205, 0.83017, 0.9999, 0.60214, 0.9999, 0.60214]),
    )


def test_mge_is_frozen():
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())

    with pytest.raises(dataclasses.FrozenInstanceError):
        mge.q = mge.q


def test_mge_is_a_jax_pytree():
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())

    doubled = jax.tree_util.tree_map(lambda leaf: leaf * 2, mge)

    assert jnp.allclose(doubled.q.ustrip(""), mge.q.ustrip("") * 2)


def test_angular_to_physical_converts_sigma_and_intensity():
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    distance = u.Quantity(30.5, "Mpc")

    physical = mge.angular_to_physical(distance)

    assert physical.sigma.unit == u.unit("Mpc")
    assert physical.I.unit == u.unit("Lsun / Mpc2")
    assert jnp.allclose(
        physical.sigma.ustrip("Mpc"),
        distance.ustrip("Mpc") * mge.sigma.ustrip("rad"),
    )
    assert jnp.allclose(
        physical.I.ustrip("Lsun / Mpc2"),
        mge.I.ustrip("Lsun / rad2") / distance.ustrip("Mpc") ** 2,
    )


def test_angular_to_physical_leaves_q_and_pa_twist_unchanged():
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    distance = u.Quantity(30.5, "Mpc")

    physical = mge.angular_to_physical(distance)

    assert jnp.allclose(physical.q.ustrip(""), mge.q.ustrip(""))
    assert jnp.allclose(
        physical.PA_twist.ustrip("rad"), mge.PA_twist.ustrip("rad")
    )


def test_angular_physical_round_trip():
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    distance = u.Quantity(30.5, "Mpc")

    round_tripped = mge.angular_to_physical(distance).physical_to_angular(distance)

    assert jnp.allclose(round_tripped.sigma.ustrip("rad"), mge.sigma.ustrip("rad"))
    assert jnp.allclose(round_tripped.I.ustrip("Lsun / rad2"), mge.I.ustrip("Lsun / rad2"))
    assert jnp.allclose(round_tripped.q.ustrip(""), mge.q.ustrip(""))
    assert jnp.allclose(
        round_tripped.PA_twist.ustrip("rad"), mge.PA_twist.ustrip("rad")
    )
