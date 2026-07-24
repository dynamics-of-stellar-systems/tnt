import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import unxt as u

from tnt.mge import LightMGE, MassMGE, read_mge

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


def test_read_mge_infers_light_kind():
    mge = read_mge(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())

    assert isinstance(mge, LightMGE)
    assert mge.I.unit == u.unit("Lsun / rad2")


def test_read_mge_infers_mass_kind():
    mge = read_mge(FIXTURES_DIR / "mge_mass.ecsv", _internal_unit_system())

    assert isinstance(mge, MassMGE)
    assert mge.I.unit == u.unit("Msun / rad2")


@pytest.mark.parametrize("bad_q", [0.0, -0.5, 1.5])
def test_read_rejects_q_out_of_range(tmp_path, bad_q):
    bad_file = tmp_path / "bad_q.ecsv"
    bad_file.write_text(
        "# %ECSV 0.9\n"
        "# ---\n"
        "# datatype:\n"
        "# - {name: I, unit: Lsun / arcsec2, datatype: float64}\n"
        "# - {name: sigma, unit: arcsec, datatype: float64}\n"
        "# - {name: q, unit: '', datatype: float64}\n"
        "# - {name: PA_twist, unit: deg, datatype: float64}\n"
        "# schema: astropy-2.0\n"
        "I sigma q PA_twist\n"
        f"1.0 1.0 {bad_q} 0.0\n"
    )

    with pytest.raises(ValueError, match="q must satisfy 0 < q <= 1"):
        LightMGE.read(bad_file, _internal_unit_system())


def test_read_accepts_q_equal_to_one(tmp_path):
    ok_file = tmp_path / "q_one.ecsv"
    ok_file.write_text(
        "# %ECSV 0.9\n"
        "# ---\n"
        "# datatype:\n"
        "# - {name: I, unit: Lsun / arcsec2, datatype: float64}\n"
        "# - {name: sigma, unit: arcsec, datatype: float64}\n"
        "# - {name: q, unit: '', datatype: float64}\n"
        "# - {name: PA_twist, unit: deg, datatype: float64}\n"
        "# schema: astropy-2.0\n"
        "I sigma q PA_twist\n"
        "1.0 1.0 1.0 0.0\n"
    )

    mge = LightMGE.read(ok_file, _internal_unit_system())

    assert jnp.allclose(mge.q.ustrip(""), 1.0)


def test_read_mge_rejects_unrecognized_units(tmp_path):
    bad_file = tmp_path / "bad.ecsv"
    bad_file.write_text(
        "# %ECSV 0.9\n"
        "# ---\n"
        "# datatype:\n"
        "# - {name: I, unit: s, datatype: float64}\n"
        "# - {name: sigma, unit: arcsec, datatype: float64}\n"
        "# - {name: q, unit: '', datatype: float64}\n"
        "# - {name: PA_twist, unit: deg, datatype: float64}\n"
        "# schema: astropy-2.0\n"
        "I sigma q PA_twist\n"
        "1.0 1.0 1.0 0.0\n"
    )

    with pytest.raises(ValueError, match="Could not infer MGE kind"):
        read_mge(bad_file, _internal_unit_system())


def test_to_mass_with_constant_ratio():
    light = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    m_over_l = u.Quantity(2.5, "Msun / Lsun")

    mass = light.to_mass(m_over_l)

    assert isinstance(mass, MassMGE)
    assert mass.I.unit == u.unit("Msun / rad2")
    assert jnp.allclose(
        mass.I.ustrip("Msun / rad2"), light.I.ustrip("Lsun / rad2") * 2.5
    )
    assert jnp.allclose(mass.sigma.ustrip("rad"), light.sigma.ustrip("rad"))
    assert jnp.allclose(mass.q.ustrip(""), light.q.ustrip(""))
    assert jnp.allclose(mass.PA_twist.ustrip("rad"), light.PA_twist.ustrip("rad"))


def test_to_mass_with_per_component_ratio():
    light = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    ratios = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    m_over_l = u.Quantity(ratios, "Msun / Lsun")

    mass = light.to_mass(m_over_l)

    assert isinstance(mass, MassMGE)
    assert jnp.allclose(
        mass.I.ustrip("Msun / rad2"), light.I.ustrip("Lsun / rad2") * ratios
    )


def test_to_mass_rejects_mismatched_component_count():
    light = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    m_over_l = u.Quantity(jnp.array([1.0, 2.0]), "Msun / Lsun")

    with pytest.raises(ValueError, match="m_over_l has 2 components"):
        light.to_mass(m_over_l)


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
