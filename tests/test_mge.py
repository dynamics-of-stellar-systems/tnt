import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import unxt as u

from tnt.mge import MGE

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _internal_unit_system() -> u.AbstractUnitSystem:
    return u.unitsystem("kpc", "Myr", "Msun", "rad", "Lsun")


def test_read_converts_columns_to_unit_system():
    unit_system = _internal_unit_system()

    mge = MGE.read(FIXTURES_DIR / "mge_lum.ecsv", unit_system)

    assert mge.I.unit == u.unit("Lsun / rad2")
    assert mge.sigma.unit == u.unit("rad")
    assert mge.q.unit == u.unit("")
    assert mge.PA_twist.unit == u.unit("rad")
    assert jnp.allclose(
        mge.q.ustrip(""),
        jnp.array([0.89541, 0.79093, 0.9999, 0.55097, 0.9999, 0.55097]),
    )


def test_mge_is_frozen():
    mge = MGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())

    with pytest.raises(dataclasses.FrozenInstanceError):
        mge.q = mge.q


def test_mge_is_a_jax_pytree():
    mge = MGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())

    doubled = jax.tree_util.tree_map(lambda leaf: leaf * 2, mge)

    assert jnp.allclose(doubled.q.ustrip(""), mge.q.ustrip("") * 2)
