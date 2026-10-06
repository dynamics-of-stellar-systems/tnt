"""Tests for `tnt.orbit_library`."""

import jax

jax.config.update("jax_enable_x64", True)

import galax.potential as gp
import jax.numpy as jnp
import pytest
from unxt import Quantity, unitsystem

from tnt.orbit_library import StationaryGridOrbitSampler

UNITS = unitsystem("kpc", "Myr", "Msun", "rad")


def _nfw_potential() -> gp.AbstractPotential:
    return gp.NFWPotential(
        m=Quantity(1e11, "Msun"), r_s=Quantity(10.0, "kpc"), units=UNITS
    )


def _flattened_potential() -> gp.AbstractPotential:
    return gp.LMJ09LogarithmicPotential(
        v_c=Quantity(200.0, "km/s"),
        r_s=Quantity(5.0, "kpc"),
        q1=1.0,
        q2=0.8,
        q3=0.9,
        phi=Quantity(0.0, "rad"),
        units=UNITS,
    )


def _sampler(nE: int = 3, nI1: int = 4, nI2: int = 4) -> StationaryGridOrbitSampler:
    return StationaryGridOrbitSampler(
        rmin=Quantity(0.5, "kpc"),
        rmax=Quantity(30.0, "kpc"),
        nE=nE,
        nI1=nI1,
        nI2=nI2,
    )


def test_n_bundles_is_the_full_energy_angle_grid():
    sampler = _sampler(nE=4, nI1=3, nI2=5)
    assert sampler.n_bundles() == 4 * 3 * 5


def test_box_orbits_start_from_rest():
    sampler = _sampler()
    ics = sampler.generate_ics(_nfw_potential())
    assert ics.shape == (sampler.n_bundles(), 6)
    assert bool(jnp.all(ics[:, 3:] == 0.0))


def test_box_orbits_land_on_the_equipotential_sphere():
    """For a spherical potential, every bundle in a shell lands at the same radius."""
    sampler = _sampler(nE=4, nI1=3, nI2=3)
    ics = sampler.generate_ics(_nfw_potential())
    r = jnp.linalg.norm(ics[:, :3], axis=-1)
    shells = r.reshape(sampler.nE, sampler.nI1 * sampler.nI2)
    for shell in shells:
        assert float(jnp.max(shell) - jnp.min(shell)) < 1e-8


def test_box_orbits_land_on_the_equipotential_surface_when_flattened():
    """For a flattened potential, the radius varies with direction, but every
    bundle in a shell sits on the same potential value.
    """
    sampler = _sampler(nE=3, nI1=4, nI2=4)
    potential = _flattened_potential()
    ics = sampler.generate_ics(potential)
    position = ics[:, :3]
    r = jnp.linalg.norm(position, axis=-1)
    r_shells = r.reshape(sampler.nE, sampler.nI1 * sampler.nI2)
    for shell in r_shells:
        assert float(jnp.max(shell) - jnp.min(shell)) > 1e-3

    t0 = Quantity(0.0, "Myr")
    value = potential.potential(Quantity(position, "kpc"), t0).ustrip("kpc2/Myr2")
    phi_shells = value.reshape(sampler.nE, sampler.nI1 * sampler.nI2)
    for shell in phi_shells:
        assert float(jnp.max(shell) - jnp.min(shell)) < 1e-9


def test_angular_grid_is_open_and_bin_centred():
    """theta/phi never land exactly on 0 or pi/2 -- both degenerate."""
    sampler = _sampler(nE=1, nI1=4, nI2=4)
    ics = sampler.generate_ics(_nfw_potential())
    position = ics[:, :3]
    r = jnp.linalg.norm(position, axis=-1)
    theta = jnp.arccos(position[:, 2] / r)
    phi = jnp.arctan2(position[:, 1], position[:, 0])
    half_pi = jnp.pi / 2
    assert bool(jnp.all((theta > 0) & (theta < half_pi)))
    assert bool(jnp.all((phi > 0) & (phi < half_pi)))


def test_rmin_rmax_bound_the_energy_grid_radii():
    sampler = _sampler(nE=5, nI1=2, nI2=2)
    ics = sampler.generate_ics(_nfw_potential())
    r = jnp.linalg.norm(ics[:, :3], axis=-1)
    assert float(r.min()) == pytest.approx(0.5, rel=1e-6)
    assert float(r.max()) == pytest.approx(30.0, rel=1e-6)
