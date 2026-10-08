"""Tests for `tnt.orbit_library`."""

import jax

jax.config.update("jax_enable_x64", True)

import galax.potential as gp
import jax.numpy as jnp
import pytest
from unxt import Quantity, unitsystem

from tnt.orbit_library import (
    StationaryGridOrbitSampler,
    XZGridFromBoundaryOrbitSampler,
    XZGridFromOriginOrbitSampler,
)

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


@pytest.mark.parametrize(
    "sampler_type", [StationaryGridOrbitSampler, XZGridFromOriginOrbitSampler]
)
@pytest.mark.parametrize("bad_rmin", [0.0, -1.0])
def test_sampler_construction_rejects_non_positive_rmin(sampler_type, bad_rmin: float):
    with pytest.raises(ValueError, match="rmin"):
        sampler_type(
            rmin=Quantity(bad_rmin, "kpc"),
            rmax=Quantity(30.0, "kpc"),
            nE=3,
            nI1=4,
            nI2=4,
        )


@pytest.mark.parametrize("bad_rmin", [0.0, -1.0])
def test_boundary_sampler_construction_rejects_non_positive_rmin(bad_rmin: float):
    with pytest.raises(ValueError, match="rmin"):
        XZGridFromBoundaryOrbitSampler(
            rmin=Quantity(bad_rmin, "kpc"),
            rmax=Quantity(30.0, "kpc"),
            nE=3,
            nI1=4,
            nI2=4,
        )


def test_sampler_construction_rejects_rmin_at_or_above_rmax():
    with pytest.raises(ValueError, match="rmin"):
        StationaryGridOrbitSampler(
            rmin=Quantity(30.0, "kpc"),
            rmax=Quantity(30.0, "kpc"),
            nE=3,
            nI1=4,
            nI2=4,
        )


def test_sampler_construction_rejects_mismatched_unit_rmin_above_rmax():
    """An equivalent-unit rmin/rmax pair must compare correctly, not just by
    their raw declared numbers.
    """
    with pytest.raises(ValueError, match="rmin"):
        StationaryGridOrbitSampler(
            rmin=Quantity(30_001.0, "pc"),
            rmax=Quantity(30.0, "kpc"),
            nE=3,
            nI1=4,
            nI2=4,
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


def _xz_sampler(
    nE: int = 3, nI1: int = 4, nI2: int = 5
) -> XZGridFromOriginOrbitSampler:
    return XZGridFromOriginOrbitSampler(
        rmin=Quantity(0.5, "kpc"),
        rmax=Quantity(30.0, "kpc"),
        nE=nE,
        nI1=nI1,
        nI2=nI2,
    )


def test_xz_n_bundles_is_the_full_energy_angle_radius_grid():
    sampler = _xz_sampler(nE=4, nI1=3, nI2=5)
    assert sampler.n_bundles() == 4 * 3 * 5


def test_xz_orbits_are_confined_to_the_xz_plane_with_only_positive_vy():
    sampler = _xz_sampler()
    ics = sampler.generate_ics(_flattened_potential())
    assert ics.shape == (sampler.n_bundles(), 6)
    position, velocity = ics[:, :3], ics[:, 3:]
    assert bool(jnp.all(position[:, 1] == 0.0))
    assert bool(jnp.all(velocity[:, 0] == 0.0))
    assert bool(jnp.all(velocity[:, 2] == 0.0))
    assert bool(jnp.all(velocity[:, 1] >= 0.0))


def test_xz_orbits_conserve_energy_per_shell():
    sampler = _xz_sampler(nE=3, nI1=4, nI2=5)
    potential = _flattened_potential()
    ics = sampler.generate_ics(potential)
    position, velocity = ics[:, :3], ics[:, 3:]
    t0 = Quantity(0.0, "Myr")
    value = potential.potential(Quantity(position, "kpc"), t0).ustrip("kpc2/Myr2")
    total_energy = value + 0.5 * velocity[:, 1] ** 2
    shells = total_energy.reshape(sampler.nE, sampler.nI1 * sampler.nI2)
    for shell in shells:
        assert float(jnp.max(shell) - jnp.min(shell)) < 1e-9


def test_xz_radii_are_strictly_increasing_and_never_reach_the_equipotential():
    sampler = _xz_sampler(nE=2, nI1=2, nI2=6)
    potential = _flattened_potential()
    ics = sampler.generate_ics(potential)
    r = jnp.linalg.norm(ics[:, :3], axis=-1)
    v_y = ics[:, 4]
    r_grid = r.reshape(sampler.nE, sampler.nI1, sampler.nI2)
    for e_i in range(sampler.nE):
        for th_i in range(sampler.nI1):
            radii = r_grid[e_i, th_i]
            assert bool(jnp.all(jnp.diff(radii) > 0))
            assert float(radii[0]) > 0.0
    # v_y = 0 exactly at the equipotential -- the radial grid should never
    # land there (the "nearly closed" fractional spacing's whole point).
    assert bool(jnp.all(v_y > 0.0))


def test_xz_theta_grid_is_open_and_bin_centred():
    sampler = _xz_sampler(nE=1, nI1=5, nI2=1)
    ics = sampler.generate_ics(_flattened_potential())
    position = ics[:, :3]
    theta = jnp.arctan2(position[:, 0], position[:, 2])
    half_pi = jnp.pi / 2
    assert bool(jnp.all((theta > 0) & (theta < half_pi)))


def _triaxial_hernquist(q1: float = 0.8, q2: float = 0.6) -> gp.AbstractPotential:
    """A genuinely triaxial test potential, away from axisymmetric degeneracy."""
    return gp.TriaxialHernquistPotential(
        m_tot=Quantity(1e11, "Msun"),
        r_s=Quantity(2.0, "kpc"),
        q1=q1,
        q2=q2,
        units="galactic",
    )


def _boundary_sampler(
    nE: int = 2, nI1: int = 3, nI2: int = 3
) -> XZGridFromBoundaryOrbitSampler:
    return XZGridFromBoundaryOrbitSampler(
        rmin=Quantity(0.1, "kpc"),
        rmax=Quantity(10.0, "kpc"),
        nE=nE,
        nI1=nI1,
        nI2=nI2,
    )


def test_xz_boundary_n_bundles_is_the_full_energy_angle_radius_grid():
    sampler = _boundary_sampler(nE=2, nI1=3, nI2=4)
    assert sampler.n_bundles() == 2 * 3 * 4


def test_xz_boundary_orbits_are_confined_to_the_xz_plane_with_only_positive_vy():
    sampler = _boundary_sampler()
    ics = sampler.generate_ics(_triaxial_hernquist())
    assert ics.shape == (sampler.n_bundles(), 6)
    position, velocity = ics[:, :3], ics[:, 3:]
    assert bool(jnp.all(position[:, 1] == 0.0))
    assert bool(jnp.all(velocity[:, 0] == 0.0))
    assert bool(jnp.all(velocity[:, 2] == 0.0))
    assert bool(jnp.all(velocity[:, 1] >= 0.0))


def test_xz_boundary_orbits_conserve_energy_per_shell():
    sampler = _boundary_sampler()
    potential = _triaxial_hernquist()
    ics = sampler.generate_ics(potential)
    position, velocity = ics[:, :3], ics[:, 3:]
    t0 = Quantity(0.0, "Myr")
    value = potential.potential(Quantity(position, "kpc"), t0).ustrip("kpc2/Myr2")
    total_energy = value + 0.5 * velocity[:, 1] ** 2
    shells = total_energy.reshape(sampler.nE, sampler.nI1 * sampler.nI2)
    for shell in shells:
        assert float(jnp.max(shell) - jnp.min(shell)) < 1e-6


def test_xz_boundary_delegates_to_xz_grid_from_origin_once_a_shell_is_irregular():
    # A strongly triaxial, near-prolate potential is where DYNAMITE's own
    # clean four-region picture tends to break down -- a real shell here is
    # expected to come back irregular, exercising the delegation path.
    potential = _triaxial_hernquist(q1=0.65, q2=0.60)
    sampler = _boundary_sampler(nE=3, nI1=3, nI2=3)
    origin_sampler = XZGridFromOriginOrbitSampler(
        rmin=sampler.rmin,
        rmax=sampler.rmax,
        nE=sampler.nE,
        nI1=sampler.nI1,
        nI2=sampler.nI2,
    )
    ics = sampler.generate_ics(potential)
    origin_ics = origin_sampler.generate_ics(potential)

    n_per_shell = sampler.nI1 * sampler.nI2
    matches_origin = [
        bool(
            jnp.allclose(
                ics[i * n_per_shell : (i + 1) * n_per_shell],
                origin_ics[i * n_per_shell : (i + 1) * n_per_shell],
            )
        )
        for i in range(sampler.nE)
    ]
    # At least one shell delegates; once one does, every shell inward of it
    # (lower index -- shells are stored innermost-first) delegates too.
    assert any(matches_origin)
    last_delegated = len(matches_origin) - 1 - matches_origin[::-1].index(True)
    assert all(matches_origin[: last_delegated + 1])
    assert not any(matches_origin[last_delegated + 1 :])
