"""Reference clocks are shared metadata, not individual launch-radius periods."""

import jax

jax.config.update("jax_enable_x64", True)

import galax.potential as gp
import jax.numpy as jnp
import numpy as np
import pytest
from unxt import Quantity

from tnt.orbit_library import (
    CubicOrbitDithering,
    StationaryGridOrbitSampler,
    XZGridFromBoundaryOrbitSampler,
    build_orbit_dithering,
    build_orbit_sampler,
)


def test_stationary_starts_preserve_supplied_shell_clocks_and_units():
    sampler = StationaryGridOrbitSampler(
        rmin=Quantity(1.0, "kpc"), rmax=Quantity(5.0, "kpc"), nE=2, nI2=2, nI3=3
    )
    potential = gp.HarmonicOscillatorPotential(
        omega=Quantity(1.0, "1/Myr"), units="galactic"
    )
    starts = sampler.generate_initial_conditions(
        potential, reference_periods=Quantity([2e6, 3e6], "yr")
    )
    np.testing.assert_allclose(starts.reference_periods.ustrip("Myr")[:6, 0], 2.0)
    np.testing.assert_allclose(starts.reference_periods.ustrip("Myr")[6:, 0], 3.0)
    assert starts.positions.shape == (12, 1, 3)
    assert bool(jnp.all(starts.velocities.ustrip("km/s") == 0))
    with pytest.raises(TypeError, match="reference_periods"):
        sampler.generate_initial_conditions(potential)


def test_boundary_clock_uses_the_inner_tube_boundary_not_the_shell_radius():
    potential = gp.TriaxialHernquistPotential(
        m_tot=Quantity(1e11, "Msun"), r_s=Quantity(2.0, "kpc"),
        q1=0.8, q2=0.6, units="galactic",
    )
    sampler = XZGridFromBoundaryOrbitSampler(
        rmin=Quantity(float(np.sqrt(15)), "kpc"), rmax=Quantity(30.0, "kpc"),
        nE=1, nI2=4, nI3=3,
    )
    starts = sampler.generate_initial_conditions(potential)
    radii = jnp.linalg.norm(starts.positions.ustrip("kpc")[:, 0], axis=-1)
    last_theta_radii = radii.reshape(1, 4, 3)[0, -1]
    # Recover the physical inner boundary from the two radial endpoint
    # samples. This checks the exported clock against the actual launch grid.
    f_first, f_last = 0.1 / 2.2, 2.1 / 2.2
    boundary = (f_last * last_theta_radii[0] - f_first * last_theta_radii[-1]) / (
        f_last - f_first
    )
    assert float(boundary) > float(sampler.rmin.ustrip("kpc")) * 1e-3
    acceleration = potential.acceleration(
        Quantity(jnp.array([boundary, 0.0, 0.0]), "kpc"), Quantity(0.0, "Myr")
    ).ustrip("kpc/Myr2")[0]
    expected_period = 2 * jnp.pi * jnp.sqrt(boundary / jnp.abs(acceleration))
    np.testing.assert_allclose(
        starts.reference_periods.ustrip("Myr"), expected_period, rtol=1e-9
    )


def test_sampler_builders_consume_the_existing_configuration_schema():
    settings = {
        "orbit_sampler": {
            "type": "StationaryGrid", "nE": 2, "nI2": 2, "nI3": 3,
            "rmin": {"value": 1.0, "unit": "kpc"},
            "rmax": {"value": 5.0, "unit": "kpc"},
        },
        "dithering": {"type": "Cubic", "n_dither": 1},
    }
    assert build_orbit_sampler(settings).n_bundles() == 12
    assert build_orbit_dithering(settings).n_orbits_per_bundle() == 1
    settings["dithering"]["n_dither"] = 2
    with pytest.raises(NotImplementedError, match="n_dither=1"):
        build_orbit_dithering(settings)
    with pytest.raises(ValueError, match="positive"):
        CubicOrbitDithering(n_dither=0)
    with pytest.raises(TypeError, match="integer"):
        CubicOrbitDithering(n_dither=True)
