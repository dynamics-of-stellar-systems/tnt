"""Numerical regression checks independent of DYNAMITE's implementation."""

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import galax.potential as gp
import jax.numpy as jnp
import numpy as np
import pytest
from galax.potential.params import LinearParameter
from unxt import Quantity, unitsystem

from tnt.orbit_library import OrbitIntegrationSettings, integrate_orbit

UNITS = unitsystem("kpc", "Myr", "Msun", "rad")
KEY = jax.random.key(17)


def _harmonic(units=UNITS):
    return gp.HarmonicOscillatorPotential(omega=Quantity(1.0, "1/Myr"), units=units)


def test_harmonic_orbit_matches_analytic_solution_over_200_reference_periods():
    orbit = integrate_orbit(
        _harmonic(), Quantity([1.0, 0.0, 0.0], "kpc"),
        Quantity([0.0, 1.0, 0.0], "kpc/Myr"), Quantity(2 * jnp.pi, "Myr"), KEY,
        OrbitIntegrationSettings(n_stored_timesteps=600, accuracy=1e-10),
    )
    assert bool(orbit.diagnostics.successful)
    times = orbit.times.ustrip("Myr")
    expected_q = jnp.stack([jnp.cos(times), jnp.sin(times), jnp.zeros_like(times)], -1)
    expected_v = jnp.stack([-jnp.sin(times), jnp.cos(times), jnp.zeros_like(times)], -1)
    np.testing.assert_allclose(orbit.positions.ustrip("kpc"), expected_q, atol=2e-6)
    np.testing.assert_allclose(
        orbit.velocities.ustrip("kpc/Myr"), expected_v, atol=2e-6
    )
    assert float(orbit.diagnostics.max_energy_error) < 2e-7
    angular_momentum = jnp.cross(
        orbit.positions.ustrip("kpc"), orbit.velocities.ustrip("kpc/Myr")
    )
    np.testing.assert_allclose(angular_momentum[:, 2], 1.0, atol=4e-7)


def test_sampling_uses_the_supplied_clock_and_reproducible_dynamite_offset():
    settings = OrbitIntegrationSettings(orbital_periods=3, n_stored_timesteps=12)
    orbit = integrate_orbit(
        _harmonic(), Quantity([1, 0, 0], "kpc"), Quantity([0, 0, 0], "km/s"),
        Quantity(2.0, "Myr"), KEY, settings,
    )
    times = orbit.times.ustrip("Myr")
    spacing = 6.0 / 16
    np.testing.assert_allclose(jnp.diff(times), spacing, rtol=1e-13)
    assert spacing <= float(times[0]) < 2 * spacing
    assert float(times[-1]) < 6.0
    expected_first = spacing * (1 + jax.random.uniform(KEY, dtype=jnp.float64))
    np.testing.assert_allclose(times[0], expected_first, rtol=1e-13)


def test_integration_and_tolerances_are_invariant_to_internal_and_input_units():
    settings = OrbitIntegrationSettings(
        orbital_periods=2, n_stored_timesteps=35, accuracy=1e-11
    )
    first = integrate_orbit(
        _harmonic(), Quantity([1.0, 0.0, 0.0], "kpc"),
        Quantity([0.0, 0.0, 0.0], "kpc/Myr"), Quantity(2 * jnp.pi, "Myr"),
        KEY, settings,
    )
    second = integrate_orbit(
        _harmonic(unitsystem("pc", "yr", "Msun", "rad")),
        Quantity([1000.0, 0.0, 0.0], "pc"), Quantity([0.0, 0.0, 0.0], "km/s"),
        Quantity(2 * jnp.pi * 1e6, "yr"), KEY, settings,
    )
    np.testing.assert_allclose(
        first.positions.ustrip("kpc"), second.positions.ustrip("kpc"),
        rtol=0, atol=2e-8,
    )
    np.testing.assert_allclose(
        first.times.ustrip("Myr"), second.times.ustrip("Myr"), rtol=1e-13,
    )
    # Roundoff can change accepted adaptive steps; both physical solutions
    # must satisfy the same accuracy bound, including near a zero crossing.
    for trajectory in (first, second):
        np.testing.assert_allclose(
            trajectory.positions.ustrip("kpc")[:, 0],
            jnp.cos(trajectory.times.ustrip("Myr")), rtol=0, atol=2e-8,
        )
        assert float(trajectory.diagnostics.max_energy_error) < 2e-8


def test_zero_total_energy_is_a_valid_initial_condition():
    potential = gp.PlummerPotential(
        m_tot=Quantity(1e10, "Msun"), r_s=Quantity(1.0, "kpc"), units=UNITS
    )
    position = Quantity([1.0, 0.0, 0.0], "kpc")
    phi = potential.potential(position, Quantity(0.0, "Myr")).ustrip("kpc2/Myr2")
    speed = jnp.sqrt(-2 * phi)
    orbit = integrate_orbit(
        potential, position, Quantity(jnp.array([0.0, speed, 0.0]), "kpc/Myr"),
        Quantity(5.0, "Myr"), KEY,
        OrbitIntegrationSettings(
            orbital_periods=2, n_stored_timesteps=50, accuracy=1e-9
        ),
    )
    assert bool(orbit.diagnostics.successful)
    assert float(orbit.diagnostics.max_energy_error) < 1e-7


def test_step_exhaustion_is_explicitly_unsuccessful():
    orbit = integrate_orbit(
        _harmonic(), Quantity([1.0, 0.0, 0.0], "kpc"),
        Quantity([0.0, 0.0, 0.0], "kpc/Myr"), Quantity(6.0, "Myr"), KEY,
        OrbitIntegrationSettings(n_stored_timesteps=10, max_steps=1),
    )
    assert not bool(orbit.diagnostics.successful)
    assert int(orbit.diagnostics.num_steps) == 1


def test_kernel_differentiates_through_initial_conditions():
    settings = OrbitIntegrationSettings(
        orbital_periods=0.3, n_stored_timesteps=12, accuracy=1e-10
    )

    def final_x(amplitude):
        orbit = integrate_orbit(
            _harmonic(), Quantity(jnp.array([amplitude, 0.0, 0.0]), "kpc"),
            Quantity([0.0, 0.0, 0.0], "kpc/Myr"), Quantity(2 * jnp.pi, "Myr"),
            KEY, settings,
        )
        return orbit.positions.ustrip("kpc")[-1, 0]

    value, derivative = eqx.filter_value_and_grad(final_x)(jnp.asarray(1.4))
    np.testing.assert_allclose(derivative, value / 1.4, rtol=1e-7, atol=1e-9)


def test_kernel_differentiates_through_potential_parameters():
    settings = OrbitIntegrationSettings(
        orbital_periods=0.3, n_stored_timesteps=12, accuracy=1e-10
    )

    def final_x(omega):
        potential = gp.HarmonicOscillatorPotential(
            omega=Quantity(omega, "1/Myr"), units=UNITS
        )
        orbit = integrate_orbit(
            potential, Quantity([1.0, 0.0, 0.0], "kpc"),
            Quantity([0.0, 0.0, 0.0], "kpc/Myr"), Quantity(2 * jnp.pi, "Myr"),
            KEY, settings,
        )
        return orbit.positions.ustrip("kpc")[-1, 0]

    omega = jnp.asarray(1.3)
    value, derivative = eqx.filter_value_and_grad(final_x)(omega)
    last_time = (0.3 * 2 * jnp.pi / 16) * (
        12 + jax.random.uniform(KEY, dtype=jnp.float64)
    )
    np.testing.assert_allclose(value, jnp.cos(omega * last_time), atol=1e-8)
    np.testing.assert_allclose(
        derivative, -last_time * jnp.sin(omega * last_time), rtol=1e-7, atol=1e-8
    )


def test_eccentric_kepler_orbit_matches_the_analytic_ellipse():
    potential = gp.KeplerPotential(m_tot=Quantity(1e10, "Msun"), units=UNITS)
    mu = float(potential.constants["G"].ustrip("kpc3/(Myr2 Msun)")) * 1e10
    eccentricity = 0.7
    period = 2 * np.pi / np.sqrt(mu)  # semi-major axis = 1 kpc
    speed = np.sqrt(mu * (1 + eccentricity) / (1 - eccentricity))
    orbit = integrate_orbit(
        potential, Quantity([1 - eccentricity, 0.0, 0.0], "kpc"),
        Quantity([0.0, speed, 0.0], "kpc/Myr"), Quantity(period, "Myr"), KEY,
        OrbitIntegrationSettings(
            orbital_periods=20, n_stored_timesteps=600, accuracy=1e-11
        ),
    )
    mean_anomaly = np.mod(2 * np.pi * orbit.times.ustrip("Myr") / period, 2 * np.pi)
    eccentric_anomaly = np.array(mean_anomaly)
    for _ in range(20):
        eccentric_anomaly -= (
            eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
        ) / (1 - eccentricity * np.cos(eccentric_anomaly))
    expected = np.stack([
        np.cos(eccentric_anomaly) - eccentricity,
        np.sqrt(1 - eccentricity**2) * np.sin(eccentric_anomaly),
        np.zeros_like(eccentric_anomaly),
    ], axis=-1)
    assert bool(orbit.diagnostics.successful)
    np.testing.assert_allclose(
        orbit.positions.ustrip("kpc"), expected, rtol=0, atol=1e-5
    )
    assert float(orbit.diagnostics.max_energy_error) < 1e-7


def test_time_dependent_parameters_are_rejected():
    potential = gp.HarmonicOscillatorPotential(
        omega=LinearParameter(
            slope=Quantity(0.1, "1/Myr2"),
            point_time=Quantity(0.0, "Myr"), point_value=Quantity(1.0, "1/Myr"),
        ), units=UNITS,
    )
    with pytest.raises(NotImplementedError, match="Time-dependent"):
        integrate_orbit(
            potential, Quantity([1.0, 0.0, 0.0], "kpc"),
            Quantity([0.0, 0.0, 0.0], "kpc/Myr"), Quantity(6.0, "Myr"), KEY,
        )


def test_rotating_bar_is_rejected_even_with_constant_parameter_objects():
    potential = gp.MonariEtAl2016BarPotential(
        R0=Quantity(8.0, "kpc"), v0=Quantity(220.0, "km/s"), units=UNITS
    )
    with pytest.raises(NotImplementedError, match="Rotating bar"):
        integrate_orbit(
            potential, Quantity([1.0, 0.0, 0.0], "kpc"),
            Quantity([0.0, 0.0, 0.0], "kpc/Myr"), Quantity(6.0, "Myr"), KEY,
        )


@pytest.mark.parametrize("bad_period", [0.0, -1.0, jnp.nan])
def test_invalid_reference_period_is_rejected(bad_period):
    with pytest.raises(Exception, match="reference period"):
        integrate_orbit(
            _harmonic(), Quantity([1.0, 0.0, 0.0], "kpc"),
            Quantity([0.0, 0.0, 0.0], "kpc/Myr"), Quantity(bad_period, "Myr"),
            KEY, OrbitIntegrationSettings(n_stored_timesteps=5),
        )
