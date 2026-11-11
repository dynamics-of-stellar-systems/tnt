"""Validation of the retained-library and transient-trajectory boundary."""

import jax.numpy as jnp
import pytest
from unxt import Quantity

from tnt.orbit_library import (
    OrbitInitialConditions,
    OrbitIntegrationDiagnostics,
    OrbitLibrary,
    OrbitTrajectory,
)


def _diagnostics(shape=()):
    return OrbitIntegrationDiagnostics(
        successful=jnp.ones(shape, dtype=bool),
        max_energy_error=jnp.zeros(shape),
        final_energy_error=jnp.zeros(shape),
        num_steps=jnp.ones(shape, dtype=int),
        num_accepted_steps=jnp.ones(shape, dtype=int),
        num_rejected_steps=jnp.zeros(shape, dtype=int),
    )


def test_library_rejects_contributions_from_a_different_bundle_count():
    with pytest.raises(ValueError, match="leading bundle axis"):
        OrbitLibrary(
            contributions={"density": jnp.ones((3, 8))},
            bundle_ids=jnp.array([5, 8]),
            reference_periods=Quantity(jnp.ones((2, 1)), "Myr"),
            diagnostics=_diagnostics((2, 1)),
        )


def test_library_rejects_diagnostics_from_a_different_member_count():
    with pytest.raises(ValueError, match="diagnostics"):
        OrbitLibrary(
            contributions={},
            bundle_ids=jnp.array([5, 8]),
            reference_periods=Quantity(jnp.ones((2, 1)), "Myr"),
            diagnostics=_diagnostics((2, 2)),
        )


def test_initial_conditions_reject_more_than_one_member():
    with pytest.raises(ValueError, match="Initial positions"):
        OrbitInitialConditions(
            positions=Quantity(jnp.ones((2, 2, 3)), "pc"),
            velocities=Quantity(jnp.ones((2, 2, 3)), "km/s"),
            reference_periods=Quantity(jnp.ones((2, 2)), "Myr"),
            bundle_ids=jnp.arange(2),
        )


def test_trajectory_requires_matching_times_positions_and_velocities():
    with pytest.raises(ValueError, match="velocities"):
        OrbitTrajectory(
            times=Quantity(jnp.arange(4.0), "Myr"),
            positions=Quantity(jnp.zeros((4, 3)), "kpc"),
            velocities=Quantity(jnp.zeros((3, 3)), "km/s"),
            reference_period=Quantity(1.0, "Myr"),
            diagnostics=_diagnostics(),
        )


def test_aggregated_library_cannot_silently_rescale_velocity_bin_counts():
    library = OrbitLibrary(
        contributions={"velocity_histogram": jnp.array([[0.2, 0.3]])},
        bundle_ids=jnp.array([12]),
        reference_periods=Quantity(jnp.ones((1, 1)), "Myr"),
        diagnostics=_diagnostics((1, 1)),
    )
    with pytest.raises(NotImplementedError, match="mass scale"):
        library.rescaled(2.0)
