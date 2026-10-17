"""Bounded-memory integration, aggregation, rescaling, and separate output."""

import json

import jax

jax.config.update("jax_enable_x64", True)

import galax.potential as gp
import jax.numpy as jnp
import numpy as np
import pytest
from unxt import Quantity

from tnt.orbit_library import (
    OrbitInitialConditions,
    OrbitIntegrationError,
    OrbitIntegrationSettings,
    build_orbit_library,
)

SETTINGS = OrbitIntegrationSettings(
    orbital_periods=2, n_stored_timesteps=100, accuracy=1e-9
)
KEY = jax.random.key(3)


def _potential():
    return gp.CompositePotential(
        {
            "stars": gp.HarmonicOscillatorPotential(
                omega=Quantity(1.0, "1/Myr"), units="galactic"
            )
        },
        units="galactic",
    )


def _starts(ids=(10, 30)):
    n = len(ids)
    return OrbitInitialConditions(
        positions=Quantity(jnp.tile(jnp.array([1.0, 0.0, 0.0]), (n, 1, 1)), "kpc"),
        velocities=Quantity(jnp.zeros((n, 1, 3)), "kpc/Myr"),
        reference_periods=Quantity(jnp.full((n, 1), 2 * jnp.pi), "Myr"),
        bundle_ids=jnp.array(ids),
    )


def _contributions(orbit):
    q = orbit.positions.ustrip("kpc")
    v = orbit.velocities.ustrip("kpc/Myr")
    return jnp.array(
        [
            jnp.mean(q[:, 0] > 0),
            jnp.mean(v[:, 0] ** 2),
            jnp.mean(jnp.abs(v[:, 0]) > 0.75),
        ]
    )


def test_library_retains_bundle_contributions_and_saves_selected_raw_data(tmp_path):
    libraries = build_orbit_library(
        _potential(),
        _starts(),
        {"observables": _contributions},
        KEY,
        SETTINGS,
        mass_scales=(1.0, 4.0),
        trajectory_directory=tmp_path,
        save_bundle_ids=(30,),
    )
    base, scaled = libraries[1.0], libraries[4.0]
    assert base.contributions["observables"].shape == (2, 3)
    assert bool(jnp.all(base.diagnostics.successful))
    assert float(base.contributions["observables"][0, 0]) < 0.7
    np.testing.assert_allclose(
        scaled.contributions["observables"][:, 1],
        4 * base.contributions["observables"][:, 1],
        rtol=1e-12,
    )
    # Re-binning scaled samples changes membership; scaling the original
    # counts would give the wrong result for this fixed velocity threshold.
    assert bool(
        jnp.all(
            scaled.contributions["observables"][:, 2]
            > base.contributions["observables"][:, 2]
        )
    )
    np.testing.assert_allclose(
        scaled.reference_periods.ustrip("Myr"),
        base.reference_periods.ustrip("Myr") / 2,
    )
    assert scaled.potential_mass_scale == 4.0
    assert len(base.trajectories) == 1
    reference = base.trajectories[0]
    assert reference.bundle_id == 30
    assert reference.member_index == 0
    assert scaled.trajectories == base.trajectories
    with np.load(reference.path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        assert data["positions"].shape == (100, 3)
        assert data["times"].shape == (100,)
        assert metadata["bundle_id"] == 30
        assert metadata["solver"] == "diffrax.Dopri8"
        assert metadata["potential_mass_scale"] == 1.0
        assert metadata["potential"]["units"]["length"] == "kpc"
        assert metadata["potential"]["components"]["stars"]["type"] == (
            "HarmonicOscillatorPotential"
        )
        assert metadata["diagnostics"]["successful"] is True
    assert not (tmp_path / "bundle-10-member-0.npz").exists()
    with pytest.raises(FileExistsError):
        build_orbit_library(
            _potential(),
            _starts(),
            {"observables": _contributions},
            KEY,
            SETTINGS,
            trajectory_directory=tmp_path,
            save_bundle_ids=(30,),
        )


def test_saved_gaussian_metadata_and_identity_include_force_quadrature(tmp_path):
    records = []
    for order in (50, 800):
        potential = gp.CompositePotential(
            {
                "stars": gp.TriaxialGaussianPotential(
                    m_tot=Quantity(1e10, "Msun"),
                    r_s=Quantity(1.0, "kpc"),
                    q1=0.8,
                    q2=0.6,
                    integration_order=order,
                    units="galactic",
                )
            },
            units="galactic",
        )
        library = build_orbit_library(
            potential,
            _starts((10,)),
            {"observables": _contributions},
            KEY,
            OrbitIntegrationSettings(
                orbital_periods=0.1, n_stored_timesteps=12, accuracy=1e-9
            ),
            trajectory_directory=tmp_path / str(order),
            save_bundle_ids=(10,),
        )[1.0]
        with np.load(library.trajectories[0].path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
        assert (
            metadata["potential"]["components"]["stars"]["integration_order"] == order
        )
        records.append(metadata)
    assert records[0]["potential_id"] != records[1]["potential_id"]
    assert (
        records[0]["potential"]["components"]["stars"]["parameters"]
        == records[1]["potential"]["components"]["stars"]["parameters"]
    )


def test_subset_keeps_the_same_sampling_key_for_each_bundle():
    full = build_orbit_library(
        _potential(), _starts(), {"observables": _contributions}, KEY, SETTINGS
    )[1.0]
    subset = build_orbit_library(
        _potential(), _starts((30,)), {"observables": _contributions}, KEY, SETTINGS
    )[1.0]
    np.testing.assert_array_equal(
        full.contributions["observables"][1], subset.contributions["observables"][0]
    )


def test_failed_orbit_is_not_reduced_or_saved(tmp_path):
    def forbidden_reducer(_orbit):
        raise AssertionError("An unsuccessful trajectory must not be reduced.")

    with pytest.raises(OrbitIntegrationError, match="bundle 10"):
        build_orbit_library(
            _potential(),
            _starts((10,)),
            {"density": forbidden_reducer},
            KEY,
            OrbitIntegrationSettings(n_stored_timesteps=10, max_steps=1),
            trajectory_directory=tmp_path,
            save_bundle_ids=(10,),
        )
    assert not list(tmp_path.iterdir())


def test_invalid_request_fails_before_integration():
    with pytest.raises(ValueError, match="unique"):
        build_orbit_library(_potential(), _starts((10, 10)), {}, KEY, SETTINGS)
    with pytest.raises(ValueError, match="at least one"):
        build_orbit_library(_potential(), _starts(), {}, KEY, SETTINGS)
    with pytest.raises(ValueError, match="not in"):
        build_orbit_library(
            _potential(),
            _starts(),
            {"obs": _contributions},
            KEY,
            SETTINGS,
            save_bundle_ids=(99,),
        )


def test_nonfinite_contributions_raise_explicitly():
    with pytest.raises(OrbitIntegrationError, match="Nonfinite contribution"):
        build_orbit_library(
            _potential(), _starts((10,)), {"bad": lambda _orbit: jnp.nan}, KEY, SETTINGS
        )
