"""Exercise the real nested Gaussian-plus-central-mass force path."""

import jax

jax.config.update("jax_enable_x64", True)

import galax.potential as gp
import jax.numpy as jnp
import numpy as np
from unxt import Quantity

from tnt.orbit_library import (
    OrbitInitialConditions,
    OrbitIntegrationSettings,
    build_orbit_library,
)


def test_nested_mge_like_potential_integrates_into_normalized_radial_fractions():
    gaussians = gp.CompositePotential({
        "0": gp.TriaxialGaussianPotential(
            m_tot=Quantity(1e10, "Msun"), r_s=Quantity(1.0, "kpc"),
            q1=0.8, q2=0.6, units="galactic",
        ),
        "1": gp.TriaxialGaussianPotential(
            m_tot=Quantity(2e10, "Msun"), r_s=Quantity(3.0, "kpc"),
            q1=0.9, q2=0.7, units="galactic",
        ),
    }, units="galactic")
    potential = gp.CompositePotential({
        "stars": gaussians,
        "central_mass": gp.PlummerPotential(
            m_tot=Quantity(1e7, "Msun"), r_s=Quantity(0.01, "kpc"), units="galactic"
        ),
    }, units="galactic")
    starts = OrbitInitialConditions(
        positions=Quantity(jnp.array([[[1.0, 0.2, 0.1]]]), "kpc"),
        velocities=Quantity(jnp.array([[[0.0, 0.1, 0.0]]]), "kpc/Myr"),
        reference_periods=Quantity(jnp.array([[10.0]]), "Myr"),
        bundle_ids=jnp.array([0]),
    )

    def radial_fractions(trajectory):
        radii = jnp.linalg.norm(trajectory.positions.ustrip("kpc"), axis=-1)
        counts, _ = jnp.histogram(radii, jnp.array([0.0, 0.5, 1.0, 2.0, jnp.inf]))
        return counts / radii.size

    library = build_orbit_library(
        potential, starts, {"radial_fractions": radial_fractions}, jax.random.key(1),
        OrbitIntegrationSettings(
            orbital_periods=1, n_stored_timesteps=128, accuracy=1e-9
        ),
    )[1.0]
    assert bool(jnp.all(library.diagnostics.successful))
    assert float(library.diagnostics.max_energy_error[0, 0]) < 1e-6
    assert library.contributions["radial_fractions"].shape == (1, 4)
    np.testing.assert_allclose(library.contributions["radial_fractions"].sum(), 1.0)
