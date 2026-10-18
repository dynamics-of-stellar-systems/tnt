"""Independent limits and rejection checks for the benchmark-only force grid."""

import jax

jax.config.update("jax_enable_x64", True)

import galax.potential as gp
import jax.numpy as jnp
import numpy as np
import pytest
from unxt import Quantity

from benchmarks.force_grid import GridSpec, build_grid, integrate_grid
from tnt.orbit_library import OrbitIntegrationSettings, integrate_orbit


def plummer():
    return gp.PlummerPotential(
        m_tot=Quantity(1.0, "Msun"), r_s=Quantity(1.0, "kpc"),
        constants={"G": Quantity(1.0, "kpc3/(Msun Myr2)")}, units="galactic",
    )


def acceleration(model, xyz):
    return np.asarray(model.acceleration(
        Quantity(xyz, "kpc"), Quantity(0.0, "Myr")
    ).ustrip("kpc/Myr2"))


def test_plummer_off_node_force_converges_and_boundaries_are_inclusive():
    radii = np.geomspace(0.013, 87, 75)
    directions = np.array([[1, 2, 3], [-3, 0, 1], [0, -1, 0]])
    directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    points = (radii[:, None, None] * directions[None]).reshape(-1, 3)
    expected = -points / (1 + np.sum(points**2, axis=1))[:, None] ** 1.5
    errors = []
    for n in (16, 32, 64):
        grid = build_grid(plummer(), GridSpec((n, 5, 5), 0.01, 100))
        force = acceleration(grid, points)
        errors.append(np.max(np.linalg.norm(force - expected, axis=1)
                             / np.linalg.norm(expected, axis=1)))
        boundary = np.array([[0.01, 0, 0], [0, -100, 0]])
        np.testing.assert_allclose(
            acceleration(grid, boundary), acceleration(plummer(), boundary),
            rtol=1e-12, atol=1e-14,
        )
    assert errors[1] < 0.4 * errors[0]
    assert errors[2] < 0.4 * errors[1]


def test_gaussian_node_planes_reflection_and_units():
    model = gp.TriaxialGaussianPotential(
        m_tot=Quantity(1e10, "Msun"), r_s=Quantity(1.0, "kpc"),
        q1=0.8, q2=0.6, integration_order=100, units="galactic",
    )
    grid = build_grid(model, GridSpec((17, 9, 9), 0.01, 100))
    # Exact radial and angular nodes, including coordinate planes and axes.
    points = np.array([
        [1, 0, 0], [0, -1, 0], [0, 0, 1],
        [2**-0.5, 2**-0.5, 0], [0.5, 0.5, 2**-0.5],
    ])
    np.testing.assert_allclose(
        acceleration(grid, points), acceleration(model, points),
        rtol=2e-12, atol=1e-14,
    )
    point = jnp.array([0.4, 0.3, 0.2])
    signs = jnp.array([-1, 1, -1])
    np.testing.assert_array_equal(
        acceleration(grid, point * signs), acceleration(grid, point) * signs
    )
    physical = grid.acceleration(Quantity(point * 1000, "pc"), Quantity(0.0, "s"))
    np.testing.assert_allclose(physical.ustrip("kpc/Myr2"), acceleration(grid, point))
    near = np.array([[1, -1e-10, 0], [1, 1e-10, 0]])
    np.testing.assert_allclose(acceleration(grid, near)[0, 0],
                               acceleration(grid, near)[1, 0], rtol=1e-14)


@pytest.mark.parametrize("point", [[0, 0, 0], [0.001, 0, 0], [101, 0, 0],
                                   [np.nan, 0, 0]])
def test_grid_rejects_invalid_queries_under_jit(point):
    grid = build_grid(plummer(), GridSpec((8, 3, 3), 0.01, 100))
    with pytest.raises(Exception, match="outside its declared radius range"):
        jax.jit(lambda p: grid.acceleration(
            Quantity(p, "kpc"), Quantity(0.0, "Myr")
        ).value)(jnp.array(point)).block_until_ready()


def test_grid_uses_production_solver_and_direct_energy():
    model = plummer()
    grid = build_grid(model, GridSpec((128, 3, 3), 0.05, 5))
    q = Quantity([1.0, 0.0, 0.0], "kpc")
    speed = 2**-0.75
    v = Quantity([0.0, speed, 0.0], "kpc/Myr")
    clock = Quantity(2 * np.pi / speed, "Myr")
    settings = OrbitIntegrationSettings(orbital_periods=1, n_stored_timesteps=64)
    args = (q, v, clock, jax.random.key(7), settings)
    direct, interpolated = integrate_orbit(model, *args), integrate_grid(grid, *args)
    assert bool(interpolated.diagnostics.successful)
    np.testing.assert_array_equal(direct.times.value, interpolated.times.value)
    delta = interpolated.positions.ustrip("kpc") - direct.positions.ustrip("kpc")
    assert float(jnp.sqrt(jnp.mean(delta**2))) < 5e-4
    energies = grid.potential(interpolated.positions, Quantity(0, "Myr"))
    np.testing.assert_array_equal(
        energies.value,
        model.potential(interpolated.positions, Quantity(0, "Myr")).value,
    )
    # The runtime's static-model gate remains closed to this experimental type.
    with pytest.raises(TypeError, match="requires a Galax potential"):
        integrate_orbit(grid, *args)


def test_invalid_spec_and_unsupported_model_are_rejected():
    for shape, low, high in [((1, 3, 3), 1, 2), ((3, 3, 3), 0, 2),
                             ((3, 3, 3), 2, 1)]:
        with pytest.raises(ValueError):
            GridSpec(shape, low, high)
    unsupported = gp.HernquistPotential(
        m_tot=Quantity(1.0, "Msun"), r_s=Quantity(1.0, "kpc"), units="galactic"
    )
    with pytest.raises(NotImplementedError, match="only coaxial"):
        build_grid(unsupported, GridSpec((3, 3, 3), 0.01, 100))
