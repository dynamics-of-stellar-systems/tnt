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


@pytest.mark.parametrize("method", ["linear", "cubic", "hermite"])
def test_plummer_off_node_force_converges_and_boundaries_are_inclusive(method):
    radii = np.geomspace(0.013, 87, 75)
    directions = np.array([[1, 2, 3], [-3, 0, 1], [0, -1, 0]])
    directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    points = (radii[:, None, None] * directions[None]).reshape(-1, 3)
    expected = -points / (1 + np.sum(points**2, axis=1))[:, None] ** 1.5
    errors = []
    for n in (16, 32, 64):
        grid = build_grid(
            plummer(), GridSpec((n, 5, 5), 0.01, 100, interpolation=method)
        )
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


@pytest.mark.parametrize("method", ["linear", "cubic", "hermite"])
def test_gaussian_node_planes_reflection_and_units(method):
    model = gp.TriaxialGaussianPotential(
        m_tot=Quantity(1e10, "Msun"), r_s=Quantity(1.0, "kpc"),
        q1=0.8, q2=0.6, integration_order=100, units="galactic",
    )
    grid = build_grid(model, GridSpec((17, 9, 9), 0.01, 100, interpolation=method))
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
@pytest.mark.parametrize("method", ["linear", "cubic", "hermite"])
def test_grid_rejects_invalid_queries_under_jit(point, method):
    grid = build_grid(plummer(), GridSpec((8, 4, 4), 0.01, 100, interpolation=method))
    with pytest.raises(Exception, match="outside its declared radius range"):
        jax.jit(lambda p: grid.acceleration(
            Quantity(p, "kpc"), Quantity(0.0, "Myr")
        ).value)(jnp.array(point)).block_until_ready()


@pytest.mark.parametrize("method", ["linear", "cubic", "hermite"])
def test_grid_uses_production_solver_and_direct_energy(method):
    model = plummer()
    grid = build_grid(model, GridSpec((128, 4, 4), 0.05, 5, interpolation=method))
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


def test_cubic_matches_independent_scipy_splines_and_has_continuous_slopes():
    from scipy.interpolate import CubicSpline

    from benchmarks.force_grid import _cubic_value, _spline_coefficients

    rng = np.random.default_rng(85)
    nodes = rng.normal(size=(7, 6, 5, 3))
    coefficients = _spline_coefficients(jnp.asarray(nodes))
    points = rng.uniform([0, 0, 0], [6, 5, 4], size=(20, 3))
    points = np.vstack([points, [0, 0, 0], [6, 5, 4], [2, 3, 1]])
    for point in points:
        expected = nodes
        for axis, coordinate in enumerate(point):
            boundary = "natural" if axis == 0 else "clamped"
            expected = CubicSpline(np.arange(expected.shape[0]), expected,
                                   axis=0, bc_type=boundary)(coordinate)
        actual = _cubic_value(coefficients, jnp.asarray(point))
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    derivative = jax.jit(jax.jacfwd(lambda x: _cubic_value(coefficients, x)))
    for axis in range(3):
        center = jnp.array([2., 3., 1.])
        shift = jnp.eye(3)[axis] * 1e-8
        np.testing.assert_allclose(derivative(center - shift),
                                   derivative(center + shift), atol=1e-6)
    for axis, maximum in [(1, 5), (2, 4)]:
        for endpoint in (0, maximum):
            point = jnp.array([2.3, 2.1, 1.4]).at[axis].set(endpoint)
            np.testing.assert_allclose(derivative(point)[:, axis], 0, atol=1e-12)


def test_cubic_spec_rejects_unknown_methods_and_small_grids():
    with pytest.raises(ValueError, match="four nodes"):
        GridSpec((8, 3, 4), 0.01, 100, interpolation="cubic")
    with pytest.raises(ValueError, match="linear or cubic"):
        GridSpec((8, 4, 4), 0.01, 100, interpolation="unknown")


@pytest.mark.parametrize("method", ["cubic", "hermite"])
def test_cubic_table_round_trip(tmp_path, method):
    from benchmarks.grid_benchmark import construct
    from benchmarks.grid_orbits import load_grid

    model = plummer()
    original, setup = construct(
        model, GridSpec((8, 5, 4), 0.01, 100, interpolation=method),
        tmp_path / "cubic.npz",
    )
    restored, _ = load_grid(model, setup, tmp_path)
    assert restored.spec.interpolation == method
    points = np.array([[0.1, 0.2, -0.3], [3., 0., 0.]])
    np.testing.assert_array_equal(acceleration(original, points),
                                  acceleration(restored, points))


@pytest.mark.parametrize("model", [plummer(), gp.TriaxialGaussianPotential(
    m_tot=Quantity(1., "Msun"), r_s=Quantity(1., "kpc"),
    q1=0.8, q2=0.6, integration_order=30, units="galactic",
)])
def test_hermite_nodes_symmetry_axes_and_gradient(model):
    import equinox as eqx

    grid = build_grid(model, GridSpec((5, 5, 5), 0.1, 10, interpolation="hermite"))
    theta, phi = np.pi / 8, np.pi / 4
    point = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),
                      np.cos(theta)])
    points = np.vstack([point, [1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    np.testing.assert_allclose(acceleration(grid, points), acceleration(model, points),
                               rtol=2e-10, atol=1e-20)
    q = Quantity(points, "kpc")
    np.testing.assert_allclose(
        grid.interpolated_potential(q, Quantity(0., "Myr")).value,
                               model.potential(q, Quantity(0., "Myr")).value,
                               rtol=2e-12)
    off = jnp.array([0.43, 0.32, 0.71])
    energy = lambda x: grid.interpolated_potential(
        Quantity(x, "kpc"), Quantity(0., "Myr")
    ).ustrip("kpc2/Myr2")
    gradient = eqx.filter_jit(jax.grad(energy))
    np.testing.assert_allclose(-gradient(off), acceleration(grid, off), rtol=2e-12)
    for axis in range(3):
        for sign in (-1., 1.):
            endpoint = np.eye(3)[axis] * sign
            nearby = endpoint + np.array([1., 2., 3.]) * 1e-8
            np.testing.assert_allclose(acceleration(grid, nearby),
                                       acceleration(grid, endpoint),
                                       rtol=1e-6,
                                       atol=np.linalg.norm(acceleration(grid, endpoint))
                                       * 1e-6)
    signs = np.array([-1, 1, -1])
    np.testing.assert_allclose(acceleration(grid, off * signs),
                               acceleration(grid, off) * signs, rtol=1e-14)



def test_hermite_cell_continuity():
    from benchmarks.force_grid import _hermite_value

    grid = build_grid(plummer(), GridSpec((9, 5, 5), 0.1, 10,
                                          interpolation="hermite"))
    value = lambda x: _hermite_value(grid.log_coefficients, x)
    gradient = jax.jit(jax.grad(value))
    for axis in range(3):
        point = jnp.array([3.2, 1.3, 2.4]).at[axis].set(2.)
        shift = jnp.eye(3)[axis] * 1e-8
        np.testing.assert_allclose(value(point - shift), value(point + shift),
                                   rtol=1e-7, atol=1e-14)
        np.testing.assert_allclose(gradient(point - shift), gradient(point + shift),
                                   rtol=1e-6, atol=1e-14)


def test_hermite_reproduces_tricubic_polynomial_with_mixed_derivatives():
    from benchmarks.force_grid import _hermite_value

    # Independent exact derivatives, including nonzero mixed third derivatives.
    powers = np.array([3, 2, 3])
    table = np.empty((4, 4, 4, 2, 2, 2))
    for index in np.ndindex(4, 4, 4):
        for orders in np.ndindex(2, 2, 2):
            orders = np.array(orders)
            table[(*index, *orders)] = np.prod(
                np.where(orders, powers, 1) * np.array(index)**(powers - orders)
            )
    points = np.random.default_rng(4).uniform(0.05, 2.95, size=(25, 3))
    actual = jax.vmap(lambda x: _hermite_value(jnp.asarray(table), x))(points)
    np.testing.assert_allclose(actual, np.prod(points**powers, axis=1), rtol=2e-12)


@pytest.mark.parametrize("method", ["linear", "cubic", "hermite"])
def test_direct_fallback_matches_direct_outside_and_preserves_inside(method):
    import equinox as eqx

    model = plummer()
    grid = build_grid(model, GridSpec((12, 5, 5), .1, 10,
                                      interpolation=method, outside_policy="direct"))
    from benchmarks.force_grid import ForceGrid

    strict = ForceGrid(model, grid.log_coefficients,
                       GridSpec((12, 5, 5), .1, 10, interpolation=method))
    points = np.array([[0., 0., 0.], [.01, 0., 0.], [20., -2., 1.]])
    evaluate = eqx.filter_jit(lambda g, x: acceleration_quantity(g, x))
    np.testing.assert_allclose(evaluate(grid, jnp.array(points)),
                               acceleration(model, points), rtol=1e-14, atol=1e-14)
    inside = np.array([[.1, 0., 0.], [.7, -.3, .4], [0., 10., 0.]])
    np.testing.assert_allclose(evaluate(grid, jnp.array(inside)),
                               acceleration(strict, inside), rtol=1e-14, atol=1e-14)
    if method == "hermite":
        np.testing.assert_allclose(
            grid.interpolated_potential(
                Quantity(points, "kpc"), Quantity(0., "Myr")
            ).value,
            model.potential(Quantity(points, "kpc"), Quantity(0., "Myr")).value,
            rtol=1e-14,
        )
    with pytest.raises(Exception, match="must be finite"):
        evaluate(grid, jnp.array([jnp.nan, 0., 0.])).block_until_ready()


def acceleration_quantity(model, xyz):
    return model.acceleration(Quantity(xyz, "kpc"), Quantity(0., "Myr")).value


@pytest.mark.parametrize("method", ["linear", "cubic", "hermite"])
def test_outside_orbit_integrates_using_direct_fallback(method):
    model = plummer()
    grid = build_grid(model, GridSpec((8, 4, 4), .1, 1.,
                                      interpolation=method, outside_policy="direct"))
    speed = np.sqrt(9 / 10**1.5)
    settings = OrbitIntegrationSettings(orbital_periods=1., n_stored_timesteps=32)
    args = (Quantity([3., 0., 0.], "kpc"), Quantity([0., speed, 0.], "kpc/Myr"),
            Quantity(6*np.pi/speed, "Myr"), jax.random.key(4), settings)
    reference = integrate_orbit(model, *args)
    actual = integrate_grid(grid, *args)
    assert bool(actual.diagnostics.successful)
    np.testing.assert_allclose(actual.positions.value, reference.positions.value,
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual.velocities.value, reference.velocities.value,
                               rtol=1e-10, atol=1e-10)
