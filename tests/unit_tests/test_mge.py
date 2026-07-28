import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import unxt as u

from tnt.mge import (
    Deprojected3DMGE,
    LightMGE,
    MassMGE,
    SphericalGrid,
    build_mges,
    read_mge,
)

FIXTURES_DIR = Path(__file__).parents[1] / "integration_tests" / "fixtures"


def _internal_unit_system() -> u.AbstractUnitSystem:
    return u.unitsystem("kpc", "Myr", "Msun", "rad", "Lsun")


def test_read_converts_light_columns_to_unit_system():
    unit_system = _internal_unit_system()

    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", unit_system)

    assert mge.I.unit == u.unit("Lsun / rad2")
    assert mge.sigma.unit == u.unit("rad")
    assert mge.q.unit == u.unit("")
    assert mge.PA_twist.unit == u.unit("rad")
    assert jnp.allclose(
        mge.q.ustrip(""),
        jnp.array([0.89541, 0.79093, 0.9999, 0.55097, 0.9999, 0.55097]),
    )


def test_read_converts_mass_columns_to_unit_system():
    unit_system = _internal_unit_system()

    mge = MassMGE.read(FIXTURES_DIR / "mge_mass.ecsv", unit_system)

    assert mge.I.unit == u.unit("Msun / rad2")
    assert mge.sigma.unit == u.unit("rad")
    assert mge.q.unit == u.unit("")
    assert mge.PA_twist.unit == u.unit("rad")
    assert jnp.allclose(
        mge.q.ustrip(""),
        jnp.array([0.91205, 0.83017, 0.9999, 0.60214, 0.9999, 0.60214]),
    )


def test_read_mge_infers_light_kind():
    mge = read_mge(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())

    assert isinstance(mge, LightMGE)
    assert mge.I.unit == u.unit("Lsun / rad2")


def test_read_mge_infers_mass_kind():
    mge = read_mge(FIXTURES_DIR / "mge_mass.ecsv", _internal_unit_system())

    assert isinstance(mge, MassMGE)
    assert mge.I.unit == u.unit("Msun / rad2")


def test_build_mges_reads_each_named_file():
    mges = build_mges(
        {"light": "mge_lum.ecsv", "mass": "mge_mass.ecsv"},
        FIXTURES_DIR,
        _internal_unit_system(),
    )

    assert isinstance(mges["light"], LightMGE)
    assert isinstance(mges["mass"], MassMGE)


def test_build_mges_without_entries_returns_empty_dict():
    assert build_mges({}, FIXTURES_DIR, _internal_unit_system()) == {}


@pytest.mark.parametrize("bad_q", [0.0, -0.5, 1.5])
def test_read_rejects_q_out_of_range(tmp_path, bad_q):
    bad_file = tmp_path / "bad_q.ecsv"
    bad_file.write_text(
        "# %ECSV 0.9\n"
        "# ---\n"
        "# datatype:\n"
        "# - {name: I, unit: Lsun / arcsec2, datatype: float64}\n"
        "# - {name: sigma, unit: arcsec, datatype: float64}\n"
        "# - {name: q, unit: '', datatype: float64}\n"
        "# - {name: PA_twist, unit: deg, datatype: float64}\n"
        "# schema: astropy-2.0\n"
        "I sigma q PA_twist\n"
        f"1.0 1.0 {bad_q} 0.0\n"
    )

    with pytest.raises(ValueError, match="q must satisfy 0 < q <= 1"):
        LightMGE.read(bad_file, _internal_unit_system())


def test_read_accepts_q_equal_to_one(tmp_path):
    ok_file = tmp_path / "q_one.ecsv"
    ok_file.write_text(
        "# %ECSV 0.9\n"
        "# ---\n"
        "# datatype:\n"
        "# - {name: I, unit: Lsun / arcsec2, datatype: float64}\n"
        "# - {name: sigma, unit: arcsec, datatype: float64}\n"
        "# - {name: q, unit: '', datatype: float64}\n"
        "# - {name: PA_twist, unit: deg, datatype: float64}\n"
        "# schema: astropy-2.0\n"
        "I sigma q PA_twist\n"
        "1.0 1.0 1.0 0.0\n"
    )

    mge = LightMGE.read(ok_file, _internal_unit_system())

    assert jnp.allclose(mge.q.ustrip(""), 1.0)


def test_read_mge_rejects_unrecognized_units(tmp_path):
    bad_file = tmp_path / "bad.ecsv"
    bad_file.write_text(
        "# %ECSV 0.9\n"
        "# ---\n"
        "# datatype:\n"
        "# - {name: I, unit: s, datatype: float64}\n"
        "# - {name: sigma, unit: arcsec, datatype: float64}\n"
        "# - {name: q, unit: '', datatype: float64}\n"
        "# - {name: PA_twist, unit: deg, datatype: float64}\n"
        "# schema: astropy-2.0\n"
        "I sigma q PA_twist\n"
        "1.0 1.0 1.0 0.0\n"
    )

    with pytest.raises(ValueError, match="Could not infer MGE kind"):
        read_mge(bad_file, _internal_unit_system())


def test_to_mass_with_constant_ratio():
    light = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    m_over_l = u.Quantity(2.5, "Msun / Lsun")

    mass = light.to_mass(m_over_l)

    assert isinstance(mass, MassMGE)
    assert mass.I.unit == u.unit("Msun / rad2")
    assert jnp.allclose(
        mass.I.ustrip("Msun / rad2"), light.I.ustrip("Lsun / rad2") * 2.5
    )
    assert jnp.allclose(mass.sigma.ustrip("rad"), light.sigma.ustrip("rad"))
    assert jnp.allclose(mass.q.ustrip(""), light.q.ustrip(""))
    assert jnp.allclose(mass.PA_twist.ustrip("rad"), light.PA_twist.ustrip("rad"))


def test_to_mass_with_per_component_ratio():
    light = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    ratios = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    m_over_l = u.Quantity(ratios, "Msun / Lsun")

    mass = light.to_mass(m_over_l)

    assert isinstance(mass, MassMGE)
    assert jnp.allclose(
        mass.I.ustrip("Msun / rad2"), light.I.ustrip("Lsun / rad2") * ratios
    )


def test_to_mass_rejects_mismatched_component_count():
    light = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    m_over_l = u.Quantity(jnp.array([1.0, 2.0]), "Msun / Lsun")

    with pytest.raises(ValueError, match="m_over_l has 2 components"):
        light.to_mass(m_over_l)


def test_deproject_axisymmetric_edge_on_recovers_observed_q():
    distance = u.Quantity(30.5, "Mpc")
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    physical = mge.angular_to_physical(distance)

    deprojected = physical.deproject_axisymmetric(u.Quantity(90.0, "deg"))

    assert isinstance(deprojected, Deprojected3DMGE)
    assert jnp.allclose(deprojected.q.ustrip(""), physical.q.ustrip(""))
    assert jnp.allclose(deprojected.p.ustrip(""), 1.0)
    assert jnp.allclose(deprojected.sigma.ustrip("Mpc"), physical.sigma.ustrip("Mpc"))


def test_deproject_axisymmetric_conserves_total_flux():
    distance = u.Quantity(30.5, "Mpc")
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    physical = mge.angular_to_physical(distance)
    inclination = u.Quantity(60.0, "deg")

    deprojected = physical.deproject_axisymmetric(inclination)

    sigma = physical.sigma.ustrip("Mpc")
    q_obs = physical.q.ustrip("")
    flux_2d = 2 * jnp.pi * sigma**2 * q_obs * physical.I.ustrip("Lsun / Mpc2")

    q_intr = deprojected.q.ustrip("")
    mass_3d = (
        (2 * jnp.pi) ** 1.5 * sigma**3 * q_intr * deprojected.I.ustrip("Lsun / Mpc3")
    )

    assert jnp.allclose(flux_2d, mass_3d, rtol=1e-5)


def test_deproject_axisymmetric_requires_physical_units():
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())

    with pytest.raises(ValueError, match="physical .length. sigma"):
        mge.deproject_axisymmetric(u.Quantity(90.0, "deg"))


def test_deproject_axisymmetric_requires_zero_pa_twist():
    distance = u.Quantity(30.5, "Mpc")
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    physical = mge.angular_to_physical(distance)
    twisted = LightMGE(
        I=physical.I,
        sigma=physical.sigma,
        q=physical.q,
        PA_twist=u.Quantity(jnp.full(physical.q.shape, 0.1), "rad"),
    )

    with pytest.raises(ValueError, match="PA_twist == 0"):
        twisted.deproject_axisymmetric(u.Quantity(90.0, "deg"))


def test_deproject_axisymmetric_invalid_inclination_gives_nan():
    distance = u.Quantity(30.5, "Mpc")
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    physical = mge.angular_to_physical(distance)

    # Smallest q in the fixture is ~0.55, so an inclination close to face-on
    # (cos(i) close to 1) makes deprojection impossible for that component.
    deprojected = physical.deproject_axisymmetric(u.Quantity(5.0, "deg"))

    assert jnp.any(jnp.isnan(deprojected.q.ustrip("")))


def _single_component_light_mge(q_obs: float, psi: float) -> LightMGE:
    return LightMGE(
        I=u.Quantity(jnp.array([5.0]), "Lsun / kpc2"),
        sigma=u.Quantity(jnp.array([2.0]), "kpc"),
        q=u.Quantity(jnp.array([q_obs]), ""),
        PA_twist=u.Quantity(jnp.array([psi]), "rad"),
    )


def test_deproject_triaxial_circular_projection_gives_sphere():
    # A perfectly circular projected Gaussian (q_obs=1) must deproject to a
    # sphere (p=q=1) regardless of viewing angle -- delta=1-q_obs**2=0 makes
    # both eq. 7 and eq. 8's numerators vanish.
    mge = _single_component_light_mge(q_obs=1.0, psi=0.3)

    deprojected = mge.deproject_triaxial(
        theta=u.Quantity(1.2, "rad"),
        phi=u.Quantity(0.7, "rad"),
        psi=u.Quantity(0.0, "rad"),
    )

    assert jnp.allclose(deprojected.p.ustrip(""), 1.0)
    assert jnp.allclose(deprojected.q.ustrip(""), 1.0)


def test_deproject_triaxial_gives_valid_axial_ratios():
    # A viewing geometry known (numerically, in a sandbox scan) to give a
    # physically valid solution: 0 < q <= p <= 1.
    mge = _single_component_light_mge(q_obs=0.9, psi=-1.0)

    deprojected = mge.deproject_triaxial(
        theta=u.Quantity(0.3, "rad"),
        phi=u.Quantity(0.96, "rad"),
        psi=u.Quantity(0.0, "rad"),
    )

    p = deprojected.p.ustrip("")
    q = deprojected.q.ustrip("")
    assert jnp.allclose(p, 0.8995, atol=1e-4)
    assert jnp.allclose(q, 0.8495, atol=1e-4)
    assert jnp.all((q > 0) & (q <= p) & (p <= 1))


def test_deproject_triaxial_conserves_total_flux():
    mge = _single_component_light_mge(q_obs=0.9, psi=-1.0)

    deprojected = mge.deproject_triaxial(
        theta=u.Quantity(0.3, "rad"),
        phi=u.Quantity(0.96, "rad"),
        psi=u.Quantity(0.0, "rad"),
    )

    sigma_obs = mge.sigma.ustrip("kpc")
    flux_2d = (
        2 * jnp.pi * sigma_obs**2 * mge.q.ustrip("") * mge.I.ustrip("Lsun / kpc2")
    )

    p, q = deprojected.p.ustrip(""), deprojected.q.ustrip("")
    sigma_intr = deprojected.sigma.ustrip("kpc")
    mass_3d = (
        (2 * jnp.pi) ** 1.5
        * sigma_intr**3
        * p
        * q
        * deprojected.I.ustrip("Lsun / kpc3")
    )

    assert jnp.allclose(flux_2d, mass_3d, rtol=1e-5)


def test_deproject_triaxial_requires_physical_units():
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())

    with pytest.raises(ValueError, match="physical .length. sigma"):
        mge.deproject_triaxial(
            theta=u.Quantity(1.0, "rad"),
            phi=u.Quantity(1.0, "rad"),
            psi=u.Quantity(0.0, "rad"),
        )


def _forward_project_triaxial(
    sigma: float, p: float, q: float, theta: float, phi: float
):
    """Independently project a triaxial Gaussian (numpy, no tnt code involved).

    Builds the projected covariance by rotating the intrinsic covariance
    into the (x', y', LOS) frame -- x' in the (x, y) plane, y' such that
    the z-axis projects onto y' (Cappellari 2002 / van den Bosch et al.
    2008's coordinate convention) -- then reads off the projected sigma,
    axial ratio, and position angle from that 2x2 sub-block. This gives
    ground-truth (sigma_obs, q_obs, psi') values for round-trip testing
    `deproject_triaxial`, independent of eqs. 6-9 themselves.
    """
    cov = np.diag([sigma**2, (p * sigma) ** 2, (q * sigma) ** 2])
    n = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    z_hat = np.array([0.0, 0.0, 1.0])
    x_prime = np.cross(n, z_hat)
    x_prime /= np.linalg.norm(x_prime)
    y_prime = np.cross(n, x_prime)
    if abs(np.dot(z_hat - np.dot(z_hat, n) * n, x_prime)) > 1e-8:
        y_prime = np.cross(x_prime, n)

    rotation = np.stack([x_prime, y_prime, n])
    projected_cov = (rotation @ cov @ rotation.T)[:2, :2]
    eigvals, eigvecs = np.linalg.eigh(projected_cov)
    sigma_obs = np.sqrt(eigvals[1])
    q_obs = np.sqrt(eigvals[0]) / sigma_obs
    x_comp, y_comp = eigvecs[:, 1]
    # Position angle measured counterclockwise from y' to the major axis,
    # in Cappellari/van den Bosch's convention -- opposite handedness to
    # the "mathematical" atan2(x_comp, y_comp).
    psi_prime = -np.arctan2(x_comp, y_comp)
    return sigma_obs, q_obs, psi_prime


@pytest.mark.parametrize(
    ("sigma_intr", "p_intr", "q_intr", "theta", "phi"),
    [
        (2.0, 0.7, 0.5, 0.9, 0.4),
        (3.0, 0.6, 0.3, 1.3, 2.1),
        (1.0, 0.9, 0.85, 0.5, 0.5),
    ],
)
def test_deproject_triaxial_recovers_independent_forward_projection(
    sigma_intr, p_intr, q_intr, theta, phi
):
    sigma_obs, q_obs, psi_prime = _forward_project_triaxial(
        sigma_intr, p_intr, q_intr, theta, phi
    )
    mge = LightMGE(
        I=u.Quantity(jnp.array([5.0]), "Lsun / kpc2"),
        sigma=u.Quantity(jnp.array([sigma_obs]), "kpc"),
        q=u.Quantity(jnp.array([q_obs]), ""),
        PA_twist=u.Quantity(jnp.array([0.0]), "rad"),
    )

    deprojected = mge.deproject_triaxial(
        theta=u.Quantity(theta, "rad"),
        phi=u.Quantity(phi, "rad"),
        psi=u.Quantity(psi_prime, "rad"),
    )

    assert jnp.allclose(deprojected.p.ustrip(""), p_intr, atol=1e-4)
    assert jnp.allclose(deprojected.q.ustrip(""), q_intr, atol=1e-4)
    assert jnp.allclose(deprojected.sigma.ustrip("kpc"), sigma_intr, atol=1e-4)


def test_deproject_triaxial_global_psi_and_pa_twist_are_additive():
    # Only the sum psi + PA_twist enters the deprojection (van den Bosch
    # et al. 2008, eq. 6), so shifting the global psi should give the same
    # result as adding that shift directly to PA_twist with psi=0.
    theta, phi = u.Quantity(0.3, "rad"), u.Quantity(0.96, "rad")

    shifted_psi = _single_component_light_mge(q_obs=0.9, psi=-1.0).deproject_triaxial(
        theta=theta, phi=phi, psi=u.Quantity(0.4, "rad")
    )
    shifted_twist = _single_component_light_mge(
        q_obs=0.9, psi=-1.0 + 0.4
    ).deproject_triaxial(theta=theta, phi=phi, psi=u.Quantity(0.0, "rad"))

    assert jnp.allclose(shifted_psi.p.ustrip(""), shifted_twist.p.ustrip(""))
    assert jnp.allclose(shifted_psi.q.ustrip(""), shifted_twist.q.ustrip(""))


def test_mge_is_frozen():
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())

    with pytest.raises(dataclasses.FrozenInstanceError):
        mge.q = mge.q


def test_mge_is_a_jax_pytree():
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())

    doubled = jax.tree_util.tree_map(lambda leaf: leaf * 2, mge)

    assert jnp.allclose(doubled.q.ustrip(""), mge.q.ustrip("") * 2)


def test_angular_to_physical_converts_sigma_and_intensity():
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    distance = u.Quantity(30.5, "Mpc")

    physical = mge.angular_to_physical(distance)

    assert physical.sigma.unit == u.unit("Mpc")
    assert physical.I.unit == u.unit("Lsun / Mpc2")
    assert jnp.allclose(
        physical.sigma.ustrip("Mpc"),
        distance.ustrip("Mpc") * mge.sigma.ustrip("rad"),
    )
    assert jnp.allclose(
        physical.I.ustrip("Lsun / Mpc2"),
        mge.I.ustrip("Lsun / rad2") / distance.ustrip("Mpc") ** 2,
    )


def test_angular_to_physical_leaves_q_and_pa_twist_unchanged():
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    distance = u.Quantity(30.5, "Mpc")

    physical = mge.angular_to_physical(distance)

    assert jnp.allclose(physical.q.ustrip(""), mge.q.ustrip(""))
    assert jnp.allclose(
        physical.PA_twist.ustrip("rad"), mge.PA_twist.ustrip("rad")
    )


def test_angular_physical_round_trip():
    mge = LightMGE.read(FIXTURES_DIR / "mge_lum.ecsv", _internal_unit_system())
    distance = u.Quantity(30.5, "Mpc")

    round_tripped = mge.angular_to_physical(distance).physical_to_angular(distance)

    assert jnp.allclose(round_tripped.sigma.ustrip("rad"), mge.sigma.ustrip("rad"))
    assert jnp.allclose(
        round_tripped.I.ustrip("Lsun / rad2"), mge.I.ustrip("Lsun / rad2")
    )
    assert jnp.allclose(round_tripped.q.ustrip(""), mge.q.ustrip(""))
    assert jnp.allclose(
        round_tripped.PA_twist.ustrip("rad"), mge.PA_twist.ustrip("rad")
    )


def _analytic_total_mass(
    I: np.ndarray, sigma: np.ndarray, p: np.ndarray, q: np.ndarray  # noqa: E741, N803
) -> np.ndarray:
    return np.sum((2 * np.pi) ** 1.5 * sigma**3 * p * q * I)


def test_spherical_grid_init_shapes_and_bounds():
    grid = SphericalGrid(
        n_r=5,
        n_theta=3,
        n_phi=4,
        r_min=u.Quantity(0.1, "kpc"),
        r_max=u.Quantity(50.0, "kpc"),
    )

    assert (grid.n_r, grid.n_theta, grid.n_phi) == (5, 3, 4)
    assert grid.r_edges.shape == (6,)
    assert grid.cos_theta_edges.shape == (4,)
    assert grid.phi_edges.shape == (5,)

    assert grid.r_edges.ustrip("kpc")[0] == 0.0
    assert jnp.isinf(grid.r_edges.ustrip("kpc")[-1])
    assert jnp.allclose(grid.cos_theta_edges.ustrip("")[0], 1.0)
    assert jnp.allclose(grid.cos_theta_edges.ustrip("")[-1], 0.0)
    assert jnp.allclose(grid.phi_edges.ustrip("rad")[0], 0.0)
    assert jnp.allclose(grid.phi_edges.ustrip("rad")[-1], jnp.pi / 2)


def test_spherical_grid_init_n_r_3_uses_both_r_min_and_r_max():
    # With the minimum allowed n_r=3, the two interior edges are exactly
    # r_min and r_max -- neither bound is silently dropped.
    grid = SphericalGrid(
        n_r=3,
        n_theta=3,
        n_phi=3,
        r_min=u.Quantity(0.1, "kpc"),
        r_max=u.Quantity(50.0, "kpc"),
    )

    assert jnp.allclose(
        grid.r_edges.ustrip("kpc"), jnp.array([0.0, 0.1, 50.0, jnp.inf])
    )


@pytest.mark.parametrize("n_r", [1, 2])
def test_spherical_grid_init_rejects_too_few_radial_bins(n_r):
    with pytest.raises(ValueError, match="n_r must be at least 3"):
        SphericalGrid(
            n_r=n_r,
            n_theta=3,
            n_phi=3,
            r_min=u.Quantity(0.1, "kpc"),
            r_max=u.Quantity(50.0, "kpc"),
        )


def test_spherical_mass_grid_shape():
    mge = Deprojected3DMGE(
        I=u.Quantity(jnp.array([3.0]), "Msun / kpc3"),
        sigma=u.Quantity(jnp.array([2.0]), "kpc"),
        p=u.Quantity(jnp.array([1.0]), ""),
        q=u.Quantity(jnp.array([1.0]), ""),
    )
    grid = SphericalGrid(
        n_r=5,
        n_theta=3,
        n_phi=4,
        r_min=u.Quantity(0.1, "kpc"),
        r_max=u.Quantity(50.0, "kpc"),
    )

    masses = mge.spherical_mass_grid(grid)

    assert masses.shape == (5, 3, 4)
    assert masses.unit == u.unit("Msun")


def test_spherical_mass_grid_conserves_total_mass_for_spherical_component():
    # A spherical (p=q=1) component, checked against the closed-form total mass
    # for a 3D Gaussian: (2 pi)^1.5 sigma^3 p q I.
    mge = Deprojected3DMGE(
        I=u.Quantity(jnp.array([3.0]), "Msun / kpc3"),
        sigma=u.Quantity(jnp.array([2.0]), "kpc"),
        p=u.Quantity(jnp.array([1.0]), ""),
        q=u.Quantity(jnp.array([1.0]), ""),
    )
    grid = SphericalGrid(
        n_r=25,
        n_theta=15,
        n_phi=15,
        r_min=u.Quantity(0.1, "kpc"),
        r_max=u.Quantity(50.0, "kpc"),
    )

    masses = mge.spherical_mass_grid(grid)
    total = 8 * jnp.sum(masses.ustrip("Msun"))

    expected = _analytic_total_mass(
        np.array([3.0]), np.array([2.0]), np.array([1.0]), np.array([1.0])
    )
    assert jnp.allclose(total, expected, rtol=1e-5)


def test_spherical_mass_grid_conserves_total_mass_for_triaxial_multicomponent():
    I = np.array([3.0, 1.5])  # noqa: E741, N806
    sigma = np.array([2.0, 5.0])
    p = np.array([0.8, 0.6])
    q = np.array([0.5, 0.3])
    mge = Deprojected3DMGE(
        I=u.Quantity(jnp.asarray(I), "Msun / kpc3"),
        sigma=u.Quantity(jnp.asarray(sigma), "kpc"),
        p=u.Quantity(jnp.asarray(p), ""),
        q=u.Quantity(jnp.asarray(q), ""),
    )
    grid = SphericalGrid(
        n_r=30,
        n_theta=20,
        n_phi=20,
        r_min=u.Quantity(0.05, "kpc"),
        r_max=u.Quantity(100.0, "kpc"),
    )

    masses = mge.spherical_mass_grid(grid)
    total = 8 * jnp.sum(masses.ustrip("Msun"))

    expected = _analytic_total_mass(I, sigma, p, q)
    assert jnp.allclose(total, expected, rtol=1e-5)


def test_spherical_mass_grid_reusable_across_components_with_different_length_units():
    # The grid's own length unit need not match sigma's -- spherical_mass_grid
    # should convert internally, so the same grid can bin quantities from MGEs
    # expressed in different (but compatible) length units.
    mge = Deprojected3DMGE(
        I=u.Quantity(jnp.array([3.0]), "Msun / pc3"),
        sigma=u.Quantity(jnp.array([2000.0]), "pc"),
        p=u.Quantity(jnp.array([1.0]), ""),
        q=u.Quantity(jnp.array([1.0]), ""),
    )
    grid = SphericalGrid(
        n_r=25,
        n_theta=15,
        n_phi=15,
        r_min=u.Quantity(0.1, "kpc"),
        r_max=u.Quantity(50.0, "kpc"),
    )

    masses = mge.spherical_mass_grid(grid)
    total = 8 * jnp.sum(masses.ustrip("Msun"))

    # sigma and I here, in mutually consistent pc-based units.
    expected = _analytic_total_mass(
        np.array([3.0]), np.array([2000.0]), np.array([1.0]), np.array([1.0])
    )
    assert jnp.allclose(total, expected, rtol=1e-5)
