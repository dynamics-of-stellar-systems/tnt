import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import unxt as u
from scipy import integrate

from tnt.mge import (
    Deprojected3DMGE,
    LightMGE,
    MassMGE,
    build_mges,
    read_mge,
)
from tnt.spatial_binnings import ProjectedBinning, SphericalGrid


def _internal_unit_system() -> u.AbstractUnitSystem:
    return u.unitsystem("kpc", "Myr", "Msun", "rad", "Lsun")


def _write_ecsv(
    path: Path,
    *,
    intensity_unit: str,
    rows: list[tuple[float, float, float, float]],
) -> None:
    """Write a minimal MGE ECSV file with the given intensity unit and rows."""
    header = (
        "# %ECSV 0.9\n"
        "# ---\n"
        "# datatype:\n"
        f"# - {{name: I, unit: {intensity_unit}, datatype: float64}}\n"
        "# - {name: sigma, unit: arcsec, datatype: float64}\n"
        "# - {name: q, unit: '', datatype: float64}\n"
        "# - {name: PA_twist, unit: deg, datatype: float64}\n"
        "# schema: astropy-2.0\n"
        "I sigma q PA_twist\n"
    )
    body = "".join(f"{i} {s} {q} {pa}\n" for i, s, q, pa in rows)
    path.write_text(header + body)


# The multi-component rows below match what used to be a checked-in fixture
# file (tests/integration_tests/fixtures/mge_lum.ecsv / mge_mass.ecsv), now
# defined directly here so unit tests don't reach into another test
# directory's data for something they can just as easily construct inline.
_LIGHT_ROWS = [
    (26819.14, 0.49416, 0.89541, 0.0),
    (2456.39, 2.04299, 0.79093, 0.0),
    (456.8, 2.44313, 0.9999, 0.0),
    (645.49, 6.5305, 0.55097, 0.0),
    (14.73, 17.41488, 0.9999, 0.0),
    (123.85, 21.84711, 0.55097, 0.0),
]
_MASS_ROWS = [
    (54129.63, 0.42169, 0.91205, 0.0),
    (5893.71, 1.98443, 0.83017, 0.0),
    (981.36, 2.61098, 0.9999, 0.0),
    (1120.84, 6.98721, 0.60214, 0.0),
    (32.19, 18.02233, 0.9999, 0.0),
    (201.47, 22.11045, 0.60214, 0.0),
]


def _multi_component_light_mge() -> LightMGE:
    """A multi-component LightMGE with realistic, varied q values.

    Same values as `_LIGHT_ROWS`, converted to radians up front (matching
    what `LightMGE.read` would produce for `_internal_unit_system`'s "rad"
    angle unit) and constructed directly rather than read from a file -- for
    tests that just need some realistic LightMGE to operate on, as opposed to
    testing file-reading behaviour itself.

    `angular_to_physical` only gives correct results for an angle unit of
    exactly "rad" (its `solid_angle` shortcut assumes it), which is why this
    doesn't just store the raw arcsec/deg values directly: real MGEs are
    always converted to "rad" by `.read()` before that method would ever see
    them.
    """
    intensity, sigma, q, pa_twist = zip(*_LIGHT_ROWS, strict=True)
    sigma_arcsec = u.Quantity(jnp.array(sigma), "arcsec")
    intensity_per_arcsec2 = u.Quantity(jnp.array(intensity), "Lsun / arcsec2")
    pa_twist_deg = u.Quantity(jnp.array(pa_twist), "deg")
    return LightMGE(
        I=u.Quantity(intensity_per_arcsec2.ustrip("Lsun / rad2"), "Lsun / rad2"),
        sigma=u.Quantity(sigma_arcsec.ustrip("rad"), "rad"),
        q=u.Quantity(jnp.array(q), ""),
        PA_twist=u.Quantity(pa_twist_deg.ustrip("rad"), "rad"),
    )


def test_read_converts_light_columns_to_unit_system(tmp_path):
    path = tmp_path / "mge_lum.ecsv"
    _write_ecsv(path, intensity_unit="Lsun / arcsec2", rows=_LIGHT_ROWS)
    unit_system = _internal_unit_system()

    mge = LightMGE.read(path, unit_system)

    assert mge.I.unit == u.unit("Lsun / rad2")
    assert mge.sigma.unit == u.unit("rad")
    assert mge.q.unit == u.unit("")
    assert mge.PA_twist.unit == u.unit("rad")
    assert jnp.allclose(
        mge.q.ustrip(""),
        jnp.array([0.89541, 0.79093, 0.9999, 0.55097, 0.9999, 0.55097]),
    )


def test_read_converts_mass_columns_to_unit_system(tmp_path):
    path = tmp_path / "mge_mass.ecsv"
    _write_ecsv(path, intensity_unit="Msun / arcsec2", rows=_MASS_ROWS)
    unit_system = _internal_unit_system()

    mge = MassMGE.read(path, unit_system)

    assert mge.I.unit == u.unit("Msun / rad2")
    assert mge.sigma.unit == u.unit("rad")
    assert mge.q.unit == u.unit("")
    assert mge.PA_twist.unit == u.unit("rad")
    assert jnp.allclose(
        mge.q.ustrip(""),
        jnp.array([0.91205, 0.83017, 0.9999, 0.60214, 0.9999, 0.60214]),
    )


def test_read_mge_infers_light_kind(tmp_path):
    path = tmp_path / "mge.ecsv"
    _write_ecsv(path, intensity_unit="Lsun / arcsec2", rows=[(1.0, 1.0, 0.9, 0.0)])

    mge = read_mge(path, _internal_unit_system())

    assert isinstance(mge, LightMGE)
    assert mge.I.unit == u.unit("Lsun / rad2")


def test_read_mge_infers_mass_kind(tmp_path):
    path = tmp_path / "mge.ecsv"
    _write_ecsv(path, intensity_unit="Msun / arcsec2", rows=[(1.0, 1.0, 0.9, 0.0)])

    mge = read_mge(path, _internal_unit_system())

    assert isinstance(mge, MassMGE)
    assert mge.I.unit == u.unit("Msun / rad2")


def test_build_mges_reads_each_named_file(tmp_path):
    _write_ecsv(
        tmp_path / "light.ecsv",
        intensity_unit="Lsun / arcsec2",
        rows=[(1.0, 1.0, 0.9, 0.0)],
    )
    _write_ecsv(
        tmp_path / "mass.ecsv",
        intensity_unit="Msun / arcsec2",
        rows=[(1.0, 1.0, 0.9, 0.0)],
    )

    mges = build_mges(
        {"light": "light.ecsv", "mass": "mass.ecsv"},
        tmp_path,
        _internal_unit_system(),
    )

    assert isinstance(mges["light"], LightMGE)
    assert isinstance(mges["mass"], MassMGE)


def test_build_mges_without_entries_returns_empty_dict(tmp_path):
    assert build_mges({}, tmp_path, _internal_unit_system()) == {}


@pytest.mark.parametrize("bad_q", [0.0, -0.5, 1.5])
def test_read_rejects_q_out_of_range(tmp_path, bad_q):
    bad_file = tmp_path / "bad_q.ecsv"
    _write_ecsv(
        bad_file, intensity_unit="Lsun / arcsec2", rows=[(1.0, 1.0, bad_q, 0.0)]
    )

    with pytest.raises(ValueError, match="q must satisfy 0 < q <= 1"):
        LightMGE.read(bad_file, _internal_unit_system())


def test_read_accepts_q_equal_to_one(tmp_path):
    ok_file = tmp_path / "q_one.ecsv"
    _write_ecsv(ok_file, intensity_unit="Lsun / arcsec2", rows=[(1.0, 1.0, 1.0, 0.0)])

    mge = LightMGE.read(ok_file, _internal_unit_system())

    assert jnp.allclose(mge.q.ustrip(""), 1.0)


def test_read_mge_rejects_unrecognized_units(tmp_path):
    bad_file = tmp_path / "bad.ecsv"
    _write_ecsv(bad_file, intensity_unit="s", rows=[(1.0, 1.0, 1.0, 0.0)])

    with pytest.raises(ValueError, match="Could not infer MGE kind"):
        read_mge(bad_file, _internal_unit_system())


def test_to_mass_with_constant_ratio():
    light = _multi_component_light_mge()
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
    light = _multi_component_light_mge()
    ratios = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    m_over_l = u.Quantity(ratios, "Msun / Lsun")

    mass = light.to_mass(m_over_l)

    assert isinstance(mass, MassMGE)
    assert jnp.allclose(
        mass.I.ustrip("Msun / rad2"), light.I.ustrip("Lsun / rad2") * ratios
    )


def test_to_mass_rejects_mismatched_component_count():
    light = _multi_component_light_mge()
    m_over_l = u.Quantity(jnp.array([1.0, 2.0]), "Msun / Lsun")

    with pytest.raises(ValueError, match="m_over_l has 2 components"):
        light.to_mass(m_over_l)


def test_deproject_axisymmetric_edge_on_recovers_observed_q():
    distance = u.Quantity(30.5, "Mpc")
    mge = _multi_component_light_mge()
    physical = mge.angular_to_physical(distance)

    deprojected = physical.deproject_axisymmetric(u.Quantity(90.0, "deg"))

    assert isinstance(deprojected, Deprojected3DMGE)
    assert jnp.allclose(deprojected.q.ustrip(""), physical.q.ustrip(""))
    assert jnp.allclose(deprojected.p.ustrip(""), 1.0)
    assert jnp.allclose(deprojected.sigma.ustrip("Mpc"), physical.sigma.ustrip("Mpc"))


def test_deproject_axisymmetric_conserves_total_flux():
    distance = u.Quantity(30.5, "Mpc")
    mge = _multi_component_light_mge()
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
    mge = _multi_component_light_mge()

    with pytest.raises(ValueError, match="physical .length. sigma"):
        mge.deproject_axisymmetric(u.Quantity(90.0, "deg"))


def test_deproject_axisymmetric_requires_zero_pa_twist():
    distance = u.Quantity(30.5, "Mpc")
    mge = _multi_component_light_mge()
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
    mge = _multi_component_light_mge()
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
    mge = _multi_component_light_mge()

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
    mge = _multi_component_light_mge()

    with pytest.raises(dataclasses.FrozenInstanceError):
        mge.q = mge.q


def test_mge_is_a_jax_pytree():
    mge = _multi_component_light_mge()

    doubled = jax.tree_util.tree_map(lambda leaf: leaf * 2, mge)

    assert jnp.allclose(doubled.q.ustrip(""), mge.q.ustrip("") * 2)


def test_angular_to_physical_converts_sigma_and_intensity():
    mge = _multi_component_light_mge()
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
    mge = _multi_component_light_mge()
    distance = u.Quantity(30.5, "Mpc")

    physical = mge.angular_to_physical(distance)

    assert jnp.allclose(physical.q.ustrip(""), mge.q.ustrip(""))
    assert jnp.allclose(
        physical.PA_twist.ustrip("rad"), mge.PA_twist.ustrip("rad")
    )


_PROJECTED_MASS_QUAD_ORDER = 10


def _projected_binning(
    *, min_x, min_y, x_extent, y_extent, pa, bins
) -> ProjectedBinning:
    return ProjectedBinning.from_settings(
        {
            "min_x": {"value": min_x, "unit": "rad"},
            "min_y": {"value": min_y, "unit": "rad"},
            "x_extent": {"value": x_extent, "unit": "rad"},
            "y_extent": {"value": y_extent, "unit": "rad"},
            "PA": {"value": pa, "unit": "rad"},
        },
        bins,
        _internal_unit_system(),
        _PROJECTED_MASS_QUAD_ORDER,
    )


def _brute_force_aperture_mass(  # noqa: N803
    I, sigma, q, pa_twist, pa, x_edges, y_edges
):
    """Independently integrate a multi-component MGE over a pixel grid.

    Uses `scipy.integrate.dblquad` directly on each component's surface
    density, rotated into the pixel grid's frame by hand (no tnt code
    involved), as ground truth for `AbstractMGE.get_projected_mass`.
    """
    n_x, n_y = len(x_edges) - 1, len(y_edges) - 1
    mass = np.zeros((n_x, n_y))
    for k in range(len(I)):
        alpha = pa - np.pi / 2 + pa_twist[k]

        def surface_density(x, y, k=k, alpha=alpha):
            x_major = x * np.cos(alpha) + y * np.sin(alpha)
            y_minor = -x * np.sin(alpha) + y * np.cos(alpha)
            return I[k] * np.exp(
                -(x_major**2 + (y_minor / q[k]) ** 2) / (2 * sigma[k] ** 2)
            )

        for i in range(n_x):
            for j in range(n_y):
                val, _ = integrate.dblquad(
                    lambda y, x, sd=surface_density: sd(x, y),
                    x_edges[i],
                    x_edges[i + 1],
                    y_edges[j],
                    y_edges[j + 1],
                    epsabs=1e-14,
                    epsrel=1e-12,
                )
                mass[i, j] += val
    return mass


@pytest.mark.parametrize(
    ("I", "sigma", "q", "pa_twist", "pa"),
    [
        ([3.0], [0.02], [0.4], [0.3], 1.1),
        ([3.0], [0.02], [1.0], [0.0], 0.0),
        ([2.0, 4.0], [0.015, 0.03], [0.6, 0.3], [0.0, 0.5], 0.7),
    ],
)
def test_get_projected_mass_matches_independent_numeric_integral(  # noqa: N803
    I, sigma, q, pa_twist, pa
):
    mge = LightMGE(
        I=u.Quantity(jnp.array(I), "Lsun / rad2"),
        sigma=u.Quantity(jnp.array(sigma), "rad"),
        q=u.Quantity(jnp.array(q), ""),
        PA_twist=u.Quantity(jnp.array(pa_twist), "rad"),
    )
    n_x, n_y = 4, 3
    min_x, min_y = -0.05, -0.04
    x_extent, y_extent = 0.1, 0.08
    x_edges = min_x + np.linspace(0, x_extent, n_x + 1)
    y_edges = min_y + np.linspace(0, y_extent, n_y + 1)
    bins = 1 + np.arange(n_x * n_y).reshape(n_x, n_y)
    binning = _projected_binning(
        min_x=min_x,
        min_y=min_y,
        x_extent=x_extent,
        y_extent=y_extent,
        pa=pa,
        bins=bins,
    )

    mass = mge.get_projected_mass(binning)

    expected_grid = _brute_force_aperture_mass(
        I, sigma, q, pa_twist, pa, x_edges, y_edges
    )
    expected = expected_grid.ravel()[np.argsort(bins.ravel())]
    assert mass.unit == u.unit("Lsun")
    assert jnp.allclose(mass.ustrip("Lsun"), expected, rtol=1e-5)


@pytest.mark.parametrize(
    ("pa", "aligned_bin_idx"),
    [
        (0.0, 0),  # PA=0 -> major axis along y -> the y-strip bin gets more mass.
        (np.pi / 2, 1),  # PA=90deg -> major axis along x -> the x-strip bin does.
    ],
)
def test_get_projected_mass_pa_convention_matches_documented_axis(pa, aligned_bin_idx):
    """PA is measured counterclockwise from the y-axis (docstring/configuration.md).

    An elongated component (small `q`) with no twist should therefore have its
    major axis along y at PA=0 and along x at PA=90deg -- checked here by
    comparing the mass caught by a thin strip along each axis, independently
    of the erf/quadrature integration formula itself.
    """
    mge = LightMGE(
        I=u.Quantity(jnp.array([1.0]), "Lsun / rad2"),
        sigma=u.Quantity(jnp.array([1.0]), "rad"),
        q=u.Quantity(jnp.array([0.2]), ""),
        PA_twist=u.Quantity(jnp.array([0.0]), "rad"),
    )
    n_x, n_y = 40, 40
    min_x, min_y = -4.0, -4.0
    x_extent, y_extent = 8.0, 8.0
    x_centers = min_x + (np.arange(n_x) + 0.5) * (x_extent / n_x)
    y_centers = min_y + (np.arange(n_y) + 0.5) * (y_extent / n_y)
    half_width = 0.3
    y_strip = np.abs(x_centers)[:, None] < half_width  # bin 1: thin in x, tall in y
    x_strip = np.abs(y_centers)[None, :] < half_width  # bin 2: thin in y, wide in x
    bins = np.where(y_strip, 1, np.where(x_strip & ~y_strip, 2, 0))
    binning = _projected_binning(
        min_x=min_x, min_y=min_y, x_extent=x_extent, y_extent=y_extent, pa=pa, bins=bins
    )

    mass = mge.get_projected_mass(binning).ustrip("Lsun")

    assert mass[aligned_bin_idx] > mass[1 - aligned_bin_idx]


def test_get_projected_mass_conserves_total_flux_for_circular_component():
    mge = LightMGE(
        I=u.Quantity(jnp.array([5.0]), "Lsun / rad2"),
        sigma=u.Quantity(jnp.array([0.01]), "rad"),
        q=u.Quantity(jnp.array([1.0]), ""),
        PA_twist=u.Quantity(jnp.array([0.0]), "rad"),
    )
    bins = np.ones((60, 60), dtype=int)
    binning = _projected_binning(
        min_x=-1.0, min_y=-1.0, x_extent=2.0, y_extent=2.0, pa=0.3, bins=bins
    )

    mass = mge.get_projected_mass(binning)

    expected = 2 * np.pi * 5.0 * 1.0 * 0.01**2
    assert jnp.allclose(mass.ustrip("Lsun"), expected, rtol=1e-6)


def test_get_projected_mass_excludes_unbinned_pixels():
    mge = LightMGE(
        I=u.Quantity(jnp.array([5.0]), "Lsun / rad2"),
        sigma=u.Quantity(jnp.array([0.01]), "rad"),
        q=u.Quantity(jnp.array([1.0]), ""),
        PA_twist=u.Quantity(jnp.array([0.0]), "rad"),
    )
    bins = np.array([[0, 1], [1, 0]])
    binning = _projected_binning(
        min_x=-0.02, min_y=-0.02, x_extent=0.04, y_extent=0.04, pa=0.0, bins=bins
    )

    mass = mge.get_projected_mass(binning)

    assert mass.shape == (1,)


def test_get_projected_mass_aggregates_multiple_pixels_per_bin():
    mge = LightMGE(
        I=u.Quantity(jnp.array([5.0]), "Lsun / rad2"),
        sigma=u.Quantity(jnp.array([0.01]), "rad"),
        q=u.Quantity(jnp.array([0.7]), ""),
        PA_twist=u.Quantity(jnp.array([0.0]), "rad"),
    )
    single_bin = np.ones((4, 4), dtype=int)
    per_pixel_bins = 1 + np.arange(16).reshape(4, 4)
    binning_kwargs = {
        "min_x": -0.02,
        "min_y": -0.02,
        "x_extent": 0.04,
        "y_extent": 0.04,
        "pa": 0.2,
    }

    combined = mge.get_projected_mass(
        _projected_binning(bins=single_bin, **binning_kwargs)
    )
    separate = mge.get_projected_mass(
        _projected_binning(bins=per_pixel_bins, **binning_kwargs)
    )

    assert jnp.allclose(combined.ustrip("Lsun"), jnp.sum(separate.ustrip("Lsun")))


def test_get_projected_mass_requires_consistent_units():
    mge = LightMGE(
        I=u.Quantity(jnp.array([5.0]), "Lsun / rad2"),
        sigma=u.Quantity(jnp.array([0.01]), "rad"),
        q=u.Quantity(jnp.array([1.0]), ""),
        PA_twist=u.Quantity(jnp.array([0.0]), "rad"),
    )
    binning = _projected_binning(
        min_x=-1.0,
        min_y=-1.0,
        x_extent=2.0,
        y_extent=2.0,
        pa=0.0,
        bins=np.ones((3, 3), dtype=int),
    ).angular_to_physical(u.Quantity(30.5, "Mpc"))

    with pytest.raises(ValueError, match="not convertible"):
        mge.get_projected_mass(binning)


def test_get_projected_mass_invariant_under_matching_physical_conversion():
    mge = LightMGE(
        I=u.Quantity(jnp.array([5.0, 2.0]), "Lsun / rad2"),
        sigma=u.Quantity(jnp.array([0.01, 0.02]), "rad"),
        q=u.Quantity(jnp.array([0.6, 0.9]), ""),
        PA_twist=u.Quantity(jnp.array([0.0, 0.4]), "rad"),
    )
    bins = 1 + np.arange(9).reshape(3, 3)
    binning = _projected_binning(
        min_x=-0.05, min_y=-0.05, x_extent=0.1, y_extent=0.1, pa=0.5, bins=bins
    )
    distance = u.Quantity(30.5, "Mpc")

    angular_mass = mge.get_projected_mass(binning)
    physical_mass = mge.angular_to_physical(distance).get_projected_mass(
        binning.angular_to_physical(distance)
    )

    assert jnp.allclose(
        angular_mass.ustrip("Lsun"), physical_mass.ustrip("Lsun"), rtol=1e-6
    )


def test_get_projected_mass_is_jit_compatible():
    """`num_segments` must come from `binning.n_bins`, not `int(jnp.max(bins))`.

    Under `jax.jit`, `bins` (like every other leaf) is traced, so computing
    `n_bins` from it inside `get_projected_mass` would raise
    `ConcretizationTypeError` -- `ProjectedBinning.n_bins` must already be a
    static Python `int`, precomputed at construction time.
    """
    mge = LightMGE(
        I=u.Quantity(jnp.array([5.0, 2.0]), "Lsun / rad2"),
        sigma=u.Quantity(jnp.array([0.01, 0.02]), "rad"),
        q=u.Quantity(jnp.array([0.7, 0.9]), ""),
        PA_twist=u.Quantity(jnp.array([0.0, 0.3]), "rad"),
    )
    binning = _projected_binning(
        min_x=-0.02,
        min_y=-0.02,
        x_extent=0.04,
        y_extent=0.04,
        pa=0.2,
        bins=np.array([[1, 2], [2, 1]]),
    )

    jitted_mass = jax.jit(lambda mge, binning: mge.get_projected_mass(binning))(
        mge, binning
    )
    eager_mass = mge.get_projected_mass(binning)

    assert jnp.allclose(jitted_mass.ustrip("Lsun"), eager_mass.ustrip("Lsun"))


def _analytic_total_mass(
    I: np.ndarray, sigma: np.ndarray, p: np.ndarray, q: np.ndarray  # noqa: E741, N803
) -> np.ndarray:
    return np.sum((2 * np.pi) ** 1.5 * sigma**3 * p * q * I)


_SPHERICAL_QUAD_ORDER = 10


def test_spherical_grid_init_shapes_and_bounds():
    grid = SphericalGrid(
        n_r=5,
        n_theta=3,
        n_phi=4,
        r_min=u.Quantity(0.1, "kpc"),
        r_max=u.Quantity(50.0, "kpc"),
        quad_order=_SPHERICAL_QUAD_ORDER,
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
        quad_order=_SPHERICAL_QUAD_ORDER,
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
            quad_order=_SPHERICAL_QUAD_ORDER,
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
        quad_order=_SPHERICAL_QUAD_ORDER,
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
    # n_r/n_theta/n_phi kept at the minimum allowed (3): the radial integral is
    # exact and the angular integral uses fixed-order Gauss-Legendre quadrature
    # per cell, so accuracy here is essentially independent of grid resolution
    # -- finer grids only add JAX dispatch overhead, not tighter agreement.
    grid = SphericalGrid(
        n_r=3,
        n_theta=3,
        n_phi=3,
        r_min=u.Quantity(0.1, "kpc"),
        r_max=u.Quantity(50.0, "kpc"),
        quad_order=_SPHERICAL_QUAD_ORDER,
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
    # See the spherical-component test above for why n_r/n_theta/n_phi=3 (the
    # minimum) is already as accurate as a much finer grid.
    grid = SphericalGrid(
        n_r=3,
        n_theta=3,
        n_phi=3,
        r_min=u.Quantity(0.05, "kpc"),
        r_max=u.Quantity(100.0, "kpc"),
        quad_order=_SPHERICAL_QUAD_ORDER,
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
        n_r=3,
        n_theta=3,
        n_phi=3,
        r_min=u.Quantity(0.1, "kpc"),
        r_max=u.Quantity(50.0, "kpc"),
        quad_order=_SPHERICAL_QUAD_ORDER,
    )

    masses = mge.spherical_mass_grid(grid)
    total = 8 * jnp.sum(masses.ustrip("Msun"))

    # sigma and I here, in mutually consistent pc-based units.
    expected = _analytic_total_mass(
        np.array([3.0]), np.array([2000.0]), np.array([1.0]), np.array([1.0])
    )
    assert jnp.allclose(total, expected, rtol=1e-5)
