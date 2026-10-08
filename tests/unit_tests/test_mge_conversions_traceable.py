"""Shape conversion references, rejection guards and precision boundaries."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unxt import Quantity

from tnt.mge import LightMGE, MGEDeprojectionError


@pytest.fixture(autouse=True)
def _release_compilation_cache():
    jax.clear_caches()
    yield
    jax.clear_caches()


def _mge(q_obs=0.76, twist=0.0):
    return LightMGE(
        I=Quantity(jnp.array([20.0]), "Lsun/pc2"),
        sigma=Quantity(jnp.array([1.5]), "kpc"),
        q=Quantity(jnp.array([q_obs]), ""),
        PA_twist=Quantity(jnp.array([twist]), "rad"),
        major_axis_pa=Quantity(0.0, "deg"),
    )


@pytest.mark.parametrize("x64", [False, True])
def test_q_min_angles_and_derivatives_match_analytic_reference(x64):
    with jax.enable_x64(x64):
        mge = _mge()

        def convert(value):
            angle, valid = mge.inclination_from_q_min_with_validity(value)
            return angle.ustrip("rad"), valid

        value = jnp.asarray(0.6)
        (angle, valid), derivative = jax.jit(jax.value_and_grad(convert, has_aux=True))(
            value
        )
        observed = float(mge.q.ustrip("")[0])
        intrinsic = float(value)
        cosine2 = (observed**2 - intrinsic**2) / (1 - intrinsic**2)
        reference = np.arccos(np.sqrt(cosine2))
        reference_derivative = (
            intrinsic
            * (1 - observed**2)
            / ((1 - intrinsic**2) ** 2 * np.sqrt(cosine2 * (1 - cosine2)))
        )
        assert bool(valid)
        assert float(angle) == pytest.approx(reference, rel=2e-6)
        assert float(derivative) == pytest.approx(reference_derivative, rel=2e-5)
        assert float(mge.inclination_from_q_min(value).ustrip("rad")) == pytest.approx(
            float(angle), rel=2e-6
        )

        for bad in (0.0, -0.1, 0.9, np.nan, np.inf):
            (result, accepted), gradient = jax.jit(
                jax.value_and_grad(convert, has_aux=True)
            )(jnp.asarray(bad))
            assert not bool(accepted)
            assert float(result) == 0 and float(gradient) == 0
            with pytest.raises(MGEDeprojectionError):
                mge.inclination_from_q_min(bad)


@pytest.mark.parametrize("x64", [False, True])
def test_q_min_precision_and_circular_anchor_rejections_agree(x64):
    with jax.enable_x64(x64):
        for observed, intrinsic in ((0.76, 0.001), (0.999999, 0.6), (1.0, 0.6)):
            mge = _mge(observed)
            _, valid = jax.jit(
                lambda x, mge=mge: mge.inclination_from_q_min_with_validity(x)
            )(jnp.asarray(intrinsic))
            expected = x64 and observed < 1
            assert bool(valid) == expected
            if expected:
                mge.inclination_from_q_min(intrinsic)
            else:
                with pytest.raises(MGEDeprojectionError):
                    mge.inclination_from_q_min(intrinsic)


def test_shape_conversion_structure_errors_remain_errors():
    mge = _mge()
    for value in (jnp.array([0.6]), 0.6 + 0.1j, True):
        with pytest.raises(TypeError, match="real scalars"):
            jax.jit(lambda x: mge.inclination_from_q_min_with_validity(x))(value)
    angular = eqx.tree_at(lambda x: x.sigma, mge, Quantity(jnp.array([1.0]), "arcsec"))
    with pytest.raises(ValueError, match="length"):
        jax.jit(lambda x: angular.inclination_from_q_min_with_validity(x))(
            jnp.asarray(0.6)
        )


def _projected_covariance(angles, shape):
    """Independent sky projection of diag(1, p^2, q^2)/u^2."""
    theta, phi, _psi = angles
    p, q, u = shape
    sky = np.array(
        [
            [np.sin(phi), -np.cos(phi), 0],
            [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)],
        ]
    )
    intrinsic = np.diag([1, p**2, q**2]) / u**2
    return sky @ intrinsic @ sky.T


@pytest.mark.parametrize("x64", [False, True])
def test_pqu_and_tmajmin_angles_project_the_requested_covariance(x64):
    with jax.enable_x64(x64):
        mge = _mge(twist=0.3)
        shape = jnp.array([0.85, 0.6, 0.93])
        p, q, u = np.asarray(shape, dtype=float)
        observed = float(mge.q.ustrip("")[0])
        coordinates = jnp.array(
            [
                (1 - p**2) / (1 - q**2),
                (1 - u**2) / (1 - p**2),
                (u**2 * observed**2 - q**2) / (p**2 - q**2),
            ]
        )
        reference = None
        for convert, eager, values in (
            (
                mge.triaxial_viewing_angles_with_validity,
                mge.triaxial_viewing_angles,
                shape,
            ),
            (
                mge.viewing_angles_from_T_Tmaj_Tmin_with_validity,
                mge.viewing_angles_from_T_Tmaj_Tmin,
                coordinates,
            ),
        ):

            def evaluate(values, convert=convert):
                angles, valid = convert(*values)
                return jnp.stack([value.ustrip("rad") for value in angles]), valid

            angles, valid = jax.jit(evaluate)(values)
            assert bool(valid) and valid.shape == ()
            assert np.isfinite(
                np.asarray(jax.jit(jax.jacrev(lambda x: evaluate(x)[0]))(values))
            ).all()
            eager_angles = eager(*values)
            np.testing.assert_allclose(
                angles, [float(x.ustrip("rad")) for x in eager_angles], rtol=2e-5
            )
            local = np.asarray(angles, dtype=float).copy()
            local[2] += float(mge.PA_twist.ustrip("rad")[0])
            covariance = _projected_covariance(local, np.asarray(shape, dtype=float))
            cs, sn = np.cos(local[2]), np.sin(local[2])
            expected = np.array(
                [
                    [sn**2 + observed**2 * cs**2, (observed**2 - 1) * sn * cs],
                    [(observed**2 - 1) * sn * cs, cs**2 + observed**2 * sn**2],
                ]
            )
            np.testing.assert_allclose(covariance, expected, rtol=2e-5, atol=2e-6)
            if reference is None:
                reference = np.asarray(angles)
            else:
                np.testing.assert_allclose(angles, reference, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("x64", [False, True])
def test_triaxial_invalid_shapes_have_zero_finite_gradients(x64):
    with jax.enable_x64(x64):
        mge = _mge()
        for convert, eager, proposals in (
            (
                mge.triaxial_viewing_angles_with_validity,
                mge.triaxial_viewing_angles,
                [
                    [0.85, 0.85, 0.93],
                    [0.85, 0.6, 0.65],
                    [0.99999999, 0.6, 1.0],
                    [np.nan, 0.6, 0.93],
                    [0.85, 0.6, np.inf],
                ],
            ),
            (
                mge.viewing_angles_from_T_Tmaj_Tmin_with_validity,
                mge.viewing_angles_from_T_Tmaj_Tmin,
                [
                    [0, 0.5, 1],
                    [1, 0.5, 0.5],
                    [1e-6, 0.1, 0.2],
                    [-0.1, 0.5, 0.5],
                    [0.5, np.nan, 0.5],
                    [0.5, 0.5, np.inf],
                ],
            ),
        ):

            def evaluate(values, convert=convert):
                angles, valid = convert(*values)
                return sum(x.ustrip("rad") for x in angles), valid

            compiled = jax.jit(jax.value_and_grad(evaluate, has_aux=True))
            for proposal in proposals:
                values = jnp.array(proposal, dtype=float)
                (result, valid), gradient = compiled(values)
                assert not bool(valid)
                assert float(result) == 0
                np.testing.assert_array_equal(gradient, np.zeros(3))
                with pytest.raises(MGEDeprojectionError):
                    eager(*values)


@pytest.mark.parametrize("x64", [False, True])
def test_tmajmin_precision_boundary_agrees_eagerly_and_traced(x64):
    with jax.enable_x64(x64):
        mge = _mge()
        for values in ((0.1, 0.02, 0.2), (0.1, 0.03, 0.2), (0.05, 0.08, 0.2)):

            def convert(values):
                return mge.viewing_angles_from_T_Tmaj_Tmin_with_validity(*values)

            _, valid = jax.jit(convert)(jnp.array(values))
            assert bool(valid) == x64
            if x64:
                mge.viewing_angles_from_T_Tmaj_Tmin(*values)
            else:
                with pytest.raises(MGEDeprojectionError):
                    mge.viewing_angles_from_T_Tmaj_Tmin(*values)


@pytest.mark.parametrize("x64", [False, True])
def test_tmajmin_zero_thickness_is_rejected_without_invalid_gradient(x64):
    with jax.enable_x64(x64):
        mge = _mge(q_obs=0.5)

        def convert(values):
            angles, valid = mge.viewing_angles_from_T_Tmaj_Tmin_with_validity(*values)
            return sum(x.ustrip("rad") for x in angles), valid

        (value, valid), gradient = jax.jit(jax.value_and_grad(convert, has_aux=True))(
            jnp.array([0.5, 0.5, 0.375])
        )
        assert not bool(valid) and float(value) == 0
        np.testing.assert_array_equal(gradient, np.zeros(3))


@pytest.mark.parametrize("x64", [False, True])
def test_edge_on_q_min_remains_valid_without_certifying_its_derivative(x64):
    with jax.enable_x64(x64):
        mge = _mge()
        endpoint = mge.q.ustrip("")[0]

        def angle(value):
            converted, valid = mge.inclination_from_q_min_with_validity(value)
            return converted.ustrip("rad"), valid

        (value, valid), gradient = jax.jit(jax.value_and_grad(angle, has_aux=True))(
            endpoint
        )
        assert bool(valid)
        assert float(value) == pytest.approx(np.pi / 2, rel=1e-6)
        assert not np.isfinite(float(gradient))
        assert float(
            mge.inclination_from_q_min(endpoint).ustrip("rad")
        ) == pytest.approx(np.pi / 2, rel=1e-6)


@pytest.mark.parametrize("x64", [False, True])
def test_upper_compression_endpoint_remains_valid_with_a_clipped_gradient(x64):
    with jax.enable_x64(x64):
        mge = _mge()

        def angle_sum(compression):
            angles, valid = mge.triaxial_viewing_angles_with_validity(
                0.85, 0.6, compression
            )
            return sum(value.ustrip("rad") for value in angles), valid

        (value, valid), gradient = jax.jit(jax.value_and_grad(angle_sum, has_aux=True))(
            jnp.asarray(1.0)
        )
        assert bool(valid) and np.isfinite(float(value))
        assert float(gradient) == 0.0
        angles = mge.triaxial_viewing_angles(0.85, 0.6, 1.0)
        _, _, recovered = mge.triaxial_intrinsic_shape(*angles)
        assert recovered == pytest.approx(1.0, abs=3e-3 if not x64 else 1e-6)
