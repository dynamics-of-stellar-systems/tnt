"""Inverse shape values, gradients, rejection guards and coordinate singularities."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unxt import Quantity

from tnt.mge import LightMGE, MGEDeprojectionError


@pytest.fixture(autouse=True)
def _release_cache():
    yield
    jax.clear_caches()


def _mge():
    # The anchor is deliberately not row zero.
    return LightMGE(
        I=Quantity(jnp.array([20.0, 10.0]), "Lsun/pc2"),
        sigma=Quantity(jnp.array([1.5, 2.0]), "kpc"),
        q=Quantity(jnp.array([0.9, 0.76]), ""),
        PA_twist=Quantity(jnp.zeros(2), "rad"),
        major_axis_pa=Quantity(0.0, "deg"),
    )


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("unit", ["rad", "deg"])
def test_oblate_inverse_matches_analytic_value_and_derivative(x64, unit):
    with jax.enable_x64(x64):
        mge = _mge()
        angle = float(Quantity(1.1, "rad").ustrip(unit))

        def inverse(value):
            return mge.q_min_from_inclination_with_validity(Quantity(value, unit))

        (result, valid), gradient = jax.jit(jax.value_and_grad(inverse, has_aux=True))(
            jnp.asarray(angle)
        )
        observed = float(mge.q.ustrip("")[1])
        radians = float(Quantity(angle, unit).ustrip("rad"))
        q = np.sqrt((observed**2 - np.cos(radians) ** 2) / np.sin(radians) ** 2)
        derivative = (1 - observed**2) * np.cos(radians) / (q * np.sin(radians) ** 3)
        derivative *= float(Quantity(1.0, unit).ustrip("rad"))
        assert bool(valid) and valid.shape == ()
        assert float(result) == pytest.approx(q, rel=2e-6)
        assert float(gradient) == pytest.approx(derivative, rel=2e-5)
        assert mge.q_min_from_inclination(Quantity(angle, unit)) == pytest.approx(
            q, rel=2e-6
        )


@pytest.mark.parametrize("x64", [False, True])
def test_oblate_inverse_rejections_batch_with_finite_zero_gradients(x64):
    with jax.enable_x64(x64):
        mge = _mge()

        def inverse(angle):
            return mge.q_min_from_inclination_with_validity(Quantity(angle, "deg"))

        angles = jnp.array([70.0, 0.0, 20.0, -70.0, 91.0, jnp.nan, jnp.inf, 90.0])
        (values, valid), gradients = jax.jit(
            jax.vmap(jax.value_and_grad(inverse, has_aux=True))
        )(angles)
        np.testing.assert_array_equal(
            valid, [True, False, False, False, False, False, False, True]
        )
        np.testing.assert_array_equal(values[1:-1], np.zeros(6))
        np.testing.assert_array_equal(gradients[1:-1], np.zeros(6))
        assert np.isfinite(gradients).all()
        assert float(values[-1]) == pytest.approx(0.76, rel=2e-6)
        for angle in angles[1:-1]:
            with pytest.raises(MGEDeprojectionError, match="oblate geometry"):
                mge.q_min_from_inclination(Quantity(angle, "deg"))


def test_oblate_inverse_uses_geometry_independently_of_normalization():
    mge = eqx.tree_at(lambda m: m.I, _mge(), Quantity(jnp.full(2, 1e305), "Lsun/pc2"))
    angle = Quantity(70.0, "deg")
    result, valid = jax.jit(lambda a: mge.q_min_from_inclination_with_validity(a))(
        angle
    )
    assert bool(valid)
    assert float(result) == pytest.approx(_mge().q_min_from_inclination(angle))
    twisted = eqx.tree_at(
        lambda m: m.PA_twist, mge, Quantity(jnp.array([0.1, 0]), "rad")
    )
    _, valid = twisted.q_min_from_inclination_with_validity(angle)
    assert not bool(valid)


@pytest.mark.parametrize("angle", [1.0, Quantity([1.0], "rad"), Quantity(1.0, "kpc")])
def test_oblate_inverse_static_errors_raise(angle):
    with pytest.raises(ValueError):
        jax.jit(lambda a: _mge().q_min_from_inclination_with_validity(a))(angle)


def _shape_from_covariance(angles, observed, twist, scheme):
    """Independent inversion of projected covariance, in NumPy float64."""
    theta, phi, psi = np.asarray(angles, dtype=float)
    psi += twist
    x = np.array([np.sin(phi), -np.cos(phi), 0])
    y = np.array(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)]
    )
    matrix = np.stack([x * x, y * y, x * y])
    cs, sn = np.cos(psi), np.sin(psi)
    sky = np.array(
        [
            sn**2 + observed**2 * cs**2,
            cs**2 + observed**2 * sn**2,
            (observed**2 - 1) * sn * cs,
        ]
    )
    variances = np.linalg.solve(matrix, sky)
    p, q = np.sqrt(variances[1:] / variances[0])
    u = 1 / np.sqrt(variances[0])
    if scheme == "pqu":
        return np.array([p, q, u])
    return np.array(
        [
            (1 - p**2) / (1 - q**2),
            (1 - u**2) / (1 - p**2),
            (u**2 * observed**2 - q**2) / (p**2 - q**2),
        ]
    )


def _triaxial_inverse(mge, scheme, values, unit="rad"):
    angles = tuple(Quantity(value, unit) for value in values)
    convert = (
        mge.triaxial_intrinsic_shape_with_validity
        if scheme == "pqu"
        else mge.T_Tmaj_Tmin_from_viewing_angles_with_validity
    )
    result, valid = convert(*angles)
    return jnp.stack(result), valid


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("scheme", ["pqu", "T_maj_min"])
@pytest.mark.parametrize("unit", ["rad", "deg"])
def test_triaxial_inverse_matches_covariance_and_finite_difference(x64, scheme, unit):
    with jax.enable_x64(x64):
        mge = eqx.tree_at(
            lambda m: m.PA_twist, _mge(), Quantity(jnp.array([0.0, 0.3]), "rad")
        )
        angles = mge.triaxial_viewing_angles(0.85, 0.6, 0.93)
        values = jnp.array([angle.ustrip(unit) for angle in angles])
        inverse = lambda v: _triaxial_inverse(mge, scheme, v, unit)
        result, valid = jax.jit(inverse)(values)
        assert bool(valid) and valid.shape == ()
        radians_per_unit = np.pi / 180 if unit == "deg" else 1.0
        observed = float(mge.q.ustrip("")[1])
        twist = float(mge.PA_twist.ustrip("rad")[1])
        reference = lambda v: _shape_from_covariance(
            np.asarray(v, dtype=float) * radians_per_unit, observed, twist, scheme
        )
        np.testing.assert_allclose(result, reference(values), rtol=3e-5, atol=2e-6)
        step = 1e-5 / radians_per_unit
        expected = np.column_stack(
            [
                (
                    reference(np.asarray(values) + np.eye(3)[i] * step)
                    - reference(np.asarray(values) - np.eye(3)[i] * step)
                )
                / (2 * step)
                for i in range(3)
            ]
        )
        for differentiate in (jax.jacfwd, jax.jacrev):
            gradient = jax.jit(differentiate(lambda v: inverse(v)[0]))(values)
            np.testing.assert_allclose(gradient, expected, rtol=3e-4, atol=2e-5)
        eager = (
            mge.triaxial_intrinsic_shape
            if scheme == "pqu"
            else mge.T_Tmaj_Tmin_from_viewing_angles
        )
        np.testing.assert_allclose(
            eager(*(Quantity(v, unit) for v in values)), result, rtol=3e-5
        )


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("scheme", ["pqu", "T_maj_min"])
def test_triaxial_inverse_rejections_batch_with_zero_gradients(x64, scheme):
    with jax.enable_x64(x64):
        mge = _mge()
        angles = jnp.array(
            [a.ustrip("rad") for a in mge.triaxial_viewing_angles(0.85, 0.6, 0.93)]
        )
        proposals = jnp.stack(
            [
                angles,
                angles.at[0].set(0),
                angles.at[1].set(0),
                angles.at[0].set(jnp.nan),
                angles.at[1].set(jnp.inf),
                jnp.array([0.3, 0.96, 0.1]),
                angles,
            ]
        )
        inverse = lambda v: _triaxial_inverse(mge, scheme, v)
        result, valid = jax.jit(jax.vmap(inverse))(proposals)
        np.testing.assert_array_equal(
            valid, [True, False, False, False, False, False, True]
        )
        np.testing.assert_array_equal(result[1:-1], np.zeros((5, 3)))
        for differentiate in (jax.jacfwd, jax.jacrev):
            gradients = jax.jit(jax.vmap(differentiate(lambda v: inverse(v)[0])))(
                proposals
            )
            np.testing.assert_array_equal(gradients[1:-1], np.zeros((5, 3, 3)))
            assert np.isfinite(gradients).all()
        eager = (
            mge.triaxial_intrinsic_shape
            if scheme == "pqu"
            else mge.T_Tmaj_Tmin_from_viewing_angles
        )
        for proposal in proposals[1:-1]:
            with pytest.raises(MGEDeprojectionError, match="anchor geometry"):
                eager(*(Quantity(value, "rad") for value in proposal))


@pytest.mark.parametrize("x64", [False, True])
def test_circular_anchor_has_shape_but_singular_T_coordinates(x64):
    with jax.enable_x64(x64):
        mge = eqx.tree_at(lambda m: m.q, _mge(), Quantity(jnp.ones(2), ""))
        angles = jnp.array([1.0, 1.0, 0.0])
        shape, valid = _triaxial_inverse(mge, "pqu", angles)
        assert bool(valid)
        np.testing.assert_array_equal(shape, [1, 1, 1])
        inverse = lambda v: _triaxial_inverse(mge, "T_maj_min", v)
        result, valid = jax.jit(inverse)(angles)
        assert not bool(valid)
        np.testing.assert_array_equal(result, np.zeros(3))
        gradient = jax.jit(jax.jacrev(lambda v: inverse(v)[0]))(angles)
        np.testing.assert_array_equal(gradient, np.zeros((3, 3)))
        with pytest.raises(MGEDeprojectionError, match="coordinate singularity"):
            mge.T_Tmaj_Tmin_from_viewing_angles(*(Quantity(v, "rad") for v in angles))


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("scheme", ["q_min", "pqu", "T_maj_min"])
def test_forward_inverse_composition_has_identity_derivatives(x64, scheme):
    with jax.enable_x64(x64):
        mge = _mge()
        values = jnp.array(
            [0.6]
            if scheme == "q_min"
            else [0.85, 0.6, 0.93]
            if scheme == "pqu"
            else [0.43359375, 0.4868468468468463, 0.3850103172413796]
        )

        def roundtrip(values):
            if scheme == "q_min":
                angle, forward_valid = mge.inclination_from_q_min_with_validity(
                    values[0]
                )
                value, inverse_valid = mge.q_min_from_inclination_with_validity(angle)
                return jnp.atleast_1d(value), forward_valid & inverse_valid
            forward = (
                mge.triaxial_viewing_angles_with_validity
                if scheme == "pqu"
                else mge.viewing_angles_from_T_Tmaj_Tmin_with_validity
            )
            angles, forward_valid = forward(*values)
            inverse = (
                mge.triaxial_intrinsic_shape_with_validity
                if scheme == "pqu"
                else mge.T_Tmaj_Tmin_from_viewing_angles_with_validity
            )
            recovered, inverse_valid = inverse(*angles)
            return jnp.stack(recovered), forward_valid & inverse_valid

        recovered, valid = jax.jit(roundtrip)(values)
        assert bool(valid)
        np.testing.assert_allclose(recovered, values, rtol=3e-5, atol=2e-6)
        for differentiate in (jax.jacfwd, jax.jacrev):
            gradient = jax.jit(differentiate(lambda v: roundtrip(v)[0]))(values)
            np.testing.assert_allclose(
                gradient, np.eye(len(values)), rtol=3e-4, atol=3e-5
            )


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("scheme", ["pqu", "T_maj_min"])
def test_valid_inverse_anchor_is_independent_of_other_gaussians_and_normalization(
    x64, scheme
):
    with jax.enable_x64(x64):
        mge = eqx.tree_at(
            lambda m: m.PA_twist, _mge(), Quantity(jnp.array([2.0, 0.3]), "rad")
        )
        angles = mge.triaxial_viewing_angles(0.85, 0.6, 0.93)
        _, whole_mge_valid = mge.deproject_triaxial_with_validity(*angles)
        assert not bool(whole_mge_valid)
        values = jnp.array([a.ustrip("rad") for a in angles])
        inverse = lambda v: _triaxial_inverse(mge, scheme, v)
        result, valid = jax.jit(inverse)(values)
        assert bool(valid)
        assert np.isfinite(jax.jit(jax.jacrev(lambda v: inverse(v)[0]))(values)).all()
        huge = eqx.tree_at(
            lambda m: m.I,
            mge,
            Quantity(jnp.full(2, 1e305 if x64 else 1e37), "Lsun/pc2"),
        )
        large_result, valid = jax.jit(lambda m: _triaxial_inverse(m, scheme, values))(
            huge
        )
        assert bool(valid)
        np.testing.assert_array_equal(large_result, result)


@pytest.mark.parametrize("scheme", ["pqu", "T_maj_min"])
def test_traced_mge_anchor_selection_and_ties_follow_component_order(scheme):
    mge = eqx.tree_at(
        lambda m: m.PA_twist, _mge(), Quantity(jnp.array([0.0, 0.3]), "rad")
    )
    angles = jnp.array(
        [a.ustrip("rad") for a in mge.triaxial_viewing_angles(0.85, 0.6, 0.93)]
    )
    inverse = eqx.filter_jit(lambda m: _triaxial_inverse(m, scheme, angles))
    reference, valid = inverse(mge)
    assert bool(valid)
    reversed_mge = eqx.tree_at(
        lambda m: (m.q, m.PA_twist), mge, (mge.q[::-1], mge.PA_twist[::-1])
    )
    reversed_result, valid = inverse(reversed_mge)
    assert bool(valid)
    np.testing.assert_array_equal(reversed_result, reference)
    tied = eqx.tree_at(
        lambda m: (m.q, m.PA_twist),
        mge,
        (Quantity(jnp.full(2, 0.76), ""), Quantity(jnp.array([0.3, 2.0]), "rad")),
    )
    tied_result, valid = inverse(tied)
    assert bool(valid)
    np.testing.assert_array_equal(tied_result, reference)


@pytest.mark.parametrize("x64", [False, True])
def test_oblate_inverse_precision_boundary_matches_eager_rejection(x64):
    # Generate the inclination independently in float64, then use active precision.
    angle = np.arccos(np.sqrt((0.76**2 - 0.001**2) / (1 - 0.001**2)))
    with jax.enable_x64(x64):
        mge = _mge()
        result, valid = jax.jit(lambda a: mge.q_min_from_inclination_with_validity(a))(
            Quantity(angle, "rad")
        )
        assert bool(valid) == x64
        if x64:
            assert float(result) == pytest.approx(0.001, rel=2e-6)
        else:
            assert float(result) == 0
            with pytest.raises(MGEDeprojectionError):
                mge.q_min_from_inclination(Quantity(angle, "rad"))


@pytest.mark.parametrize("x64", [False, True])
def test_pqu_inverse_reports_compression_endpoint_actually_constructed(x64):
    with jax.enable_x64(x64):
        mge = _mge()
        angles, valid = mge.triaxial_viewing_angles_with_validity(0.85, 0.6, 1.0)
        assert bool(valid)
        shape, valid = jax.jit(
            lambda *a: mge.triaxial_intrinsic_shape_with_validity(*a)
        )(*angles)
        assert bool(valid)
        expected_u = 1 - 4 * np.sqrt(np.finfo(np.float64 if x64 else np.float32).eps)
        np.testing.assert_allclose(shape, [0.85, 0.6, expected_u], rtol=3e-5)
        assert float(shape[2]) < 1.0


@pytest.mark.parametrize(
    "method",
    [
        "q_min_from_inclination_with_validity",
        "triaxial_intrinsic_shape_with_validity",
        "T_Tmaj_Tmin_from_viewing_angles_with_validity",
    ],
)
@pytest.mark.parametrize("value", [True, 1.0 + 0.1j])
def test_inverse_rejects_nonreal_angle_types_as_static_errors(method, value):
    convert = getattr(_mge(), method)
    arguments = (Quantity(value, "rad"),)
    if method != "q_min_from_inclination_with_validity":
        arguments += (Quantity(0.96, "rad"), Quantity(-1.0, "rad"))
    with pytest.raises(TypeError, match="real scalar Quantities"):
        jax.jit(lambda *a: convert(*a))(*arguments)
