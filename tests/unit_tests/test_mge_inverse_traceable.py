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
