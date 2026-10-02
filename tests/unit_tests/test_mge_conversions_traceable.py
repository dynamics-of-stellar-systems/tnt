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

        for bad in (0.0, -0.1, observed, 0.9, np.nan, np.inf):
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
