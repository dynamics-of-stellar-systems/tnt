"""All registered MGE coordinates use the same traced proposal builder."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unxt import Quantity

from tnt.mge import LightMGE
from tnt.potential import Potential

SCHEMES = [
    ("Oblate", "q_min", ("q_min",), [0.6]),
    ("Triaxial", "pqu", ("p", "q", "u"), [0.85, 0.6, 0.93]),
    (
        "Triaxial",
        "T_maj_min",
        ("T", "T_maj", "T_min"),
        [0.43359375, 0.4868468468468463, 0.3850103172413796],
    ),
]


@pytest.fixture(autouse=True)
def _release_cache():
    yield
    jax.clear_caches()


def _setup(scheme, mass):
    geometry, name, coordinates, values = scheme
    mge = LightMGE(
        I=Quantity(jnp.array([20.0]), "Lsun/pc2"),
        sigma=Quantity(jnp.array([1.5]), "kpc"),
        q=Quantity(jnp.array([0.76]), ""),
        PA_twist=Quantity(jnp.array([0.0]), "rad"),
        major_axis_pa=Quantity(0.0, "rad"),
    )
    if mass:
        mge = mge.to_mass(Quantity(1.0, "Msun/Lsun"))
    kind = f"{geometry}{'Mass' if mass else 'Light'}MGEPotential"
    resolved = Potential.resolve(
        {"stars": {"type": kind, "parameterization": name, "mge": "m"}},
        {"m": mge},
    )

    def build(numbers):
        params = {
            key: Quantity(value, "")
            for key, value in zip(coordinates, numbers[1:], strict=True)
        }
        params["mge_mass_scale" if mass else "ml"] = Quantity(
            numbers[0], "" if mass else "Msun/Lsun"
        )
        return Potential.build_with_validity(resolved, {"stars": params}, {})

    return build, jnp.array([2.0, *values])


@pytest.mark.parametrize("scheme", SCHEMES, ids=[s[1] for s in SCHEMES])
@pytest.mark.parametrize("mass", [False, True], ids=["light", "mass"])
@pytest.mark.parametrize("x64", [False, True])
def test_all_coordinates_build_and_batch_with_rejected_proposals(scheme, mass, x64):
    with jax.enable_x64(x64):
        build, values = _setup(scheme, mass)
        eager, eager_valid = build(values)
        compiled, valid = jax.jit(build)(values)
        assert bool(eager_valid) and bool(valid)
        model = compiled.components["stars"].deprojected
        np.testing.assert_allclose(model.q.ustrip(""), [0.6], rtol=3e-4)
        np.testing.assert_allclose(
            model.p.ustrip(""), [1 if scheme[1] == "q_min" else 0.85], rtol=3e-4
        )
        expected_mass = 2 * np.pi * 20 * 1500**2 * 0.76 * 2
        np.testing.assert_allclose(
            model.component_masses.ustrip("Msun"), [expected_mass], rtol=3e-4
        )
        np.testing.assert_allclose(
            model.I.ustrip("Msun/kpc3"),
            eager.components["stars"].deprojected.I.ustrip("Msun/kpc3"),
            rtol=3e-4,
        )
        # A raw-domain failure, an MGE-dependent failure, and a nonfinite input.
        bad_geometry = values.at[1].set(0.9 if scheme[1] == "q_min" else 1.0)
        if scheme[1] == "pqu":
            bad_geometry = values.at[2].set(values[1])  # prolate q == p
        proposals = jnp.stack(
            [
                values,
                values.at[0].set(-1),
                bad_geometry,
                values.at[1].set(jnp.nan),
                values.at[0].set(3),
            ]
        )
        potentials, flags = jax.jit(jax.vmap(build))(proposals)
        np.testing.assert_array_equal(flags, [True, False, False, False, True])
        density = potentials.components["stars"].deprojected.I.ustrip("Msun/kpc3")
        np.testing.assert_array_equal(density[1:4], np.zeros((3, 1)))
        np.testing.assert_allclose(density[4], density[0] * 1.5, rtol=3e-4)


@pytest.mark.parametrize("scheme", SCHEMES, ids=[s[1] for s in SCHEMES])
def test_coordinate_gradients_and_invalid_scalar_guard(scheme):
    # One representative of each conversion; light/mass share its shape math.
    with jax.enable_x64(True):
        build, values = _setup(scheme, mass=False)

        def evaluate(numbers):
            potential, valid = build(numbers)
            model = potential.components["stars"].deprojected
            value = jax.lax.cond(
                valid,
                lambda: (
                    jnp.sum(model.I.ustrip("Msun/kpc3")) / 1e7
                    + jnp.sum(model.sigma.ustrip("kpc"))
                    + jnp.sum(model.q.ustrip(""))
                ),
                lambda: jnp.asarray(0.0),
            )
            return value, valid

        compiled = jax.jit(jax.value_and_grad(evaluate, has_aux=True))
        (_, valid), derivative = compiled(values)
        assert bool(valid)
        assert bool(jnp.all(jnp.isfinite(derivative)))
        direct = jax.grad(lambda x: evaluate(x)[0])(values)
        np.testing.assert_allclose(direct, derivative, rtol=2e-5, atol=1e-8)
        for index in range(len(values)):
            step = jnp.zeros_like(values).at[index].set(1e-5)
            expected = (evaluate(values + step)[0] - evaluate(values - step)[0]) / 2e-5
            assert float(derivative[index]) == pytest.approx(float(expected), rel=2e-5)
        (value, valid), derivative = compiled(values.at[1].set(jnp.nan))
        assert not bool(valid)
        assert float(value) == 0
        np.testing.assert_array_equal(derivative, np.zeros(len(values)))
