"""Registered MGE inverses preserve configured units inside potential traces."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unxt import Quantity

from tnt.mge import LightMGE, MGEDeprojectionError
from tnt.potential import Potential, raw_potential_parameters
from tnt.potential.registry import get_parameterization

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


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("mass", [False, True], ids=["light", "mass"])
@pytest.mark.parametrize("scheme", SCHEMES, ids=[s[1] for s in SCHEMES])
def test_build_rescale_and_inverse_report_trace_with_unit_scaled_derivatives(
    x64, mass, scheme
):
    with jax.enable_x64(x64):
        geometry, parameterization, names, shape = scheme
        mge = LightMGE(
            I=Quantity(jnp.array([20.0]), "Lsun/pc2"),
            sigma=Quantity(jnp.array([1500.0]), "pc"),
            q=Quantity(jnp.array([76.0]), "percent"),
            PA_twist=Quantity(
                jnp.array([0.0 if geometry == "Oblate" else 12.0]), "deg"
            ),
            major_axis_pa=Quantity(0.0, "deg"),
        )
        mass_name, mass_unit = (
            ("mge_mass_scale", "percent") if mass else ("ml", "kg/Lsun")
        )
        if mass:
            mge = mge.to_mass(Quantity(1.0, "Msun/Lsun"))
        # Both dimensionless ratios and normalization use non-default units.
        mass_value = (
            200.0 if mass else float(Quantity(2.0, "Msun/Lsun").ustrip(mass_unit))
        )
        values = jnp.array([mass_value, *(100 * np.asarray(shape))])
        units = {mass_name: mass_unit, **dict.fromkeys(names, "percent")}
        settings = {
            "stars": {
                "type": f"{geometry}{'Mass' if mass else 'Light'}MGEPotential",
                "parameterization": parameterization,
                "mge": "m",
                "parameters": {name: {"unit": unit} for name, unit in units.items()},
            }
        }
        resolved = Potential.resolve(settings, {"m": mge})

        def recover(numbers):
            parameters = {
                name: Quantity(value, units[name])
                for name, value in zip((mass_name, *names), numbers, strict=True)
            }
            potential, valid = Potential.build_with_validity(
                resolved, {"stars": parameters}, {}
            )
            raw = raw_potential_parameters(settings, potential.rescale(3.0), {})[
                "stars"
            ]
            reported = jnp.stack(
                [raw[name].ustrip(units[name]) for name in (mass_name, *names)]
            )
            return reported, valid

        result, valid = jax.jit(recover)(values)
        assert bool(valid)
        expected = np.asarray(values).copy()
        expected[0] *= 3
        np.testing.assert_allclose(result, expected, rtol=3e-5, atol=2e-5)
        for differentiate in (jax.jacfwd, jax.jacrev):
            derivative = jax.jit(differentiate(lambda v: recover(v)[0]))(values)
            np.testing.assert_allclose(
                derivative, np.diag([3, *np.ones(len(names))]), rtol=3e-4, atol=3e-5
            )
        np.testing.assert_allclose(recover(values)[0], result, rtol=3e-5)
        proposals = jnp.stack([values, values.at[0].multiply(2)])
        batched, flags = jax.jit(jax.vmap(recover))(proposals)
        np.testing.assert_array_equal(flags, [True, True])
        np.testing.assert_allclose(batched[0], expected, rtol=3e-5, atol=2e-5)
        expected[0] *= 2
        np.testing.assert_allclose(batched[1], expected, rtol=3e-5, atol=2e-5)


@pytest.mark.parametrize("scheme", SCHEMES, ids=[s[1] for s in SCHEMES])
def test_invalid_registered_inverse_raises_eagerly_and_is_nonfinite_in_trace(scheme):
    geometry, parameterization, names, _ = scheme
    mge = LightMGE(
        I=Quantity(jnp.array([20.0]), "Lsun/pc2"),
        sigma=Quantity(jnp.array([1.5]), "kpc"),
        q=Quantity(jnp.array([0.76]), ""),
        PA_twist=Quantity(jnp.zeros(1), "rad"),
        major_axis_pa=Quantity(0.0, "deg"),
    )
    spec = get_parameterization(f"{geometry}LightMGEPotential", parameterization)

    def inverse(value):
        native = {"ml": Quantity(2.0, "Msun/Lsun")}
        if geometry == "Oblate":
            native["inclination"] = Quantity(value, "rad")
        else:
            native.update(
                {
                    "theta": Quantity(value, "rad"),
                    "phi": Quantity(0.96, "rad"),
                    "psi": Quantity(0.1, "rad"),
                }
            )
        return spec.invert(native, dict.fromkeys(names, "percent"), {}, mge)

    with pytest.raises(MGEDeprojectionError):
        inverse(jnp.asarray(0.0))
    reported = jax.jit(inverse)(jnp.asarray(0.0))
    for name in names:
        assert not np.isfinite(reported[name].ustrip("percent"))
    assert float(reported["ml"].ustrip("Msun/Lsun")) == 2.0
