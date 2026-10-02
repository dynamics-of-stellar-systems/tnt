"""Complete MGE proposals and potential gradients through shape conversions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.constants import G
from scipy.integrate import quad
from unxt import Quantity, unitsystem

from tnt.mge import LightMGE
from tnt.potential import Potential

PATHS = [
    ("OblateLightMGEPotential", "q_min"),
    ("OblateMassMGEPotential", "q_min"),
    ("TriaxialLightMGEPotential", "pqu"),
    ("TriaxialMassMGEPotential", "pqu"),
    ("TriaxialLightMGEPotential", "T_maj_min"),
    ("TriaxialMassMGEPotential", "T_maj_min"),
]
NAMES = {
    "q_min": ("q_min",),
    "pqu": ("p", "q", "u"),
    "T_maj_min": ("T", "T_maj", "T_min"),
}
SHAPES = {
    "q_min": [0.6],
    "pqu": [0.85, 0.6, 0.93],
    "T_maj_min": [0.43359375, 0.4868468468468463, 0.3850103172413796],
}


@pytest.fixture(autouse=True)
def _release_compilation_cache():
    jax.clear_caches()
    yield
    jax.clear_caches()


def _setup(kind, parameterization, sigma_unit="kpc"):
    source = LightMGE(
        I=Quantity(jnp.array([20.0, 5.0]), "Lsun/pc2"),
        sigma=Quantity(jnp.array([1.5, 3.0]), "kpc").uconvert(sigma_unit),
        q=Quantity(jnp.array([0.9, 0.76]), ""),
        PA_twist=Quantity(
            jnp.zeros(2) if parameterization == "q_min" else jnp.array([0.305, 0.3]),
            "rad",
        ),
        major_axis_pa=Quantity(0.0, "deg"),
    )
    if "Mass" in kind:
        source = source.to_mass(Quantity(2.0, "Msun/Lsun"))
    settings = {
        "stars": {"type": kind, "parameterization": parameterization, "mge": "m"}
    }
    resolved = Potential.resolve(settings, {"m": source})

    def proposal(values):
        name, unit = ("ml", "Msun/Lsun") if "Light" in kind else ("mge_mass_scale", "")
        return {
            "stars": {
                name: Quantity(values[0], unit),
                **{
                    key: Quantity(value, "")
                    for key, value in zip(
                        NAMES[parameterization], values[1:], strict=True
                    )
                },
            }
        }

    values = jnp.array([2.0, *SHAPES[parameterization]])
    return source, resolved, proposal, values


def _independent_potential(values, source, parameterization):
    """NumPy sky covariance inversion and adaptive Gaussian quadrature.

    Neither TNT deprojection nor Galax potential evaluation supplies this
    reference. The potential is in kpc^2/Myr^2 at [0.7, 0.4, 0.2] kpc.
    """
    observed = np.asarray(source.q.ustrip(""), dtype=float)
    sigma = np.asarray(source.sigma.ustrip("kpc"), dtype=float)
    twist = np.asarray(source.PA_twist.ustrip("rad"), dtype=float)
    anchor = np.argmin(observed)
    if parameterization == "q_min":
        cosine2 = (observed[anchor] ** 2 - values[1] ** 2) / (1 - values[1] ** 2)
        axes = np.column_stack(
            [np.ones(2), np.sqrt((observed**2 - cosine2) / (1 - cosine2)), np.ones(2)]
        )
    else:
        if parameterization == "pqu":
            p, q, u = values[1:]
        else:
            T, major, minor = values[1:]
            denominator = 1 - (1 - T) * minor - observed[anchor] ** 2 * T * major
            q2 = 1 - (1 - observed[anchor] ** 2) / denominator
            p2 = 1 - T * (1 - q2)
            p, q, u = np.sqrt([p2, q2, 1 - major * (1 - p2)])
        p2, q2, u2, o2 = p * p, q * q, u * u, observed[anchor] ** 2
        theta = np.arccos(np.sqrt((u2 - q2) * (o2 * u2 - q2) / ((1 - q2) * (p2 - q2))))
        phi = np.arctan(
            np.sqrt(
                (u2 - p2)
                * (p2 - o2 * u2)
                * (1 - q2)
                / ((1 - u2) * (1 - o2 * u2) * (p2 - q2))
            )
        )
        psi = (
            np.pi
            - np.arctan(
                np.sqrt(
                    (1 - o2 * u2)
                    * (p2 - o2 * u2)
                    * (u2 - q2)
                    / ((1 - u2) * (u2 - p2) * (o2 * u2 - q2))
                )
            )
            - twist[anchor]
        )
        sky = np.array(
            [
                [np.sin(phi), -np.cos(phi), 0],
                [
                    np.cos(theta) * np.cos(phi),
                    np.cos(theta) * np.sin(phi),
                    -np.sin(theta),
                ],
            ]
        )
        matrix = np.array([sky[0] ** 2, sky[1] ** 2, sky[0] * sky[1]])
        axes = []
        for ratio, offset in zip(observed, twist, strict=True):
            cs, sn = np.cos(psi + offset), np.sin(psi + offset)
            covariance = [
                sn * sn + ratio * ratio * cs * cs,
                cs * cs + ratio * ratio * sn * sn,
                (ratio * ratio - 1) * sn * cs,
            ]
            variances = np.linalg.solve(matrix, covariance)
            axes.append(
                [
                    np.sqrt(variances[1] / variances[0]),
                    np.sqrt(variances[2] / variances[0]),
                    1 / np.sqrt(variances[0]),
                ]
            )
        axes = np.asarray(axes)
    surface_unit = "Lsun/pc2" if isinstance(source, LightMGE) else "Msun/pc2"
    intensity = np.asarray(source.I.ustrip(surface_unit), dtype=float)
    masses = 2 * np.pi * intensity * (sigma * 1000) ** 2 * observed * values[0]
    gravity = G.to_value("kpc3/(Msun Myr2)")
    result = 0.0
    for mass, width, (p, q, u) in zip(masses, sigma, axes, strict=True):
        intrinsic_width = width / u

        def integrand(t, p=p, q=q, intrinsic_width=intrinsic_width):
            d_y, d_z = 1 - (1 - p * p) * t * t, 1 - (1 - q * q) * t * t
            radius2 = 0.7**2 + 0.4**2 / d_y + 0.2**2 / d_z
            return np.exp(-t * t * radius2 / (2 * intrinsic_width**2)) / np.sqrt(
                d_y * d_z
            )

        result -= (
            gravity
            * mass
            * np.sqrt(2 / np.pi)
            / intrinsic_width
            * quad(integrand, 0, 1, epsabs=1e-12, epsrel=1e-12)[0]
        )
    return result


@pytest.mark.parametrize(("kind", "parameterization"), PATHS)
@pytest.mark.parametrize("x64", [False, True])
def test_parameterized_potential_values_gradients_and_rejections(
    kind, parameterization, x64
):
    with jax.enable_x64(x64):
        source, resolved, proposal, values = _setup(kind, parameterization)
        units = unitsystem("kpc", "Myr", "Msun", "rad")

        def evaluate(values):
            potential, valid = Potential.build_with_validity(
                resolved, proposal(values), {}
            )
            value = jax.lax.cond(
                valid,
                lambda: (
                    potential.to_galax(units)
                    .potential(
                        Quantity(jnp.array([0.7, 0.4, 0.2]), "kpc"),
                        Quantity(0.0, "Myr"),
                    )
                    .ustrip("kpc2/Myr2")
                ),
                lambda: jnp.asarray(-1.0),
            )
            return value, valid

        compiled = jax.jit(jax.value_and_grad(evaluate, has_aux=True))
        (value, valid), gradient = compiled(values)
        assert bool(valid) and valid.shape == ()
        assert np.isfinite(np.asarray(gradient)).all()
        reference_values = np.asarray(values, dtype=float)
        expected = _independent_potential(reference_values, source, parameterization)
        assert float(value) == pytest.approx(expected, rel=3e-5)
        for index in range(len(values)):
            step = np.zeros(len(values))
            step[index] = 1e-5
            expected_derivative = (
                _independent_potential(
                    reference_values + step, source, parameterization
                )
                - _independent_potential(
                    reference_values - step, source, parameterization
                )
            ) / (2e-5)
            assert float(gradient[index]) == pytest.approx(
                expected_derivative, rel=2e-3, abs=1e-8
            )
        eager = Potential.build(resolved, proposal(values), {})
        _, eager_valid = Potential.build_with_validity(resolved, proposal(values), {})
        assert bool(eager_valid)
        projected = (
            2
            * np.pi
            * np.asarray(
                source.I.ustrip(
                    "Lsun/pc2" if isinstance(source, LightMGE) else "Msun/pc2"
                )
            )
            * np.asarray(source.sigma.ustrip("pc")) ** 2
            * np.asarray(source.q.ustrip(""))
            * values[0]
        )
        np.testing.assert_allclose(
            eager.components["stars"].deprojected.component_masses.ustrip("Msun"),
            projected,
            rtol=2e-5,
        )
        invalid = [
            values.at[0].set(value)
            for value in (0, -1, np.nan, np.inf, 1e301 if x64 else 1e31)
        ]
        invalid += [values.at[1].set(np.nan)]
        if parameterization == "q_min":
            invalid += [values.at[1].set(value) for value in (0, 0.76, 0.9)]
        elif parameterization == "pqu":
            invalid += [
                values.at[2].set(0.85),
                values.at[3].set(0.65),
                values.at[3].set(1),
            ]
        else:
            invalid += [values.at[1].set(1), values.at[2].set(-0.1)]
        for rejected in invalid:
            (result, accepted), derivative = compiled(rejected)
            assert not bool(accepted) and float(result) == -1
            np.testing.assert_array_equal(derivative, np.zeros(len(values)))
            with pytest.raises(ValueError):
                Potential.build(resolved, proposal(rejected), {})
        (again, accepted), again_gradient = compiled(values)
        assert bool(accepted) and float(again) == float(value)
        np.testing.assert_array_equal(again_gradient, gradient)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("parameterization", ["q_min", "T_maj_min"])
def test_parameterized_construction_precision_boundaries(parameterization, x64):
    with jax.enable_x64(x64):
        kind = (
            "OblateMassMGEPotential"
            if parameterization == "q_min"
            else "TriaxialMassMGEPotential"
        )
        _, resolved, proposal, values = _setup(kind, parameterization)
        values = (
            values.at[1].set(0.001)
            if parameterization == "q_min"
            else jnp.array([2.0, 0.1, 0.02, 0.2])
        )
        _, valid = jax.jit(
            lambda x: Potential.build_with_validity(resolved, proposal(x), {})
        )(values)
        assert bool(valid) == x64
        if x64:
            Potential.build(resolved, proposal(values), {})
        else:
            with pytest.raises(ValueError):
                Potential.build(resolved, proposal(values), {})


@pytest.mark.parametrize("parameterization", ["pqu", "T_maj_min"])
def test_valid_anchor_does_not_hide_an_invalid_other_gaussian(parameterization):
    source, _, proposal, values = _setup("TriaxialMassMGEPotential", parameterization)
    source = eqx.tree_at(
        lambda x: x.PA_twist, source, Quantity(jnp.array([2.0, 0.3]), "rad")
    )
    resolved = Potential.resolve(
        {
            "stars": {
                "type": "TriaxialMassMGEPotential",
                "parameterization": parameterization,
                "mge": "m",
            }
        },
        {"m": source},
    )
    potential, valid = jax.jit(
        lambda x: Potential.build_with_validity(resolved, proposal(x), {})
    )(values)
    assert not bool(valid)
    for leaf in jax.tree.leaves(potential.components["stars"].deprojected):
        np.testing.assert_array_equal(leaf, np.zeros(leaf.shape))
    with pytest.raises(ValueError):
        Potential.build(resolved, proposal(values), {})


@pytest.mark.parametrize("parameterization", ["q_min", "pqu", "T_maj_min"])
def test_parameterized_static_errors_raise_inside_trace(parameterization):
    kind = (
        "OblateMassMGEPotential"
        if parameterization == "q_min"
        else "TriaxialMassMGEPotential"
    )
    _, resolved, proposal, values = _setup(kind, parameterization)
    key = NAMES[parameterization][0]
    parameters = proposal(values)
    for bad in (
        Quantity(jnp.array([0.6]), ""),
        Quantity(0.6, "kpc"),
        0.6,
        Quantity(0.6 + 0.1j, ""),
    ):

        def build(unused, bad=bad):
            return Potential.build_with_validity(
                resolved, {"stars": {**parameters["stars"], key: bad}}, {}
            )

        with pytest.raises((ValueError, TypeError)):
            jax.jit(build)(jnp.asarray(0.0))
    for mapping in ({}, {**parameters["stars"], "unknown": Quantity(1.0, "")}):
        with pytest.raises(ValueError):
            jax.jit(
                lambda unused, mapping=mapping: Potential.build_with_validity(
                    resolved, {"stars": mapping}, {}
                )
            )(jnp.asarray(0.0))


def test_guarded_converter_output_structure_is_a_setup_contract():
    _, resolved, proposal, values = _setup("OblateMassMGEPotential", "q_min")
    for output in (
        {"wrong": Quantity(1.0, "")},
        {"mge_mass_scale": Quantity(1.0, ""), "inclination": Quantity(1.0, "kpc")},
    ):
        component = resolved["stars"]._replace(
            convert_with_validity=lambda *args, output=output: (
                output,
                jnp.asarray(True),
            )
        )
        with pytest.raises(ValueError):
            jax.jit(
                lambda x, component=component: Potential.build_with_validity(
                    {"stars": component}, proposal(x), {}
                )
            )(values)
    original = resolved["stars"].convert_with_validity
    component = resolved["stars"]._replace(
        convert_with_validity=lambda *args: (original(*args)[0], jnp.array([True]))
    )
    with pytest.raises(TypeError, match="scalar boolean"):
        jax.jit(
            lambda x: Potential.build_with_validity(
                {"stars": component}, proposal(x), {}
            )
        )(values)


@pytest.mark.parametrize("x64", [False, True])
def test_complete_mixed_proposal_combines_flags_and_guards_gradients(x64):
    with jax.enable_x64(x64):
        _, oblate, oblate_proposal, oblate_values = _setup(
            "OblateLightMGEPotential", "q_min"
        )
        _, triaxial, triaxial_proposal, triaxial_values = _setup(
            "TriaxialMassMGEPotential", "T_maj_min"
        )
        halo = Potential.resolve({"halo": {"type": "PlummerPotential"}}, {})
        resolved = {"oblate": oblate["stars"], "triaxial": triaxial["stars"], **halo}

        def proposal(values):
            return {
                "oblate": oblate_proposal(values[:2])["stars"],
                "triaxial": triaxial_proposal(values[2:])["stars"],
                "halo": {"m_tot": Quantity(1e10, "Msun"), "r_s": Quantity(5.0, "kpc")},
            }

        def evaluate(values):
            potential, valid = Potential.build_with_validity(
                resolved, proposal(values), {}
            )
            mass = jax.lax.cond(
                valid,
                lambda: sum(
                    jnp.sum(
                        potential.components[name].deprojected.component_masses.ustrip(
                            "Msun"
                        )
                    )
                    for name in ("oblate", "triaxial")
                ),
                lambda: jnp.asarray(-1.0),
            )
            return mass, valid

        values = jnp.concatenate([oblate_values, triaxial_values])
        compiled = jax.jit(jax.value_and_grad(evaluate, has_aux=True))
        (mass, valid), gradient = compiled(values)
        assert bool(valid) and valid.shape == () and float(mass) > 0
        assert np.isfinite(np.asarray(gradient)).all()
        for rejected in (
            values.at[1].set(0.9),
            values.at[3].set(1),
            values.at[2].set(np.nan),
        ):
            (result, accepted), derivative = compiled(rejected)
            assert not bool(accepted) and float(result) == -1
            np.testing.assert_array_equal(derivative, np.zeros(len(values)))
        bad_halo = proposal(values)
        bad_halo["halo"]["r_s"] = Quantity(0.0, "kpc")
        _, accepted = Potential.build_with_validity(resolved, bad_halo, {})
        assert not bool(accepted)


@pytest.mark.parametrize("parameterization", ["q_min", "pqu", "T_maj_min"])
@pytest.mark.parametrize("x64", [False, True])
def test_frozen_conversion_is_rejected_even_when_its_values_are_correct(
    parameterization, x64
):
    with jax.enable_x64(x64):
        kind = (
            "OblateMassMGEPotential"
            if parameterization == "q_min"
            else "TriaxialMassMGEPotential"
        )
        _, resolved, proposal, values = _setup(kind, parameterization)
        original = resolved["stars"].convert_with_validity

        def frozen(*args):
            native, valid = original(*args)
            native = {
                name: value
                if name == "mge_mass_scale"
                else jax.lax.stop_gradient(value)
                for name, value in native.items()
            }
            return native, valid

        broken = {"stars": resolved["stars"]._replace(convert_with_validity=frozen)}
        _, valid = jax.jit(
            lambda x: Potential.build_with_validity(broken, proposal(x), {})
        )(values)
        assert not bool(valid)
        with pytest.raises(ValueError, match="derivatives"):
            Potential.build(broken, proposal(values), {})


@pytest.mark.parametrize("x64", [False, True])
def test_q_min_conversion_does_not_validate_unscaled_template_mass(x64):
    with jax.enable_x64(x64):
        source, _, proposal, values = _setup("OblateLightMGEPotential", "q_min")
        # The proposal and its derivatives are representable in declared and
        # physical units, although the unscaled luminosity integral overflows.
        intensity, normalization_unit = (
            (1e305, "1e-290 Msun/Lsun") if x64 else (1e37, "1e-20 Msun/Lsun")
        )
        source = eqx.tree_at(
            lambda m: m.I, source, Quantity(jnp.full(2, intensity), "Lsun/pc2")
        )
        resolved = Potential.resolve(
            {
                "stars": {
                    "type": "OblateLightMGEPotential",
                    "parameterization": "q_min",
                    "mge": "m",
                }
            },
            {"m": source},
        )

        def parameters(values):
            result = proposal(values)
            result["stars"]["ml"] = Quantity(values[0], normalization_unit)
            return result

        potential, valid = jax.jit(
            lambda x: Potential.build_with_validity(resolved, parameters(x), {})
        )(values)
        assert bool(valid)
        assert np.isfinite(
            np.asarray(
                potential.components["stars"].deprojected.component_masses.ustrip(
                    "Msun"
                )
            )
        ).all()
        Potential.build(resolved, parameters(values), {})
