"""Native MGE construction inside one trace, at both supported precisions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unxt import Quantity, unitsystem

from tnt.mge import LightMGE, MassMGE, MGEDeprojectionError
from tnt.potential import Potential

TYPES = [
    "OblateLightMGEPotential",
    "OblateMassMGEPotential",
    "TriaxialLightMGEPotential",
    "TriaxialMassMGEPotential",
]


@pytest.fixture(autouse=True)
def _release_compilation_cache():
    # These tests compile distinct whole-proposal gradient graphs in x32/x64.
    # Bound their memory footprint on the 2 GB Linux development environment.
    jax.clear_caches()
    yield
    jax.clear_caches()


def _setup(kind, sigma_unit="kpc"):
    mge = LightMGE(
        I=Quantity(jnp.array([20.0, 5.0]), "Lsun / pc2"),
        sigma=Quantity(jnp.array([1.5, 3.0]), "kpc").uconvert(sigma_unit),
        q=Quantity(jnp.array([0.9, 0.95]), ""),
        PA_twist=Quantity(jnp.zeros(2), "rad"),
        major_axis_pa=Quantity(0.0, "deg"),
    )
    if "Mass" in kind:
        mge = mge.to_mass(Quantity(2.0, "Msun / Lsun"))
    resolved = Potential.resolve({"stars": {"type": kind, "mge": "m"}}, {"m": mge})

    def proposal(values):
        name = "ml" if "Light" in kind else "mge_mass_scale"
        unit = "Msun / Lsun" if "Light" in kind else ""
        params = {name: Quantity(values[0], unit)}
        if "Oblate" in kind:
            params["inclination"] = Quantity(values[1], "rad")
        else:
            params.update(
                {
                    n: Quantity(v, "rad")
                    for n, v in zip(("theta", "phi", "psi"), values[1:], strict=True)
                }
            )
        return {"stars": params}

    values = jnp.array([2.0, 1.0] if "Oblate" in kind else [2.0, 0.3, 0.96, -1.08])
    return resolved, proposal, values


@pytest.mark.parametrize("kind", TYPES)
@pytest.mark.parametrize("x64", [False, True])
def test_equivalent_width_units_preserve_native_mge_values_and_gradients(kind, x64):
    with jax.enable_x64(x64):
        units = unitsystem("kpc", "Myr", "Msun", "rad")
        position = Quantity(jnp.array([0.7, 0.4, 0.2]), "kpc")
        # Independent surface integral, in pc and Msun/pc2. The mass template
        # includes an additional factor of two from _setup's light conversion.
        expected_mass = (
            2
            * np.pi
            * np.sum(
                np.array([20.0, 5.0]) * np.array([1500.0, 3000.0]) ** 2 * [0.9, 0.95]
            )
            * (4 if "Mass" in kind else 2)
        )
        reference = None
        for sigma_unit in ("kpc", "pc", "km"):
            resolved, proposal, values = _setup(kind, sigma_unit)

            def evaluate(values, resolved=resolved, proposal=proposal):
                potential, valid = Potential.build_with_validity(
                    resolved, proposal(values), {}
                )
                mass = potential.components["stars"].deprojected.component_masses
                value = jax.lax.cond(
                    valid,
                    lambda: (
                        potential.to_galax(units)
                        .potential(position, Quantity(0.0, "Myr"))
                        .ustrip("kpc2 / Myr2")
                    ),
                    lambda: jnp.asarray(0.0),
                )
                return value, (valid, jnp.sum(mass.ustrip("Msun")))

            (value, (traced_valid, mass)), gradient = jax.jit(
                jax.value_and_grad(evaluate, has_aux=True)
            )(values)
            eager, eager_valid = Potential.build_with_validity(
                resolved, proposal(values), {}
            )
            assert bool(eager_valid) and bool(traced_valid)
            assert float(mass) == pytest.approx(expected_mass, rel=2e-5)
            assert bool(jnp.all(jnp.isfinite(gradient)))
            assert float(gradient[0]) == pytest.approx(
                float(value / values[0]), rel=2e-5
            )
            if reference is None:
                reference = float(value), np.asarray(gradient)
            else:
                assert float(value) == pytest.approx(reference[0], rel=2e-5)
                np.testing.assert_allclose(gradient, reference[1], rtol=2e-5, atol=1e-8)
            component = eager.components["stars"]
            assert component.mge.sigma.unit == Quantity(1.0, sigma_unit).unit

            def rescaled_mass(scale, component=component):
                mass = component.rescale(scale).deprojected.component_masses
                return jnp.sum(mass.ustrip(mass.unit))

            # A finite construction Jacobian in forward mode used to miss an
            # infinite reverse-mode gradient through the integrated mass.
            assert float(jax.grad(rescaled_mass)(jnp.asarray(1.0))) == pytest.approx(
                expected_mass, rel=2e-5
            )
            rescaled_value, rescaled_gradient = jax.jit(
                jax.value_and_grad(rescaled_mass)
            )(jnp.asarray(1.0))
            assert float(rescaled_value) == pytest.approx(expected_mass, rel=2e-5)
            assert float(rescaled_gradient) == pytest.approx(expected_mass, rel=2e-5)
            jax.clear_caches()  # Bound memory across the equivalent-unit graphs.


@pytest.mark.parametrize("kind", TYPES)
@pytest.mark.parametrize("x64", [False, True])
def test_integer_mge_columns_support_native_construction_and_gradients(kind, x64):
    with jax.enable_x64(x64):
        resolved, proposal, values = _setup(kind)
        source = resolved["stars"].extra_fields["mge"]
        source = eqx.tree_at(
            lambda m: (m.I, m.sigma),
            source,
            (
                Quantity(jnp.array([20, 5]), source.I.unit),
                Quantity(jnp.array([100, 300]), "kpc"),
            ),
        )
        resolved = Potential.resolve(
            {"stars": {"type": kind, "mge": "m"}}, {"m": source}
        )
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
                        Quantity(jnp.array([70.0, 40.0, 20.0]), "kpc"),
                        Quantity(0.0, "Myr"),
                    )
                    .ustrip("kpc2 / Myr2")
                ),
                lambda: jnp.asarray(0.0),
            )
            return value, valid

        (value, valid), gradient = jax.jit(jax.value_and_grad(evaluate, has_aux=True))(
            values
        )
        _, eager_valid = Potential.build_with_validity(resolved, proposal(values), {})
        assert bool(valid) and bool(eager_valid)
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert float(gradient[0]) == pytest.approx(float(value / values[0]), rel=2e-5)
        assert jnp.issubdtype(source.sigma.dtype, jnp.integer)


@pytest.mark.parametrize("kind", TYPES)
@pytest.mark.parametrize("x64", [False, True])
def test_vmap_builds_mixed_valid_and_invalid_native_proposals(kind, x64):
    with jax.enable_x64(x64):
        resolved, proposal, values = _setup(kind)
        proposals = jnp.stack([
            values, values.at[0].set(-1), values.at[1].set(0),
            values.at[0].set(jnp.nan), values.at[0].set(3),
        ])
        potentials, valid = jax.jit(jax.vmap(
            lambda x: Potential.build_with_validity(resolved, proposal(x), {})
        ))(proposals)
        np.testing.assert_array_equal(valid, [True, False, False, False, True])
        density = potentials.components["stars"].deprojected.I.ustrip("Msun/kpc3")
        np.testing.assert_array_equal(density[1:4], np.zeros((3, 2)))
        np.testing.assert_allclose(density[4], density[0] * 1.5, rtol=2e-5)


@pytest.mark.parametrize("x64", [False, True])
def test_compact_mge_in_kilometres_has_finite_compiled_mass_and_gradient(x64):
    with jax.enable_x64(x64):
        source = MassMGE(
            I=Quantity(jnp.array([1e10]), "Msun/kpc2"),
            sigma=Quantity(jnp.array([3e12]), "km"),
            q=Quantity(jnp.array([0.9]), ""),
            PA_twist=Quantity(jnp.zeros(1), "rad"),
            major_axis_pa=Quantity(0.0, "rad"),
        )
        resolved = Potential.resolve(
            {"stars": {"type": "OblateMassMGEPotential", "mge": "m"}}, {"m": source}
        )

        def evaluate(scale):
            potential, valid = Potential.build_with_validity(
                resolved,
                {
                    "stars": {
                        "mge_mass_scale": Quantity(scale, ""),
                        "inclination": Quantity(1.0, "rad"),
                    }
                },
                {},
            )
            mass = jax.lax.cond(
                valid,
                lambda: jnp.sum(
                    potential.components["stars"].deprojected.component_masses.ustrip(
                        "Msun"
                    )
                ),
                lambda: jnp.asarray(0.0),
            )
            return mass, valid

        # About 0.097 pc wide and 535 Msun; use the independently converted
        # width in the analytical 2D Gaussian integral.
        width_kpc = 3e12 / 3.085677581491367e16
        expected = 2 * np.pi * 1e10 * width_kpc**2 * 0.9
        (mass, valid), derivative = jax.jit(jax.value_and_grad(evaluate, has_aux=True))(
            jnp.asarray(1.0)
        )
        assert bool(valid)
        assert float(mass) == pytest.approx(expected, rel=2e-5)
        assert float(derivative) == pytest.approx(expected, rel=2e-5)


@pytest.mark.parametrize("kind", TYPES)
@pytest.mark.parametrize("x64", [False, True])
def test_native_mge_potential_values_gradients_and_rejection(kind, x64):
    with jax.enable_x64(x64):
        resolved, proposal, values = _setup(kind)
        units = unitsystem("kpc", "Myr", "Msun", "rad")
        position = Quantity(jnp.array([0.7, 0.4, 0.2]), "kpc")

        def evaluate(values):
            potential, valid = Potential.build_with_validity(
                resolved, proposal(values), {}
            )
            value = jax.lax.cond(
                valid,
                lambda: (
                    potential.to_galax(units)
                    .potential(position, Quantity(0.0, "Myr"))
                    .ustrip("kpc2 / Myr2")
                ),
                lambda: jnp.asarray(-1.0),
            )
            return value, valid

        compiled = jax.jit(jax.value_and_grad(evaluate, has_aux=True))
        (value, valid), gradient = compiled(values)
        assert valid.shape == () and valid.dtype == jnp.bool_
        assert bool(valid) and bool(jnp.all(jnp.isfinite(gradient)))
        eager = Potential.build(resolved, proposal(values), {})
        model = eager.components["stars"].deprojected
        source = resolved["stars"].extra_fields["mge"]
        surface = (
            source.to_mass(Quantity(values[0], "Msun / Lsun"))
            if "Light" in kind
            else source.rescaled(Quantity(values[0], ""))
        )
        projected_mass = (2 * jnp.pi * surface.I * surface.sigma**2 * surface.q).ustrip(
            "Msun"
        )
        intrinsic_mass = (
            model.I * model.p * model.q * (2 * jnp.pi) ** 1.5 * model.sigma**3
        ).ustrip("Msun")
        assert jnp.allclose(intrinsic_mass, projected_mass, rtol=2e-5)
        reference = (
            eager.to_galax(units)
            .potential(position, Quantity(0.0, "Myr"))
            .ustrip("kpc2 / Myr2")
        )
        assert float(value) == pytest.approx(float(reference), rel=2e-5)
        # Potential is linear in normalization; angle derivatives use independent
        # central finite differences, rather than differentiating the same path.
        assert float(gradient[0]) == pytest.approx(float(value / values[0]), rel=2e-5)
        step = 2e-4 if x64 else 2e-3
        for index in range(1, len(values)):
            delta = jnp.zeros_like(values).at[index].set(step)
            difference = (evaluate(values + delta)[0] - evaluate(values - delta)[0]) / (
                2 * step
            )
            assert float(gradient[index]) == pytest.approx(
                float(difference), rel=0.01, abs=1e-7
            )
        bad = [values.at[0].set(v) for v in [0.0, -1.0, jnp.inf, jnp.nan]]
        bad += [values.at[1].set(0.0)]
        bad += [values.at[1].set(jnp.nan)]
        # Finite positive raw normalization can overflow converted mass or
        # underflow density. Both are proposal failures.
        bad += [
            values.at[0].set(v) for v in ([1e301, 1e-320] if x64 else [1e31, 1e-44])
        ]
        if "Triaxial" in kind:
            bad += [values.at[3].set(0.1)]  # finite, convention-violating solution
        for rejected in bad:
            (result, valid), derivative = compiled(rejected)
            assert not bool(valid)
            assert float(result) == -1.0
            assert bool(jnp.all(jnp.isfinite(derivative)))
            with pytest.raises(ValueError):
                Potential.build(resolved, proposal(rejected), {})


@pytest.mark.parametrize("x64", [False, True])
def test_oblate_native_cancellation_boundary_has_precision_dependent_validity(x64):
    # q_intr=0.001 has severe cancellation at x32, but is well resolved at x64.
    # Compute the requested angle independently with NumPy float64.
    inclination = np.arccos(np.sqrt((0.9**2 - 0.001**2) / (1 - 0.001**2)))
    with jax.enable_x64(x64):
        resolved, proposal, values = _setup("OblateMassMGEPotential")
        values = values.at[1].set(inclination)
        _, valid = jax.jit(
            lambda x: Potential.build_with_validity(resolved, proposal(x), {})
        )(values)
        assert bool(valid) is x64
        if x64:
            model = (
                Potential.build(resolved, proposal(values), {})
                .components["stars"]
                .deprojected
            )
            assert float(model.q.ustrip("")[0]) == pytest.approx(0.001, rel=1e-6)
        else:
            with pytest.raises(MGEDeprojectionError, match="precision"):
                Potential.build(resolved, proposal(values), {})


@pytest.mark.parametrize("kind", TYPES)
def test_native_mge_static_parameter_contract_errors_still_raise(kind):
    resolved, proposal, values = _setup(kind)
    params = proposal(values)
    for invalid in ({}, {**params["stars"], "unknown": Quantity(1.0, "")}):
        with pytest.raises(ValueError, match="parameters"):
            jax.jit(
                lambda invalid=invalid: Potential.build_with_validity(
                    resolved, {"stars": invalid}, {}
                )
            )()
    with pytest.raises(TypeError, match="mapping"):
        Potential.build_with_validity(
            resolved, {"stars": list(params["stars"].items())}, {}
        )
    name = "inclination" if "Oblate" in kind else "theta"
    for invalid in (Quantity(jnp.ones(2), "rad"), Quantity(1.0, "kpc"), 1.0):
        proposed = {"stars": {**params["stars"], name: invalid}}
        with pytest.raises((ValueError, TypeError)):
            jax.jit(
                lambda proposed=proposed: Potential.build_with_validity(
                    resolved, proposed, {}
                )
            )()


def test_native_mge_integer_numeric_parameters_are_supported():
    resolved, proposal, _ = _setup("OblateMassMGEPotential")
    proposed = proposal(jnp.array([2, 1]))
    _, valid = jax.jit(lambda: Potential.build_with_validity(resolved, proposed, {}))()
    assert bool(valid)
    Potential.build(resolved, proposed, {})


@pytest.mark.parametrize("x64", [False, True])
def test_complete_proposal_combines_mge_and_native_component_validity(x64):
    with jax.enable_x64(x64):
        resolved, proposal, values = _setup("OblateLightMGEPotential")
        resolved.update(Potential.resolve({"halo": {"type": "PlummerPotential"}}, {}))

        def build(mass):
            proposed = proposal(values)
            proposed["halo"] = {
                "m_tot": Quantity(mass, "Msun"),
                "r_s": Quantity(1.0, "kpc"),
            }
            return Potential.build_with_validity(resolved, proposed, {})[1]

        assert bool(jax.jit(build)(jnp.asarray(1e8)))
        assert not bool(jax.jit(build)(jnp.asarray(-1.0)))
        with pytest.raises(ValueError, match="proposal components"):
            Potential.build_with_validity(resolved, proposal(values), {})
        complete = {
            **proposal(values),
            "halo": {"m_tot": Quantity(1e8, "Msun"), "r_s": Quantity(1.0, "kpc")},
        }
        with pytest.raises(ValueError, match="proposal components"):
            Potential.build_with_validity(resolved, {**complete, "typo": {}}, {})


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("oblate", [False, True])
def test_exact_circular_rows_are_spherical_without_roundoff_rejection(x64, oblate):
    with jax.enable_x64(x64):
        mge = LightMGE(
            I=Quantity(jnp.array([20.0]), "Lsun / pc2"),
            sigma=Quantity(jnp.array([1.5]), "kpc"),
            q=Quantity(jnp.ones(1), ""),
            PA_twist=Quantity(jnp.zeros(1), "rad"),
            major_axis_pa=Quantity(0.0, "rad"),
        )

        def evaluate(angle):
            if oblate:
                model, valid = mge.deproject_oblate_with_validity(
                    Quantity(angle, "rad")
                )
            else:
                model, valid = mge.deproject_triaxial_with_validity(
                    Quantity(angle, "rad"), Quantity(0.4, "rad"), Quantity(0.3, "rad")
                )
            value = jax.lax.cond(
                valid,
                lambda: jnp.sum(model.I.ustrip(model.I.unit)),
                lambda: jnp.asarray(-1.0),
            )
            return value, (valid, model.p.ustrip(""), model.q.ustrip(""))

        (value, (valid, p, q)), gradient = jax.jit(
            jax.value_and_grad(evaluate, has_aux=True)
        )(jnp.asarray(0.01))
        assert bool(valid)
        assert bool(jnp.all(p == 1)) and bool(jnp.all(q == 1))
        assert float(value) == pytest.approx(20 / (np.sqrt(2 * np.pi) * 1.5), rel=1e-6)
        assert float(gradient) == pytest.approx(0.0, abs=1e-7)
