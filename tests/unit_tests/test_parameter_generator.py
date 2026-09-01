"""Unit tests for `tnt.parameter_generator`."""

from __future__ import annotations

import pytest

from tnt.all_models import AllModels
from tnt.parameter_generator import (
    PriorSampler,
    SinglePointParameterGenerator,
    build_parameter_generator,
    parameter_generator_required_settings,
    parameter_generator_type_names,
)
from tnt.priors import Prior


def test_parameter_generator_registry_contains_every_explicit_type() -> None:
    assert parameter_generator_type_names() == {
        "GridSearch",
        "PriorSampler",
        "SinglePoint",
    }


def test_parameter_generator_registry_exposes_required_settings() -> None:
    assert parameter_generator_required_settings() == {
        "GridSearch": frozenset({"delta_chi2_threshold"}),
        "PriorSampler": frozenset({"num_warmup", "seed"}),
        "SinglePoint": frozenset(),
    }


def test_build_parameter_generator_dispatches_through_the_registry() -> None:
    generator = build_parameter_generator(
        {
            "generator_type": "SinglePoint",
            "generator_settings": {},
            "stopping_criteria": {"max_new_mods_per_iter": 1},
        },
        {},
    )

    assert isinstance(generator, SinglePointParameterGenerator)


def test_build_parameter_generator_rejects_an_unregistered_type() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Unknown parameter_space_settings.generator_type: 'Unknown'; "
            "expected one of: GridSearch, PriorSampler, SinglePoint"
        ),
    ):
        build_parameter_generator(
            {
                "generator_type": "Unknown",
                "generator_settings": {},
            },
            {},
        )


def test_single_point_generator_proposes_quantities_in_their_declared_unit() -> None:
    # Declared units on purpose don't match any "internal" unit system --
    # SinglePointParameterGenerator never converts, it just parses what the
    # config declared (see tnt.potential's module docstring for why).
    potential_settings = {
        "bh": {
            "type": "PlummerPotential",
            "parameters": {
                "m_tot": {"value": 5.0, "unit": "kg", "fixed": True},
                "r_s": {"value": 1.0, "unit": "pc", "fixed": True},
            },
        },
        "dh": {
            "type": "NFWPotential",
            "parameterization": "concentration_m200",
            "parameters": {
                "c": {"value": 8.0, "fixed": True},
                "M_200": {"value": 1.0e12, "unit": "Msun", "fixed": True},
            },
        },
    }
    generator = SinglePointParameterGenerator(
        potential_settings=potential_settings,
        generator_settings={},
        max_new_mods_per_iter=10,
    )

    (proposed,) = generator.generate_parameters(AllModels())

    assert proposed["bh"]["m_tot"].ustrip("kg") == pytest.approx(5.0)
    assert proposed["bh"]["r_s"].ustrip("pc") == pytest.approx(1.0)
    assert proposed["dh"]["c"].ustrip("") == pytest.approx(8.0)
    assert proposed["dh"]["M_200"].ustrip("Msun") == pytest.approx(1.0e12)


_PRIOR_SAMPLER_POTENTIAL_SETTINGS = {
    "dh": {
        "type": "NFWPotential",
        "parameterization": "concentration_m200",
        "include": True,
        "parameters": {
            "c": {"fixed": True, "value": 8.0},
            "M_200": {
                "unit": "Msun",
                "fixed": False,
                "prior": {"distribution": "LogUniform", "args": [1.0e10, 1.0e14]},
            },
        },
    },
}


def test_prior_sampler_converts_samples_into_parameter_sets() -> None:
    prior = Prior(_PRIOR_SAMPLER_POTENTIAL_SETTINGS, {}, {})
    generator = PriorSampler(
        potential_settings=_PRIOR_SAMPLER_POTENTIAL_SETTINGS,
        generator_settings={"seed": 0, "num_warmup": 10},
        max_new_mods_per_iter=5,
        prior=prior,
        seed=0,
        num_warmup=10,
    )

    proposed = generator.generate_parameters(AllModels())

    assert len(proposed) == 5
    for candidate in proposed:
        assert candidate["dh"]["c"].ustrip("") == pytest.approx(8.0)
        m200 = candidate["dh"]["M_200"].ustrip("Msun")
        assert 1.0e10 <= m200 <= 1.0e14


def test_prior_sampler_caps_at_max_new_mods_per_iter() -> None:
    prior = Prior(_PRIOR_SAMPLER_POTENTIAL_SETTINGS, {}, {})
    generator = PriorSampler(
        potential_settings=_PRIOR_SAMPLER_POTENTIAL_SETTINGS,
        generator_settings={"seed": 0, "num_warmup": 10},
        max_new_mods_per_iter=3,
        prior=prior,
        seed=0,
        num_warmup=10,
    )

    assert len(generator.generate_parameters(AllModels())) == 3


def test_build_parameter_generator_dispatches_prior_sampler() -> None:
    prior = Prior(_PRIOR_SAMPLER_POTENTIAL_SETTINGS, {}, {})
    parameter_space_settings = {
        "generator_type": "PriorSampler",
        "generator_settings": {"seed": 0, "num_warmup": 10},
        "stopping_criteria": {"max_new_mods_per_iter": 4},
    }

    generator = build_parameter_generator(
        parameter_space_settings, _PRIOR_SAMPLER_POTENTIAL_SETTINGS, prior=prior
    )

    assert isinstance(generator, PriorSampler)
    assert len(generator.generate_parameters(AllModels())) == 4


def test_build_parameter_generator_requires_a_prior_for_prior_sampler() -> None:
    parameter_space_settings = {
        "generator_type": "PriorSampler",
        "generator_settings": {"seed": 0, "num_warmup": 10},
        "stopping_criteria": {"max_new_mods_per_iter": 4},
    }

    with pytest.raises(ValueError, match="requires a built Prior"):
        build_parameter_generator(
            parameter_space_settings, _PRIOR_SAMPLER_POTENTIAL_SETTINGS
        )
