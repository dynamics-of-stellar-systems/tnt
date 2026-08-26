"""Unit tests for `tnt.parameter_generator`."""

from __future__ import annotations

import pytest

from tnt.all_models import AllModels
from tnt.configuration import validation as configuration_validation
from tnt.parameter_generator import (
    _GENERATOR_CLASSES,
    PriorSampler,
    SinglePointParameterGenerator,
    build_parameter_generator,
)
from tnt.priors import Prior


def test_generator_settings_keys_match_the_real_classes() -> None:
    """`tnt.configuration.validation._GENERATOR_SETTINGS_KEYS` is duplicated,
    plain-data information -- it can't import these classes directly (see
    its own comment for why: preparation-phase code shouldn't depend on
    execution-phase modules). This is the regression test that keeps that
    duplicated data in sync with each class's own
    `_required_generator_settings`.
    """
    expected = {
        generator_cls._type: generator_cls._required_generator_settings
        for generator_cls in _GENERATOR_CLASSES
    }
    assert configuration_validation._GENERATOR_SETTINGS_KEYS == expected


def test_single_point_generator_proposes_quantities_in_their_declared_unit() -> None:
    # Declared units on purpose don't match any "internal" unit system --
    # SinglePointParameterGenerator never converts, it just parses what the
    # config declared (see tnt.potential's module docstring for why).
    potential_settings = {
        "bh": {
            "type": "PlummerPotential",
            "include": True,
            "parameters": {
                "m_tot": {"value": 5.0, "unit": "kg", "fixed": True},
                "r_s": {"value": 1.0, "unit": "pc", "fixed": True},
            },
        },
        "dh": {
            "type": "NFWPotential",
            "parameterization": "concentration_m200",
            "include": True,
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
