"""Unit tests for `tnt.parameter_generator`."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy import units as au
from astropy.table import QTable
from unxt import Quantity

from tnt.all_models import AllModels
from tnt.model import Model
from tnt.parameter_generator import (
    AbstractParameterGenerator,
    PriorSampler,
    SinglePointParameterGenerator,
    build_parameter_generator,
    parameter_generator_required_settings,
    parameter_generator_type_names,
)
from tnt.parameter_generator.samples import _format_batch_diagnostics_summary
from tnt.potential import Potential
from tnt.priors import Prior, PriorDiagnostics


def test_parameter_generator_registry_contains_every_explicit_type() -> None:
    assert parameter_generator_type_names() == {
        "GridSearch",
        "PriorSampler",
        "PriorSupportPoints",
        "SinglePoint",
    }


def test_parameter_generator_registry_exposes_required_settings() -> None:
    assert parameter_generator_required_settings() == {
        "GridSearch": frozenset({"delta_chi2_threshold"}),
        "PriorSampler": frozenset(
            {"num_samples", "num_warmup", "seed", "target_accept_prob"}
        ),
        "PriorSupportPoints": frozenset(
            {
                "num_samples",
                "num_warmup",
                "seed",
                "target_accept_prob",
            }
        ),
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
            "expected one of: GridSearch, PriorSampler, PriorSupportPoints, SinglePoint"
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
        generator_settings={
            "seed": 0,
            "num_samples": 10,
            "num_warmup": 10,
            "target_accept_prob": 0.8,
        },
        max_new_mods_per_iter=5,
        prior=prior,
        seed=0,
        num_samples=10,
        num_warmup=10,
        target_accept_prob=0.8,
    )

    proposed = generator.generate_parameters(AllModels())

    assert len(proposed) == 5
    for candidate in proposed:
        assert candidate["dh"]["c"].ustrip("") == pytest.approx(8.0)
        m200 = candidate["dh"]["M_200"].ustrip("Msun")
        assert 1.0e10 <= m200 <= 1.0e14


def test_prior_sampler_logs_its_own_batch_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Re-review R2: `Prior.sample` discarded its own diagnostics, so the
    # batch PriorSampler actually proposes from had no retained or
    # reported diagnostics at all -- only a separate, independently-keyed
    # pilot draw did (and never actually matched any real iteration's key,
    # not even the first; see PriorSampler's own docstring).
    prior = Prior(_PRIOR_SAMPLER_POTENTIAL_SETTINGS, {}, {})
    generator = PriorSampler(
        potential_settings=_PRIOR_SAMPLER_POTENTIAL_SETTINGS,
        generator_settings={
            "seed": 0,
            "num_samples": 10,
            "num_warmup": 10,
            "target_accept_prob": 0.8,
        },
        max_new_mods_per_iter=5,
        prior=prior,
        seed=0,
        num_samples=10,
        num_warmup=10,
        target_accept_prob=0.8,
    )

    with caplog.at_level("INFO", logger="tnt.parameter_generator"):
        generator.generate_parameters(AllModels())

    messages = [r.getMessage() for r in caplog.records if "batch for" in r.getMessage()]
    assert len(messages) == 1
    message = messages[0]
    assert "iteration 0" in message
    assert "divergent draws" in message
    assert "ESS" in message
    assert "split-R-hat" in message


def test_prior_sampler_logs_a_constant_sites_non_finite_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Re-review R2 follow-up: a non-fixed parameter declaring a genuinely
    # constant prior (e.g. Delta(...)) gives that site a real NaN ESS/
    # split-R-hat -- plain min()/max() over the per-site dicts can hide
    # that NaN behind another site's healthy number depending on dict
    # iteration order, reporting a batch as fully healthy when one site's
    # diagnostics are actually undefined.
    settings = {
        "dh": {
            "type": "NFWPotential",
            "parameters": {
                "m": {
                    "unit": "Msun",
                    "fixed": False,
                    "prior": {"distribution": "LogUniform", "args": [1.0e10, 1.0e13]},
                },
                "r_s": {
                    "unit": "kpc",
                    "fixed": False,
                    "prior": {"distribution": "Delta", "args": [15.0]},
                },
            },
        },
    }
    prior = Prior(settings, {}, {})
    generator = PriorSampler(
        potential_settings=settings,
        generator_settings={
            "seed": 0,
            "num_samples": 1000,
            "num_warmup": 1000,
            "target_accept_prob": 0.8,
        },
        max_new_mods_per_iter=5,
        prior=prior,
        seed=0,
        num_samples=1000,
        num_warmup=1000,
        target_accept_prob=0.8,
    )

    with caplog.at_level("INFO", logger="tnt.parameter_generator"):
        generator.generate_parameters(AllModels())

    message = caplog.records[0].getMessage()
    assert "dh.r_s" in message
    assert "non-finite" in message
    # dh.m's own finite ESS/split-R-hat is still reported, not swallowed.
    assert "min ESS" in message
    assert "max split-R-hat" in message


def test_format_batch_diagnostics_summary_all_finite() -> None:
    diagnostics = PriorDiagnostics(
        diverging=None, ess={"a": 500.0, "b": 200.0}, split_rhat={"a": 1.0, "b": 1.01}
    )
    summary = _format_batch_diagnostics_summary(diagnostics)
    assert summary == "min ESS 200, max split-R-hat 1.010"


def test_format_batch_diagnostics_summary_names_non_finite_site_either_order() -> None:
    first_nan = PriorDiagnostics(
        diverging=None,
        ess={"a": math.nan, "b": 500.0},
        split_rhat={"a": math.nan, "b": 1.0},
    )
    last_nan = PriorDiagnostics(
        diverging=None,
        ess={"a": 500.0, "b": math.nan},
        split_rhat={"a": 1.0, "b": math.nan},
    )
    for diagnostics, non_finite_name in ((first_nan, "a"), (last_nan, "b")):
        summary = _format_batch_diagnostics_summary(diagnostics)
        assert "min ESS 500" in summary
        assert f"non-finite ESS/split-R-hat at 1 site(s): {non_finite_name}" in summary


def test_format_batch_diagnostics_summary_all_non_finite() -> None:
    diagnostics = PriorDiagnostics(
        diverging=None, ess={"a": math.nan}, split_rhat={"a": math.nan}
    )
    summary = _format_batch_diagnostics_summary(diagnostics)
    assert summary == (
        "no site has a finite ESS/split-R-hat; "
        "non-finite ESS/split-R-hat at 1 site(s): a"
    )


def test_format_batch_diagnostics_summary_no_parameters() -> None:
    diagnostics = PriorDiagnostics(diverging=None, ess={}, split_rhat={})
    assert _format_batch_diagnostics_summary(diagnostics) == (
        "no non-fixed parameters to sample"
    )


def test_prior_sampler_caps_at_max_new_mods_per_iter() -> None:
    prior = Prior(_PRIOR_SAMPLER_POTENTIAL_SETTINGS, {}, {})
    generator = PriorSampler(
        potential_settings=_PRIOR_SAMPLER_POTENTIAL_SETTINGS,
        generator_settings={
            "seed": 0,
            "num_samples": 10,
            "num_warmup": 10,
            "target_accept_prob": 0.8,
        },
        max_new_mods_per_iter=3,
        prior=prior,
        seed=0,
        num_samples=10,
        num_warmup=10,
        target_accept_prob=0.8,
    )

    assert len(generator.generate_parameters(AllModels())) == 3


def _completed_iteration_model(iteration: int) -> Model:
    return Model(
        potential=Potential(components={}),
        valid_potential=True,
        raw_parameters={
            "dh": {"c": Quantity(8.0, ""), "M_200": Quantity(1e12, "Msun")}
        },
        orblib_done=False,
        weights_done=False,
        weights=None,
        chi2=None,
        iteration=iteration,
    )


def test_prior_sampler_draws_a_fresh_batch_each_iteration() -> None:
    # PR 61 audit F1: PriorSampler used to reconstruct jax.random.PRNGKey(seed)
    # on every call, ignoring all_models entirely, so every iteration (and
    # every resume) repeated the exact first batch. Folding
    # all_models.n_iterations() into the key fixes this without making
    # PriorSampler score-conditioned -- it still never reads chi2.
    prior = Prior(_PRIOR_SAMPLER_POTENTIAL_SETTINGS, {}, {})
    generator = PriorSampler(
        potential_settings=_PRIOR_SAMPLER_POTENTIAL_SETTINGS,
        generator_settings={
            "seed": 0,
            "num_samples": 10,
            "num_warmup": 10,
            "target_accept_prob": 0.8,
        },
        max_new_mods_per_iter=4,
        prior=prior,
        seed=0,
        num_samples=10,
        num_warmup=10,
        target_accept_prob=0.8,
    )

    first_round = [
        c["dh"]["M_200"].ustrip("Msun")
        for c in generator.generate_parameters(AllModels())
    ]

    one_iteration_done = AllModels().append(_completed_iteration_model(0))
    assert one_iteration_done.n_iterations() == 1
    second_round = [
        c["dh"]["M_200"].ustrip("Msun")
        for c in generator.generate_parameters(one_iteration_done)
    ]

    assert first_round != second_round

    # Reproducible: resuming from the same recorded history (same
    # n_iterations()) proposes the identical next batch, not a new one.
    resumed_second_round = [
        c["dh"]["M_200"].ustrip("Msun")
        for c in generator.generate_parameters(one_iteration_done)
    ]
    assert second_round == resumed_second_round

    # And matches an uninterrupted run that reached the same point directly.
    two_iterations_done = one_iteration_done.append(_completed_iteration_model(1))
    assert two_iterations_done.n_iterations() == 2
    third_round = [
        c["dh"]["M_200"].ustrip("Msun")
        for c in generator.generate_parameters(two_iterations_done)
    ]
    assert third_round != second_round


def test_build_parameter_generator_dispatches_prior_sampler() -> None:
    prior = Prior(_PRIOR_SAMPLER_POTENTIAL_SETTINGS, {}, {})
    parameter_space_settings = {
        "generator_type": "PriorSampler",
        "generator_settings": {
            "seed": 0,
            "num_samples": 10,
            "num_warmup": 10,
            "target_accept_prob": 0.8,
        },
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
        "generator_settings": {
            "seed": 0,
            "num_samples": 10,
            "num_warmup": 10,
            "target_accept_prob": 0.8,
        },
        "stopping_criteria": {"max_new_mods_per_iter": 4},
    }

    with pytest.raises(ValueError, match="requires a built Prior"):
        build_parameter_generator(
            parameter_space_settings, _PRIOR_SAMPLER_POTENTIAL_SETTINGS
        )


class _BatchGenerator(AbstractParameterGenerator):
    """Exercise the shared contract with a controlled ordered proposal batch."""

    _type = "Batch"
    candidates: list

    def _propose_free_parameters(self, all_models):
        return self.candidates


_IDENTITY_SETTINGS = {
    "halo": {
        "parameters": {
            "m": {"unit": "Msun", "fixed": False},
            "r_s": {"unit": "kpc", "fixed": True, "value": 1.0},
        }
    }
}


def _point(mass, radius=1.0, radius_unit="kpc"):
    return {"halo": {"m": Quantity(mass, "Msun"), "r_s": Quantity(radius, radius_unit)}}


def test_duplicates_are_removed_before_cap_with_units_and_stable_order() -> None:
    candidates = [
        _point(1.0),
        _point(1.0, 1000.0, "pc"),
        _point(2.0),
        _point(2.0),
        _point(3.0),
        _point(4.0),
    ]
    history = AllModels(
        QTable({"halo.m": [1.0] * au.Msun, "halo.r_s": [1000.0] * au.pc})
    )
    snapshot = history.table.copy()
    generator = _BatchGenerator(
        potential_settings=_IDENTITY_SETTINGS,
        generator_settings={},
        max_new_mods_per_iter=2,
        candidates=candidates,
    )
    actual = generator.generate_parameters(history)
    assert actual[0] is candidates[2]
    assert actual[1] is candidates[4]
    for column in snapshot.colnames:
        np.testing.assert_array_equal(history.table[column], snapshot[column])
    assert len(generator.candidates) == 6


def test_duplicate_identity_has_no_tolerance_and_includes_fixed_values() -> None:
    points = [_point(1.0), _point(np.nextafter(1.0, 2.0)), _point(1.0, 2.0)]
    generator = _BatchGenerator(
        potential_settings=_IDENTITY_SETTINGS,
        generator_settings={},
        max_new_mods_per_iter=10,
        candidates=points,
    )
    actual = generator.generate_parameters(AllModels())
    assert len(actual) == 3
    assert float(actual[0]["halo"]["m"].ustrip("Msun")) != float(
        actual[1]["halo"]["m"].ustrip("Msun")
    )


def test_single_point_stops_after_existing_failed_evaluation() -> None:
    generator = SinglePointParameterGenerator(
        potential_settings={
            "dh": {
                "type": "NFWPotential",
                "parameterization": "concentration_m200",
                "parameters": {
                    "c": {"fixed": True, "value": 8.0},
                    "M_200": {"fixed": True, "value": 1e12, "unit": "Msun"},
                },
            }
        },
        generator_settings={},
        max_new_mods_per_iter=10,
    )
    assert len(generator.generate_parameters(AllModels())) == 1
    history = AllModels().append(_completed_iteration_model(0))
    assert not history.has_successful_model()
    assert generator.generate_parameters(history) == []


@pytest.mark.parametrize("bad_unit", ["s", None])
def test_duplicate_filter_rejects_incompatible_or_missing_history_columns(bad_unit):
    table = QTable({"halo.m": [1.0] * au.Msun})
    if bad_unit is not None:
        table["halo.r_s"] = [1.0] * au.Unit(bad_unit)
    generator = _BatchGenerator(
        potential_settings=_IDENTITY_SETTINGS,
        generator_settings={},
        max_new_mods_per_iter=2,
        candidates=[_point(2.0)],
    )
    with pytest.raises(ValueError, match="halo.r_s"):
        generator.generate_parameters(AllModels(table))


def test_factory_configures_generator_outputs_without_creating_them(tmp_path) -> None:
    generator = build_parameter_generator(
        {
            "generator_type": "SinglePoint",
            "generator_settings": {},
            "stopping_criteria": {"max_new_mods_per_iter": 1},
        },
        {},
        output_directory=tmp_path,
    )
    directory = tmp_path / "parameter_generator"
    assert not directory.exists()
    assert generator.diagnostic_path("prior_diagnostics") == (
        directory / "prior_diagnostics.png"
    )
    assert generator.diagnostic_path("support_points_diagnostics", 3) == (
        directory / "support_points_diagnostics_iteration_0003.png"
    )


def test_single_point_needs_no_diagnostic_implementation_or_output(tmp_path) -> None:
    generator = build_parameter_generator(
        {
            "generator_type": "SinglePoint",
            "generator_settings": {"plot_diagnostics": True},
            "stopping_criteria": {"max_new_mods_per_iter": 1},
        },
        {},
        output_directory=tmp_path,
    )
    assert generator.generate_parameters(AllModels()) == [{}]
    assert not (tmp_path / "parameter_generator").exists()
