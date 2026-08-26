"""Unit tests for `ModelIterator`'s control flow, using fake collaborators.

Real orbit integration and weight solving are still unimplemented (see
`tnt.model_iterator`'s module docstring). Potential-building is now real for
native-galax types (`tnt.potential`), but these tests still exercise
`ModelIterator`'s own logic -- the generate/evaluate/record/stop loop,
mass-scale rescaling, failure handling, and logging -- against small fakes
standing in for `Potential`/`OrbitLibrary`/`AbstractWeightSolver`, with
`build_potential` itself monkeypatched per test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import unxt as u

import tnt.model_iterator as model_iterator_module
from tnt.model_iterator import ModelIterator
from tnt.run_config_log import (
    RunConfigLog,
    RunManifestReference,
)

# ---------------------------------------------------------------------------
# Fakes standing in for Potential / OrbitLibrary / AbstractWeightSolver.
# ---------------------------------------------------------------------------


class FakePotential:
    def __init__(self, mass: float = 1.0, fail_orbit_library: bool = False) -> None:
        self.mass = mass
        self.fail_orbit_library = fail_orbit_library
        self.components: dict[str, Any] = {}

    def generate_orbit_library(
        self,
        settings: Any,
        sampler: Any,
        dithering: Any,
    ) -> Any:
        if self.fail_orbit_library:
            raise RuntimeError("orbit integration failed")
        return FakeOrbitLibrary(self.mass)

    def rescale(self, mass_scale: float) -> FakePotential:
        return FakePotential(mass=self.mass * mass_scale)


class FakeOrbitLibrary:
    def __init__(self, mass: float) -> None:
        self.mass = mass

    def rescaled(self, mass_scale: float) -> FakeOrbitLibrary:
        return FakeOrbitLibrary(self.mass * mass_scale)


class _FakeWeightResult(NamedTuple):
    weights: str
    chi2: dict[str, float]


class FakeWeightSolver:
    """`chi2["chi2"] == 10.0 / orbit_library.mass`, so the mass is recoverable."""

    def __init__(self, fail_on_mass: frozenset[float] = frozenset()) -> None:
        self.fail_on_mass = fail_on_mass

    def solve(
        self, orbit_library: FakeOrbitLibrary, kinematic_data: Any
    ) -> _FakeWeightResult:
        if orbit_library.mass in self.fail_on_mass:
            raise RuntimeError("weight solve failed")
        return _FakeWeightResult(
            weights="weights", chi2={"chi2": 10.0 / orbit_library.mass}
        )


class FakeParameterGenerator:
    """Proposes one `ParameterSet` per round, for `n_rounds` rounds, then stops."""

    def __init__(self, n_rounds: int) -> None:
        self.n_rounds = n_rounds
        self.calls = 0

    def generate_parameters(self, all_models: Any) -> list[dict[str, dict[str, float]]]:
        self.calls += 1
        if self.calls > self.n_rounds:
            return []
        return [{"bh": {"m": 1.0}}]


def _run_reference(run_id: int = 0) -> RunManifestReference:
    return RunManifestReference(
        repository=Path("/archive/config_repository"),
        run_id=run_id,
        run_manifest_path=f"runs/{run_id:04d}/run_manifest.yaml",
        resolved_config_path=f"runs/{run_id:04d}/resolved_config.yaml",
    )


def _make_iterator(**overrides: Any) -> ModelIterator:
    iterator = ModelIterator.__new__(ModelIterator)
    iterator.potential_settings = {}
    iterator.resolved_potential = {}
    iterator.unit_system = u.unitsystem("kpc", "Myr", "Msun", "rad", "Lsun")
    iterator.cosmological_parameters = {}
    iterator.kinematic_data = {}
    iterator.weight_solver = FakeWeightSolver()
    iterator.orbit_library_settings = {}
    iterator.orbit_sampler = None
    iterator.orbit_dithering = None
    iterator.potential_rescalings = {"enabled": False}
    iterator.parameter_generator = FakeParameterGenerator(n_rounds=1)
    iterator.stopping_criteria = {
        "minimum_delta_chi2": {
            "enabled": True,
            "mode": "absolute",
            "value": 0.5,
        },
        "n_new_iter": 10,
        "target_model_count": 10,
    }
    iterator.which_chi2 = "chi2"
    iterator.execution_settings = {
        "model_processing_order": "model_by_model",
        "orbit_workers": "all_available",
        "weight_workers": "all_available",
    }
    iterator.runtime_configuration = {}
    iterator.portable_configuration = {}
    iterator.workspace_root = Path("/workspace")
    iterator.logfile_path = None
    iterator.configuration_repository = Path("/archive/config_repository")
    iterator.critical_configuration = {"potential": {}}
    iterator.run_id = None
    iterator.run_manifest = None
    for name, value in overrides.items():
        setattr(iterator, name, value)
    return iterator


@pytest.fixture(autouse=True)
def _isolate_run_archiving(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep control-flow unit tests independent of filesystem provenance."""

    def fake_archive(iterator: ModelIterator) -> RunManifestReference:
        run_id = getattr(iterator, "_test_run_id", 0)
        iterator._test_run_id = run_id + 1
        return _run_reference(run_id)

    monkeypatch.setattr(
        ModelIterator,
        "_archive_run",
        fake_archive,
    )
    monkeypatch.setattr(
        RunConfigLog,
        "baseline_run_reference",
        lambda *_args, **_kwargs: None,
    )


def _patch_build_potential(
    monkeypatch: pytest.MonkeyPatch, potential: FakePotential
) -> None:
    monkeypatch.setattr(
        model_iterator_module,
        "build_potential",
        lambda resolved, parameter_values, unit_system, cosmological_parameters: (
            potential
        ),
    )


# ---------------------------------------------------------------------------
# _evaluate / _solve
# ---------------------------------------------------------------------------


def test_evaluate_returns_solved_model_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterator = _make_iterator()
    _patch_build_potential(monkeypatch, FakePotential(mass=1.0))

    (model,) = iterator._evaluate({"bh": {"m": 1.0}})

    assert model.orblib_done
    assert model.weights_done
    assert model.chi2 == {"chi2": 10.0}
    assert model.weights == "weights"


def test_evaluate_logs_and_flags_orbit_integration_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    iterator = _make_iterator()
    _patch_build_potential(monkeypatch, FakePotential(fail_orbit_library=True))

    with caplog.at_level(logging.WARNING, logger="tnt.model_iterator"):
        (model,) = iterator._evaluate({"bh": {"m": 1.0}})

    assert model.orblib_done is False
    assert model.weights_done is False
    assert model.weights is None
    assert model.chi2 is None
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert "orbit integration failed" in record.message.lower()
    assert "{'bh': {'m': 1.0}}" in record.message


def test_evaluate_logs_and_flags_weight_solve_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    iterator = _make_iterator(
        weight_solver=FakeWeightSolver(fail_on_mass=frozenset({1.0}))
    )
    _patch_build_potential(monkeypatch, FakePotential(mass=1.0))

    with caplog.at_level(logging.WARNING, logger="tnt.model_iterator"):
        (model,) = iterator._evaluate({"bh": {"m": 1.0}})

    assert model.orblib_done is True
    assert model.weights_done is False
    assert model.chi2 is None
    [record] = caplog.records
    assert "weight solve failed" in record.message.lower()
    assert "mass_scale=1.0" in record.message


def test_evaluate_adds_rescaled_models_without_duplicating_the_unscaled_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterator = _make_iterator(
        potential_rescalings={
            "enabled": True,
            "mass_scale_range": {"minimum": 0.5, "maximum": 2.0},
            "range_count": 4,  # linspace(0.5, 2.0, 4) == [0.5, 1.0, 1.5, 2.0]
            "spacing": "linear",
        }
    )
    _patch_build_potential(monkeypatch, FakePotential(mass=1.0))

    models = iterator._evaluate({"bh": {"m": 1.0}})

    masses = sorted(10.0 / model.chi2["chi2"] for model in models)
    assert masses == [0.5, 1.0, 1.5, 2.0]
    assert all(model.orblib_done and model.weights_done for model in models)


def test_evaluate_logs_mass_scale_of_a_failed_rescaled_model(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    iterator = _make_iterator(
        weight_solver=FakeWeightSolver(fail_on_mass=frozenset({1.5, 2.0})),
        potential_rescalings={
            "enabled": True,
            "mass_scale_range": {"minimum": 0.5, "maximum": 2.0},
            "range_count": 4,
            "spacing": "linear",
        },
    )
    _patch_build_potential(monkeypatch, FakePotential(mass=1.0))

    with caplog.at_level(logging.WARNING, logger="tnt.model_iterator"):
        models = iterator._evaluate({"bh": {"m": 1.0}})

    failed = [model for model in models if not model.weights_done]
    succeeded = [model for model in models if model.weights_done]
    assert len(failed) == 2
    assert len(succeeded) == 2
    logged_mass_scales = {record.message for record in caplog.records}
    assert any("mass_scale=1.5" in message for message in logged_mass_scales)
    assert any("mass_scale=2.0" in message for message in logged_mass_scales)


# ---------------------------------------------------------------------------
# mass_scales
# ---------------------------------------------------------------------------


def test_mass_scales_excludes_the_unscaled_point_and_is_cached() -> None:
    iterator = _make_iterator(
        potential_rescalings={
            "enabled": True,
            "mass_scale_range": {"minimum": 0.5, "maximum": 2.0},
            "range_count": 4,
            "spacing": "linear",
        }
    )

    scales = iterator.mass_scales

    assert 1.0 not in scales
    assert scales == (0.5, 1.5, 2.0)
    assert iterator.mass_scales is scales  # cached_property: computed once


# ---------------------------------------------------------------------------
# chi2 stopping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "threshold", "previous", "best", "expected"),
    [
        ("absolute", 5.0, 100.0, 90.0, False),
        ("absolute", 10.0, 100.0, 90.0, False),
        ("absolute", 11.0, 100.0, 90.0, True),
        ("relative", 0.04, 100.0, 95.0, False),
        ("relative", 0.05, 100.0, 95.0, False),
        ("relative", 0.06, 100.0, 95.0, True),
        ("absolute", 0.0, 100.0, 110.0, True),
        ("relative", 0.1, 0.0, 0.0, True),
        ("relative", 0.0, 0.0, 0.0, False),
    ],
)
def test_chi2_stopped_improving(
    mode: str,
    threshold: float,
    previous: float,
    best: float,
    expected: bool,
) -> None:
    iterator = _make_iterator(
        stopping_criteria={
            "minimum_delta_chi2": {
                "enabled": True,
                "mode": mode,
                "value": threshold,
            },
            "n_new_iter": 10,
            "target_model_count": 10,
        }
    )

    assert iterator._chi2_stopped_improving(previous, best) is expected


def test_chi2_stopping_can_be_disabled() -> None:
    iterator = _make_iterator(
        stopping_criteria={
            "minimum_delta_chi2": {
                "enabled": False,
                "mode": "absolute",
                "value": 0.5,
            },
            "n_new_iter": 10,
            "target_model_count": 10,
        }
    )

    assert iterator._chi2_stopped_improving(100.0, 110.0) is False


def test_chi2_stopped_improving_rejects_unknown_mode() -> None:
    iterator = _make_iterator(
        stopping_criteria={
            "minimum_delta_chi2": {
                "enabled": True,
                "mode": "unknown",
                "value": 0.5,
            },
            "n_new_iter": 10,
            "target_model_count": 10,
        }
    )

    with pytest.raises(ValueError, match="must be 'absolute' or 'relative'"):
        iterator._chi2_stopped_improving(100.0, 90.0)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_rejects_stage_by_stage_before_generating_parameters() -> None:
    generator = FakeParameterGenerator(n_rounds=1)
    iterator = _make_iterator(
        parameter_generator=generator,
        execution_settings={
            "model_processing_order": "stage_by_stage",
            "orbit_workers": "all_available",
            "weight_workers": "all_available",
        },
    )

    with pytest.raises(
        NotImplementedError,
        match=r"model_processing_order='stage_by_stage' is not implemented",
    ):
        iterator.run()

    assert generator.calls == 0


def test_run_stops_when_generator_proposes_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    iterator = _make_iterator(parameter_generator=FakeParameterGenerator(n_rounds=2))
    monkeypatch.setattr(ModelIterator, "_chi2_stopped_improving", lambda *_: False)
    _patch_build_potential(monkeypatch, FakePotential(mass=1.0))

    with caplog.at_level(logging.INFO, logger="tnt.model_iterator"):
        models, config_log = iterator.run()

    assert len(models) == 2
    assert models.n_iterations() == 2
    assert len(config_log) == 2
    assert list(config_log.table["iteration"]) == [0, 1]
    assert "proposed nothing more" in caplog.text
    assert "Stopped after 2 iteration(s), 2 model(s) total." in caplog.text


def test_run_records_then_stops_when_initial_iteration_has_no_success(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    generator = FakeParameterGenerator(n_rounds=3)
    iterator = _make_iterator(
        parameter_generator=generator,
        weight_solver=FakeWeightSolver(fail_on_mass=frozenset({1.0})),
    )
    _patch_build_potential(monkeypatch, FakePotential(mass=1.0))

    with caplog.at_level(logging.INFO, logger="tnt.model_iterator"):
        models, config_log = iterator.run()

    assert generator.calls == 1
    assert len(models) == 1
    assert models.n_iterations() == 1
    assert models.has_successful_model() is False
    assert len(config_log) == 1
    assert "Initial iteration 0 produced no successful model" in caplog.text


def test_run_does_not_resume_an_all_failed_model_table(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    failing_iterator = _make_iterator(
        weight_solver=FakeWeightSolver(fail_on_mass=frozenset({1.0}))
    )
    _patch_build_potential(monkeypatch, FakePotential(mass=1.0))
    models, config_log = failing_iterator.run()

    generator = FakeParameterGenerator(n_rounds=3)
    resumed_iterator = _make_iterator(parameter_generator=generator)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="tnt.model_iterator"):
        resumed_models, resumed_config_log = resumed_iterator.run(models, config_log)

    assert generator.calls == 0
    assert resumed_models is models
    assert resumed_config_log is config_log
    assert len(resumed_models) == 1
    assert len(resumed_config_log) == 1
    assert "Resumed AllModels contains no successful model" in caplog.text


def test_each_call_on_the_same_iterator_gets_a_new_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterator = _make_iterator(
        parameter_generator=FakeParameterGenerator(n_rounds=2),
        stopping_criteria={
            "minimum_delta_chi2": {
                "enabled": False,
                "mode": "absolute",
                "value": 0.0,
            },
            "n_new_iter": 1,
            "target_model_count": 10,
        },
    )
    _patch_build_potential(monkeypatch, FakePotential(mass=1.0))
    models, config_log = iterator.run()
    assert iterator.run_id == 0

    models, config_log = iterator.run(models, config_log)

    assert iterator.run_id == 1
    assert iterator.run_manifest == _run_reference(1)
    assert len(models) == 2
    assert list(config_log.table["run_id"]) == [0, 1]


def test_run_rejects_mismatched_models_and_config_log_before_archiving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_iterator = _make_iterator()
    _patch_build_potential(monkeypatch, FakePotential(mass=1.0))
    models, _ = first_iterator.run()
    resumed_iterator = _make_iterator()

    with pytest.raises(
        ValueError,
        match="AllModels and RunConfigLog must describe the same number",
    ):
        resumed_iterator.run(models, RunConfigLog())

    assert resumed_iterator.run_manifest is None


def test_run_continues_after_later_iteration_has_no_success(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    generator = FakeParameterGenerator(n_rounds=3)
    iterator = _make_iterator(
        parameter_generator=generator,
        weight_solver=FakeWeightSolver(fail_on_mass=frozenset({2.0})),
    )
    potentials = iter(
        [
            FakePotential(mass=1.0),
            FakePotential(mass=2.0),
            FakePotential(mass=3.0),
        ]
    )
    monkeypatch.setattr(
        model_iterator_module,
        "build_potential",
        lambda settings, mges, unit_system, cosmological_parameters: next(potentials),
    )

    with caplog.at_level(logging.INFO, logger="tnt.model_iterator"):
        models, config_log = iterator.run()

    assert generator.calls == 4
    assert len(models) == 3
    assert models.n_iterations() == 3
    assert list(models.table["weights_done"]) == [True, False, True]
    assert len(config_log) == 3
    assert "Iteration 1 produced no successful model" in caplog.text
    assert "retaining the previous best value for chi2 and continuing" in caplog.text


def test_run_stops_starting_iterations_at_target_model_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterator = _make_iterator(
        parameter_generator=FakeParameterGenerator(n_rounds=10),
        stopping_criteria={"n_new_iter": 10, "target_model_count": 2},
    )
    monkeypatch.setattr(ModelIterator, "_chi2_stopped_improving", lambda *_: False)
    _patch_build_potential(monkeypatch, FakePotential(mass=1.0))

    models, config_log = iterator.run()

    assert len(models) == 2
    assert len(config_log) == 2


def test_run_records_one_config_log_row_per_iteration_not_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterator = _make_iterator(
        parameter_generator=FakeParameterGenerator(n_rounds=2),
        potential_rescalings={
            "enabled": True,
            "mass_scale_range": {"minimum": 0.5, "maximum": 2.0},
            "range_count": 4,
            "spacing": "linear",
        },
    )
    monkeypatch.setattr(ModelIterator, "_chi2_stopped_improving", lambda *_: False)
    _patch_build_potential(monkeypatch, FakePotential(mass=1.0))

    models, config_log = iterator.run()

    assert len(models) == 2 * 4  # 2 rounds, 1 base + 3 rescaled models each
    assert len(config_log) == 2  # still one row per round


def test_new_run_keeps_cumulative_labels_when_resuming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_iterator_module,
        "ensure_resume_compatible",
        lambda *_: None,
    )
    first_iterator = _make_iterator(
        parameter_generator=FakeParameterGenerator(n_rounds=2),
        stopping_criteria={"n_new_iter": 2, "target_model_count": 10},
    )
    monkeypatch.setattr(ModelIterator, "_chi2_stopped_improving", lambda *_: False)
    _patch_build_potential(monkeypatch, FakePotential(mass=1.0))

    models, config_log = first_iterator.run()
    assert len(models) == 2

    resumed_iterator = _make_iterator(
        parameter_generator=FakeParameterGenerator(n_rounds=2),
        stopping_criteria={"n_new_iter": 2, "target_model_count": 10},
        _test_run_id=1,
    )
    models, config_log = resumed_iterator.run(models, config_log)

    assert len(models) == 4
    assert models.n_iterations() == 4
    assert len(config_log) == 4
    assert list(config_log.table["run_id"]) == [0, 0, 1, 1]


def test_run_logs_improving_chi2_stopping_reason(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    iterator = _make_iterator(parameter_generator=FakeParameterGenerator(n_rounds=10))
    _patch_build_potential(monkeypatch, FakePotential(mass=1.0))

    with caplog.at_level(logging.INFO, logger="tnt.model_iterator"):
        models, _ = iterator.run()

    # previous_best_chi2 starts None, so the delta-chi2 check can't fire
    # until the *second* round -- it's what stops the third round from running.
    assert len(models) == 2
    assert "chi2 stopped improving" in caplog.text


def test_run_disabled_chi2_threshold_leaves_iteration_limit_to_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterator = _make_iterator(
        parameter_generator=FakeParameterGenerator(n_rounds=10),
        stopping_criteria={
            "minimum_delta_chi2": {
                "enabled": False,
                "mode": "absolute",
                "value": 0.5,
            },
            "n_new_iter": 3,
            "target_model_count": 10,
        },
    )
    _patch_build_potential(monkeypatch, FakePotential(mass=1.0))

    models, _ = iterator.run()

    # Every round has the same chi2, but the disabled threshold leaves the
    # iteration limit responsible for stopping the search.
    assert len(models) == 3
    assert models.n_iterations() == 3
