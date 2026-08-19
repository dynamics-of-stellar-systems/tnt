from hashlib import sha256
from pathlib import Path

import pytest
import yaml
from astropy.table import QTable

import tnt.model_search_state as state_module
from tnt.all_models import AllModels
from tnt.configuration import _semantic_configuration_sha256
from tnt.model_search_state import ModelSearchState, ModelSearchStateError
from tnt.run_config_log import (
    RUN_CONFIG_LOG_FILENAME,
    RUN_IDS_WITHOUT_ITERATIONS_METADATA_KEY,
    TOTAL_RUNS_METADATA_KEY,
    RunConfigLog,
)


def _write_run_manifest(tmp_path: Path, run_id: int) -> Path:
    config = {"run": run_id}
    semantic_sha256 = _semantic_configuration_sha256(config)
    repository = tmp_path / "config_repository"
    resolved_path = (
        repository
        / "configurations"
        / f"{run_id:04d}-{semantic_sha256[:8]}"
        / "resolved_config.yaml"
    )
    resolved_path.parent.mkdir(parents=True)
    resolved_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest_path = repository / "manifests" / f"{run_id:04d}-run_manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": 2,
        "run_id": run_id,
        "configuration": {
            "snapshot_id": run_id,
            "semantic_sha256": semantic_sha256,
            "resolved": resolved_path.relative_to(repository).as_posix(),
            "manifest": manifest_path.relative_to(repository).as_posix(),
            "resolved_config_sha256": sha256(resolved_path.read_bytes()).hexdigest(),
        },
    }
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return manifest_path


def _models(*iterations: int) -> AllModels:
    return AllModels(
        QTable(
            {
                "iteration": list(iterations),
                "weights_done": [True] * len(iterations),
            }
        )
    )


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    return (
        tmp_path / "output" / "all_models.ecsv",
        tmp_path / "config_repository" / RUN_CONFIG_LOG_FILENAME,
    )


def test_write_and_read_publish_one_validated_pair(tmp_path: Path) -> None:
    _write_run_manifest(tmp_path, 0)
    models_path, log_path = _paths(tmp_path)
    state = ModelSearchState(_models(0), RunConfigLog().append(0, 0))

    state.write(models_path, log_path)
    restored = ModelSearchState.read(models_path, log_path)

    assert restored.all_models.n_iterations() == 1
    assert list(restored.run_config_log.table["run_id"]) == [0]
    assert restored.run_config_log.table.meta[TOTAL_RUNS_METADATA_KEY] == 1
    assert (
        restored.run_config_log.table.meta[RUN_IDS_WITHOUT_ITERATIONS_METADATA_KEY]
        == []
    )
    assert not list(models_path.parent.glob(".*.tmp"))
    assert not list(log_path.parent.glob(".*.tmp"))


def test_write_records_zero_iteration_run_without_empty_models_file(
    tmp_path: Path,
) -> None:
    _write_run_manifest(tmp_path, 0)
    models_path, log_path = _paths(tmp_path)

    ModelSearchState(AllModels(), RunConfigLog()).write(models_path, log_path)
    restored = ModelSearchState.read(models_path, log_path)

    assert not models_path.exists()
    assert len(restored.all_models) == 0
    assert restored.run_config_log.table.meta[TOTAL_RUNS_METADATA_KEY] == 1
    assert restored.run_config_log.table.meta[
        RUN_IDS_WITHOUT_ITERATIONS_METADATA_KEY
    ] == [0]


def test_read_explicitly_repairs_run_log_ahead_of_models(tmp_path: Path) -> None:
    _write_run_manifest(tmp_path, 0)
    models_path, log_path = _paths(tmp_path)
    ModelSearchState(_models(0), RunConfigLog().append(0, 0)).write(
        models_path, log_path
    )
    RunConfigLog.read(log_path).append(1, 0).write(log_path)

    with pytest.raises(ModelSearchStateError, match="RunConfigLog is ahead"):
        ModelSearchState.read(models_path, log_path)

    repaired = ModelSearchState.read(
        models_path,
        log_path,
        repair_log_ahead=True,
    )

    assert len(repaired.run_config_log) == 1
    assert len(RunConfigLog.read(log_path)) == 1


def test_read_rejects_models_ahead_of_run_log(tmp_path: Path) -> None:
    _write_run_manifest(tmp_path, 0)
    models_path, log_path = _paths(tmp_path)
    ModelSearchState(_models(0), RunConfigLog().append(0, 0)).write(
        models_path, log_path
    )
    _models(0, 1).write(models_path)

    with pytest.raises(ModelSearchStateError, match="provenance is missing"):
        ModelSearchState.read(models_path, log_path, repair_log_ahead=True)


def test_write_publishes_run_log_before_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_run_manifest(tmp_path, 0)
    models_path, log_path = _paths(tmp_path)
    destinations: list[Path] = []
    replace = state_module.os.replace

    def record_replace(source: Path, destination: Path) -> None:
        destinations.append(destination)
        replace(source, destination)

    monkeypatch.setattr(state_module.os, "replace", record_replace)

    ModelSearchState(_models(0), RunConfigLog().append(0, 0)).write(
        models_path, log_path
    )

    assert destinations == [log_path, models_path]


def test_log_first_failure_is_recoverable(tmp_path: Path) -> None:
    _write_run_manifest(tmp_path, 0)
    models_path, log_path = _paths(tmp_path)
    ModelSearchState(_models(0), RunConfigLog().append(0, 0)).write(
        models_path, log_path
    )
    newer_state = ModelSearchState(
        _models(0, 1),
        RunConfigLog.read(log_path).append(1, 0),
    )
    replace = state_module.os.replace

    def fail_models_replace(source: Path, destination: Path) -> None:
        if destination == models_path:
            raise OSError("simulated interruption")
        replace(source, destination)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(state_module.os, "replace", fail_models_replace)
        with pytest.raises(OSError, match="simulated interruption"):
            newer_state.write(models_path, log_path)

    assert AllModels.read(models_path).n_iterations() == 1
    assert len(RunConfigLog.read(log_path)) == 2

    recovered = ModelSearchState.read(
        models_path,
        log_path,
        repair_log_ahead=True,
    )

    assert recovered.all_models.n_iterations() == 1
    assert len(recovered.run_config_log) == 1
