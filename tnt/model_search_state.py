"""Coordinated persistence and recovery for model-search state."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Self

from astropy.table import QTable

from tnt.all_models import AllModels
from tnt.run_config_log import (
    RUN_IDS_WITHOUT_ITERATIONS_METADATA_KEY,
    TOTAL_RUNS_METADATA_KEY,
    RunConfigLog,
    _refresh_run_metadata,
    _validate_log_path,
    _validate_table,
)


class ModelSearchStateError(ValueError):
    """Raised when persisted model-search state is incomplete or inconsistent."""


@dataclass(frozen=True)
class ModelSearchState:
    """The `AllModels` and `RunConfigLog` pair forming one checkpoint."""

    all_models: AllModels
    run_config_log: RunConfigLog

    def __post_init__(self) -> None:
        """Require both objects to describe the same cumulative iterations."""
        _validate_pair(self.all_models, self.run_config_log)

    @classmethod
    def read(
        cls,
        all_models_path: Path,
        run_config_log_path: Path,
        *,
        repair_log_ahead: bool = False,
    ) -> Self:
        """Read a checkpoint, optionally repairing the safely recoverable mismatch.

        Publishing writes the run log first. A crash before the models-table
        replacement can therefore leave extra trailing log rows; explicit
        repair truncates those rows to the iterations present in `AllModels`.
        A models table ahead of its log is rejected because its missing run
        provenance cannot be reconstructed safely.
        """
        _validate_destinations(all_models_path, run_config_log_path)
        models = (
            AllModels.read(all_models_path)
            if all_models_path.is_file()
            else AllModels()
        )
        run_log = (
            RunConfigLog.read(run_config_log_path)
            if run_config_log_path.is_file()
            else RunConfigLog()
        )
        model_iterations = models.n_iterations()
        logged_iterations = len(run_log)
        if logged_iterations == model_iterations:
            return cls(models, run_log)
        if logged_iterations > model_iterations:
            if not repair_log_ahead:
                raise ModelSearchStateError(
                    "RunConfigLog is ahead of AllModels; read again with "
                    "repair_log_ahead=True to discard its unpublished trailing "
                    "iteration provenance."
                )
            repaired_log = run_log.truncate(model_iterations)
            repaired_log.write(run_config_log_path)
            return cls(models, repaired_log)
        raise ModelSearchStateError(
            "AllModels is ahead of RunConfigLog; run provenance is missing and "
            "cannot be reconstructed safely."
        )

    def write(self, all_models_path: Path, run_config_log_path: Path) -> None:
        """Validate temporary files, then publish the log before `AllModels`.

        Each replacement is atomic on its own filesystem. If the second
        replacement is interrupted, `read(..., repair_log_ahead=True)` can
        safely return to the last fully published models-table checkpoint.
        """
        _validate_pair(self.all_models, self.run_config_log)
        _validate_destinations(all_models_path, run_config_log_path)
        all_models_path.parent.mkdir(parents=True, exist_ok=True)
        run_config_log_path.parent.mkdir(parents=True, exist_ok=True)

        log_table = self.run_config_log.table.copy()
        _validate_table(log_table)
        _refresh_run_metadata(log_table, run_config_log_path.parent)

        temporary_log = _temporary_path(run_config_log_path)
        temporary_models: Path | None = None
        write_models = bool(self.all_models.table.colnames)
        try:
            log_table.write(temporary_log, format="ascii.ecsv", overwrite=True)
            _validate_temporary_log(
                temporary_log,
                log_table,
            )

            if write_models:
                temporary_models = _temporary_path(all_models_path)
                self.all_models.table.write(
                    temporary_models,
                    format="ascii.ecsv",
                    overwrite=True,
                )
                restored_models = AllModels.read(temporary_models)
                _validate_pair(restored_models, RunConfigLog(log_table.copy()))
            elif all_models_path.is_file():
                persisted_models = AllModels.read(all_models_path)
                if len(persisted_models):
                    raise ModelSearchStateError(
                        "Refusing to publish empty AllModels state over an existing "
                        "nonempty models table."
                    )

            os.replace(temporary_log, run_config_log_path)
            temporary_log = None
            if temporary_models is not None:
                os.replace(temporary_models, all_models_path)
                temporary_models = None
            self.run_config_log.table.meta.update(log_table.meta)
        finally:
            if temporary_log is not None:
                temporary_log.unlink(missing_ok=True)
            if temporary_models is not None:
                temporary_models.unlink(missing_ok=True)


def _validate_pair(all_models: AllModels, run_config_log: RunConfigLog) -> None:
    """Require model and provenance state to cover the same iterations."""
    model_iterations = all_models.n_iterations()
    if len(run_config_log) != model_iterations:
        raise ModelSearchStateError(
            "AllModels and RunConfigLog must describe the same number of "
            f"iterations; received {model_iterations} and {len(run_config_log)}."
        )


def _validate_destinations(
    all_models_path: Path,
    run_config_log_path: Path,
) -> None:
    """Validate distinct checkpoint destinations and the fixed log filename."""
    _validate_log_path(run_config_log_path)
    models_destination = all_models_path.expanduser().resolve()
    log_destination = run_config_log_path.expanduser().resolve()
    if models_destination == log_destination:
        raise ValueError("AllModels and RunConfigLog require different paths.")


def _temporary_path(destination: Path) -> Path:
    """Reserve a temporary path beside its eventual atomic-replace target."""
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        return Path(stream.name)


def _validate_temporary_log(
    path: Path,
    expected: QTable,
) -> None:
    """Read back and validate a prepared run log before publishing it."""
    restored = QTable.read(path, format="ascii.ecsv")
    _validate_table(restored)
    if any(
        list(restored[name]) != list(expected[name]) for name in restored.colnames
    ):
        raise ModelSearchStateError(
            "Temporary RunConfigLog did not preserve its iteration rows."
        )
    for key in (TOTAL_RUNS_METADATA_KEY, RUN_IDS_WITHOUT_ITERATIONS_METADATA_KEY):
        if restored.meta.get(key) != expected.meta[key]:
            raise ModelSearchStateError(
                f"Temporary RunConfigLog did not preserve metadata {key!r}."
            )
