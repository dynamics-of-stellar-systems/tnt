"""Persist iteration-to-run provenance and derived run summary metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Self

from astropy.table import QTable

from tnt.config_parsing import _mapping, _required, _required_string
from tnt.configuration import (
    RESOLVED_CONFIG_FILENAME,
    RUN_MANIFEST_FILENAME,
    RUNS_DIRECTORY,
    _read_yaml_bytes_mapping,
)

RUN_CONFIG_LOG_FILENAME = "run_config_log.ecsv"
TOTAL_RUNS_METADATA_KEY = "total_runs"
RUN_IDS_WITHOUT_ITERATIONS_METADATA_KEY = "run_ids_without_iterations"
_COLUMNS = ("iteration", "run_id")


@dataclass(frozen=True)
class RunManifestReference:
    """Validated identity and archived resolved configuration of one TNT run."""

    repository: Path
    run_id: int
    run_manifest_path: str
    resolved_config_path: str

    @classmethod
    def from_run_manifest(cls, path: Path) -> Self:
        """Read and validate one immutable per-run manifest/configuration bundle."""
        manifest_path = path.expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Run manifest does not exist: {manifest_path}")
        if manifest_path.name != RUN_MANIFEST_FILENAME:
            raise ValueError(
                f"Run manifest filename must be {RUN_MANIFEST_FILENAME!r}."
            )
        run_directory = manifest_path.parent
        runs_directory = run_directory.parent
        if runs_directory.name != RUNS_DIRECTORY:
            raise ValueError(
                f"Run manifest {manifest_path} is not below a "
                f"{RUNS_DIRECTORY!r} directory."
            )
        repository = runs_directory.parent
        run_id = _run_id(run_directory.name)
        manifest = _read_yaml_bytes_mapping(
            manifest_path.read_bytes(), f"run manifest {manifest_path}"
        )
        recorded_run_id = _validate_run_id(_required(manifest, "run_id", "manifest"))
        if recorded_run_id != run_id:
            raise ValueError(
                f"Run manifest for directory {run_directory.name!r} records run_id "
                f"{recorded_run_id}, expected {run_id}."
            )

        configuration = _mapping(
            _required(manifest, "configuration", "manifest"),
            "manifest.configuration",
        )
        resolved_config_path = _required_string(
            configuration, "resolved", "manifest.configuration"
        )
        expected_resolved_path = PurePosixPath(
            RUNS_DIRECTORY,
            f"{run_id:04d}",
            RESOLVED_CONFIG_FILENAME,
        )
        if PurePosixPath(resolved_config_path) != expected_resolved_path:
            raise ValueError(
                f"Run manifest for run {run_id} records resolved configuration "
                f"path {resolved_config_path!r}, expected "
                f"{expected_resolved_path.as_posix()!r}."
            )
        absolute_resolved_path = repository.joinpath(*expected_resolved_path.parts)
        if not absolute_resolved_path.is_file():
            raise FileNotFoundError(
                f"Resolved configuration does not exist: {absolute_resolved_path}"
            )
        _read_yaml_bytes_mapping(
            absolute_resolved_path.read_bytes(),
            f"resolved configuration for run {run_id}",
        )
        return cls(
            repository=repository,
            run_id=run_id,
            run_manifest_path=manifest_path.relative_to(repository).as_posix(),
            resolved_config_path=resolved_config_path,
        )

    @classmethod
    def from_run_id(cls, repository: Path, run_id: int) -> Self:
        """Read the canonical run manifest identified by `run_id`."""
        validated_run_id = _validate_run_id(run_id)
        path = (
            repository
            / RUNS_DIRECTORY
            / f"{validated_run_id:04d}"
            / RUN_MANIFEST_FILENAME
        )
        return cls.from_run_manifest(path)

    @property
    def absolute_run_manifest_path(self) -> Path:
        """Return the referenced manifest path below its repository."""
        relative = PurePosixPath(self.run_manifest_path)
        return self.repository.joinpath(*relative.parts)

    @property
    def absolute_resolved_config_path(self) -> Path:
        """Return this run's archived resolved configuration path."""
        relative = PurePosixPath(self.resolved_config_path)
        return self.repository.joinpath(*relative.parts)


class RunConfigLog:
    """Map iterations to runs and summarize runs that produced no iteration."""

    def __init__(self, table: QTable | None = None) -> None:
        self.table = _empty_table() if table is None else table
        _validate_table(self.table)

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read and verify a previously written run/configuration log."""
        _validate_log_path(path)
        table = QTable.read(path, format="ascii.ecsv")
        _validate_table(table)
        _refresh_run_metadata(table, path.parent)
        return cls(table)

    @staticmethod
    def path_for(run_manifest_path: Path) -> Path:
        """Return the fixed log path for a validated run manifest."""
        run = RunManifestReference.from_run_manifest(run_manifest_path)
        return run.repository / RUN_CONFIG_LOG_FILENAME

    def write(self, path: Path) -> None:
        """Write only this log; checkpoints should use `ModelSearchState`."""
        _validate_log_path(path)
        _validate_table(self.table)
        _refresh_run_metadata(self.table, path.parent)
        _write_table_atomically(self.table, path)

    def append(self, iteration: int, run_id: int) -> Self:
        """Return a log linking one cumulative iteration to `run_id`."""
        row = {
            "iteration": iteration,
            "run_id": _validate_run_id(run_id),
        }
        if iteration < len(self.table):
            existing = {
                name: self.table[name][iteration].item()
                if hasattr(self.table[name][iteration], "item")
                else self.table[name][iteration]
                for name in _COLUMNS
            }
            if existing == row:
                return self
            raise ValueError(
                f"Iteration {iteration} is already linked to a different TNT run."
            )
        if iteration != len(self.table):
            raise ValueError(
                f"Expected iteration {len(self.table)}, received {iteration}."
            )
        table = self.table.copy()
        table.add_row(row)
        return type(self)(table)

    def truncate(self, n_iterations: int) -> Self:
        """Return a log containing only the first `n_iterations` rows."""
        if (
            isinstance(n_iterations, bool)
            or not isinstance(n_iterations, Integral)
            or not 0 <= n_iterations <= len(self.table)
        ):
            raise ValueError(
                "n_iterations must be an integer between 0 and the log length."
            )
        return type(self)(self.table[: int(n_iterations)].copy())

    def baseline_run_reference(
        self,
        repository: Path,
        *,
        current_run_id: int,
    ) -> RunManifestReference | None:
        """Return an earlier run that contributed the first search iteration."""
        if not len(self.table):
            return None
        run_id = _validate_run_id(self.table["run_id"][0])
        if run_id == _validate_run_id(current_run_id):
            return None
        return RunManifestReference.from_run_id(repository, run_id)

    def __len__(self) -> int:
        """Return the total number of iterations recorded so far."""
        return len(self.table)


def _empty_table() -> QTable:
    """Create the stable ECSV schema used before the first iteration."""
    return QTable(names=_COLUMNS, dtype=("i8", "i8"))


def _validate_log_path(path: Path) -> None:
    """Require the single repository filename defined for this log."""
    if path.name != RUN_CONFIG_LOG_FILENAME:
        raise ValueError(
            f"RunConfigLog must use the filename {RUN_CONFIG_LOG_FILENAME!r}."
        )


def _validate_table(table: QTable) -> None:
    """Validate the run-log schema, ordering, and run IDs."""
    if tuple(table.colnames) != _COLUMNS:
        raise ValueError(
            f"RunConfigLog columns must be {list(_COLUMNS)!r}; received "
            f"{table.colnames!r}."
        )
    iterations = [int(value) for value in table["iteration"]]
    if iterations != list(range(len(table))):
        raise ValueError(
            "RunConfigLog iterations must be unique, contiguous, and start at 0."
        )
    for value in table["run_id"]:
        _validate_run_id(value)


def _refresh_run_metadata(table: QTable, repository: Path) -> None:
    """Derive the human-readable run summary from immutable manifests."""
    runs_directory = repository / RUNS_DIRECTORY
    manifest_paths = (
        sorted(runs_directory.glob(f"*/{RUN_MANIFEST_FILENAME}"))
        if runs_directory.is_dir()
        else []
    )
    manifest_run_ids = [
        RunManifestReference.from_run_manifest(path).run_id for path in manifest_paths
    ]
    iteration_run_ids = {int(value) for value in table["run_id"]}
    missing_run_ids = iteration_run_ids.difference(manifest_run_ids)
    if missing_run_ids:
        RunManifestReference.from_run_id(repository, min(missing_run_ids))
    table.meta[TOTAL_RUNS_METADATA_KEY] = len(manifest_run_ids)
    table.meta[RUN_IDS_WITHOUT_ITERATIONS_METADATA_KEY] = [
        run_id for run_id in manifest_run_ids if run_id not in iteration_run_ids
    ]


def _validate_run_id(value: object) -> int:
    """Return one nonnegative run ID."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("run_id must be a nonnegative integer.")
    return int(value)


def _run_id(directory_name: str) -> int:
    """Read and validate the run ID encoded in a run-directory name."""
    if not directory_name.isdigit():
        raise ValueError(
            f"Run directory name has no numeric run ID: {directory_name!r}."
        )
    run_id = int(directory_name)
    expected = f"{run_id:04d}"
    if directory_name != expected:
        raise ValueError(
            f"Run directory name {directory_name!r} is not canonical; expected "
            f"{expected!r}."
        )
    return run_id


def _write_table_atomically(table: QTable, destination: Path) -> None:
    """Write an ECSV table via atomic replacement in its destination directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        table.write(temporary_path, format="ascii.ecsv", overwrite=True)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
