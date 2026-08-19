"""Persist iteration-to-run provenance and derived run summary metadata."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Self

from astropy.table import QTable

from tnt.config_parsing import _mapping, _required
from tnt.configuration import (
    CONFIGURATIONS_DIRECTORY,
    HASH_PREFIX_LENGTH,
    MANIFESTS_DIRECTORY,
    RESOLVED_CONFIG_FILENAME,
    _read_yaml_bytes_mapping,
    _semantic_configuration_sha256,
)

RUN_CONFIG_LOG_FILENAME = "run_config_log.ecsv"
TOTAL_RUNS_METADATA_KEY = "total_runs"
RUN_IDS_WITHOUT_ITERATIONS_METADATA_KEY = "run_ids_without_iterations"
_RUN_MANIFEST_SUFFIX = "-run_manifest.yaml"
_COLUMNS = ("iteration", "run_id")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ConfigurationSnapshotReference:
    """Portable identity of one immutable resolved-configuration snapshot."""

    repository: Path
    snapshot_id: int
    semantic_sha256: str
    resolved_config_path: str

    def __post_init__(self) -> None:
        """Validate the reference without requiring its file to be accessible."""
        _validate_snapshot_fields(
            self.snapshot_id,
            self.semantic_sha256,
            self.resolved_config_path,
        )

    @classmethod
    def from_resolved_config(cls, path: Path) -> Self:
        """Build and verify a reference from a versioned resolved snapshot."""
        resolved_path = path.expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Resolved configuration snapshot does not exist: {resolved_path}"
            )
        snapshot_directory = resolved_path.parent
        configurations_directory = snapshot_directory.parent
        if configurations_directory.name != CONFIGURATIONS_DIRECTORY:
            raise ValueError(
                f"Resolved configuration {resolved_path} is not below a "
                f"{CONFIGURATIONS_DIRECTORY!r} directory."
            )
        repository = configurations_directory.parent
        snapshot_id = _snapshot_id(snapshot_directory.name)
        archived = _read_yaml_bytes_mapping(
            resolved_path.read_bytes(),
            f"resolved configuration snapshot {resolved_path}",
        )
        semantic_sha256 = _semantic_configuration_sha256(archived)
        expected_directory = _snapshot_directory_name(snapshot_id, semantic_sha256)
        if snapshot_directory.name != expected_directory:
            raise ValueError(
                f"Resolved configuration directory {snapshot_directory.name!r} "
                f"does not match its snapshot ID and semantic hash; expected "
                f"{expected_directory!r}."
            )
        return cls(
            repository=repository,
            snapshot_id=snapshot_id,
            semantic_sha256=semantic_sha256,
            resolved_config_path=resolved_path.relative_to(repository).as_posix(),
        )

    @property
    def absolute_resolved_config_path(self) -> Path:
        """Return the referenced path resolved against its repository."""
        relative = PurePosixPath(self.resolved_config_path)
        return self.repository.joinpath(*relative.parts)


@dataclass(frozen=True)
class RunManifestReference:
    """Validated identity and configuration snapshot of one TNT run."""

    repository: Path
    run_id: int
    run_manifest_path: str
    configuration_snapshot: ConfigurationSnapshotReference

    @classmethod
    def from_run_manifest(cls, path: Path) -> Self:
        """Read and validate one immutable run manifest and its snapshot."""
        manifest_path = path.expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Run manifest does not exist: {manifest_path}")
        if manifest_path.parent.name != MANIFESTS_DIRECTORY:
            raise ValueError(
                f"Run manifest {manifest_path} is not below a "
                f"{MANIFESTS_DIRECTORY!r} directory."
            )
        repository = manifest_path.parent.parent
        run_id = _run_id(manifest_path.name)
        manifest = _read_yaml_bytes_mapping(
            manifest_path.read_bytes(), f"run manifest {manifest_path}"
        )
        recorded_run_id = _validate_run_id(_required(manifest, "run_id", "manifest"))
        if recorded_run_id != run_id:
            raise ValueError(
                f"Run manifest {manifest_path.name!r} records run_id "
                f"{recorded_run_id}, expected {run_id}."
            )

        configuration = _mapping(
            _required(manifest, "configuration", "manifest"),
            "manifest.configuration",
        )
        recorded_manifest_path = _required_string(
            configuration, "manifest", "manifest.configuration"
        )
        expected_manifest_path = manifest_path.relative_to(repository).as_posix()
        if recorded_manifest_path != expected_manifest_path:
            raise ValueError(
                f"Run manifest {manifest_path.name!r} records manifest path "
                f"{recorded_manifest_path!r}, expected {expected_manifest_path!r}."
            )

        resolved_config_path = _required_string(
            configuration, "resolved", "manifest.configuration"
        )
        recorded_snapshot_id = _validate_snapshot_id(
            _required(configuration, "snapshot_id", "manifest.configuration")
        )
        recorded_semantic_sha256 = _required_sha256(
            configuration, "semantic_sha256", "manifest.configuration"
        )
        _validate_snapshot_fields(
            recorded_snapshot_id,
            recorded_semantic_sha256,
            resolved_config_path,
        )
        snapshot = ConfigurationSnapshotReference.from_resolved_config(
            repository.joinpath(*PurePosixPath(resolved_config_path).parts)
        )
        if (
            snapshot.snapshot_id != recorded_snapshot_id
            or snapshot.semantic_sha256 != recorded_semantic_sha256
            or snapshot.resolved_config_path != resolved_config_path
        ):
            raise ValueError(
                f"Run manifest {manifest_path.name!r} does not match its resolved "
                "configuration snapshot."
            )

        recorded_resolved_sha256 = _required_sha256(
            configuration, "resolved_config_sha256", "manifest.configuration"
        )
        actual_resolved_sha256 = sha256(
            snapshot.absolute_resolved_config_path.read_bytes()
        ).hexdigest()
        if recorded_resolved_sha256 != actual_resolved_sha256:
            raise ValueError(
                f"Run manifest {manifest_path.name!r} does not match the exact "
                "resolved configuration file."
            )
        return cls(
            repository=repository,
            run_id=run_id,
            run_manifest_path=expected_manifest_path,
            configuration_snapshot=snapshot,
        )

    @classmethod
    def from_run_id(cls, repository: Path, run_id: int) -> Self:
        """Read the canonical run manifest identified by `run_id`."""
        validated_run_id = _validate_run_id(run_id)
        path = (
            repository
            / MANIFESTS_DIRECTORY
            / f"{validated_run_id:04d}{_RUN_MANIFEST_SUFFIX}"
        )
        return cls.from_run_manifest(path)

    @property
    def absolute_run_manifest_path(self) -> Path:
        """Return the referenced manifest path below its repository."""
        relative = PurePosixPath(self.run_manifest_path)
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
        _validate_table(table, repository=path.parent)
        _refresh_run_metadata(table, path.parent)
        return cls(table)

    @staticmethod
    def path_for(run_manifest_path: Path) -> Path:
        """Return the fixed log path for a validated run manifest."""
        run = RunManifestReference.from_run_manifest(run_manifest_path)
        return run.repository / RUN_CONFIG_LOG_FILENAME

    def write(self, path: Path) -> None:
        """Refresh run metadata and atomically replace the persisted log."""
        _validate_log_path(path)
        _validate_table(self.table, repository=path.parent)
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

    def snapshot_references(
        self,
        repository: Path,
        *,
        current_run_id: int,
        current_snapshot: ConfigurationSnapshotReference,
    ) -> list[ConfigurationSnapshotReference]:
        """Resolve each recorded run through its immutable manifest."""
        snapshots_by_run: dict[int, ConfigurationSnapshotReference] = {
            _validate_run_id(current_run_id): current_snapshot
        }
        references: list[ConfigurationSnapshotReference] = []
        for value in self.table["run_id"]:
            run_id = int(value)
            if run_id not in snapshots_by_run:
                snapshots_by_run[run_id] = RunManifestReference.from_run_id(
                    repository, run_id
                ).configuration_snapshot
            references.append(snapshots_by_run[run_id])
        return references

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


def _validate_table(table: QTable, repository: Path | None = None) -> None:
    """Validate log schema, ordering, run IDs, and optional manifests."""
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
    run_ids = [_validate_run_id(value) for value in table["run_id"]]
    if repository is not None:
        for run_id in set(run_ids):
            RunManifestReference.from_run_id(repository, run_id)


def _refresh_run_metadata(table: QTable, repository: Path) -> None:
    """Derive the human-readable run summary from immutable manifests."""
    manifests_directory = repository / MANIFESTS_DIRECTORY
    manifest_paths = (
        sorted(manifests_directory.glob(f"*{_RUN_MANIFEST_SUFFIX}"))
        if manifests_directory.is_dir()
        else []
    )
    manifest_run_ids = [
        RunManifestReference.from_run_manifest(path).run_id for path in manifest_paths
    ]
    iteration_run_ids = {int(value) for value in table["run_id"]}
    table.meta[TOTAL_RUNS_METADATA_KEY] = len(manifest_run_ids)
    table.meta[RUN_IDS_WITHOUT_ITERATIONS_METADATA_KEY] = [
        run_id for run_id in manifest_run_ids if run_id not in iteration_run_ids
    ]


def _validate_run_id(value: object) -> int:
    """Return one nonnegative run ID."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("run_id must be a nonnegative integer.")
    return int(value)


def _validate_snapshot_id(value: object) -> int:
    """Return one nonnegative configuration snapshot ID."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("configuration snapshot_id must be a nonnegative integer.")
    return int(value)


def _required_string(mapping: object, key: str, path: str) -> str:
    """Return one required nonempty string from a manifest mapping."""
    settings = _mapping(mapping, path)
    value = _required(settings, key, path)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}.{key} must be a nonempty string.")
    return value


def _required_sha256(mapping: object, key: str, path: str) -> str:
    """Return one required lowercase SHA-256 value from a manifest mapping."""
    value = _required_string(mapping, key, path)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{path}.{key} must be a lowercase SHA-256 hash.")
    return value


def _validate_snapshot_fields(
    snapshot_id: int,
    semantic_sha256: str,
    resolved_config_path: str,
) -> None:
    """Validate one portable configuration snapshot reference."""
    _validate_snapshot_id(snapshot_id)
    if _SHA256_PATTERN.fullmatch(semantic_sha256) is None:
        raise ValueError(
            "semantic_sha256 must be a lowercase SHA-256 hexadecimal hash."
        )
    relative = PurePosixPath(resolved_config_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("resolved_config_path must be relative to config_repository.")
    expected = PurePosixPath(
        CONFIGURATIONS_DIRECTORY,
        _snapshot_directory_name(snapshot_id, semantic_sha256),
        RESOLVED_CONFIG_FILENAME,
    )
    if relative != expected:
        raise ValueError(
            f"resolved_config_path {resolved_config_path!r} does not match "
            f"snapshot {snapshot_id}; expected {expected.as_posix()!r}."
        )


def _snapshot_id(directory_name: str) -> int:
    """Read a nonnegative snapshot ID from its versioned directory name."""
    prefix = directory_name.split("-", maxsplit=1)[0]
    if not prefix.isdigit():
        raise ValueError(
            f"Configuration snapshot directory {directory_name!r} has no "
            "numeric prefix."
        )
    return int(prefix)


def _snapshot_directory_name(snapshot_id: int, semantic_sha256: str) -> str:
    """Return the canonical directory name for a semantic snapshot."""
    return f"{snapshot_id:04d}-{semantic_sha256[:HASH_PREFIX_LENGTH]}"


def _run_id(filename: str) -> int:
    """Read and validate the run ID encoded in a manifest filename."""
    if not filename.endswith(_RUN_MANIFEST_SUFFIX):
        raise ValueError(f"Run manifest filename is invalid: {filename!r}.")
    prefix = filename[: -len(_RUN_MANIFEST_SUFFIX)]
    if not prefix.isdigit():
        raise ValueError(f"Run manifest filename has no numeric run ID: {filename!r}.")
    run_id = int(prefix)
    expected = f"{run_id:04d}{_RUN_MANIFEST_SUFFIX}"
    if filename != expected:
        raise ValueError(
            f"Run manifest filename {filename!r} is not canonical; expected "
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
