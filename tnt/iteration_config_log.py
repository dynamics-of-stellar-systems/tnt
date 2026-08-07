"""Persist the configuration snapshot used by every model-search iteration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Self

from astropy.table import QTable

from tnt.configuration import (
    CONFIGURATIONS_DIRECTORY,
    HASH_PREFIX_LENGTH,
    RESOLVED_CONFIG_FILENAME,
    _read_yaml_bytes_mapping,
    _semantic_configuration_sha256,
)

ITERATION_CONFIG_LOG_FILENAME = "iteration_config_log.ecsv"
_COLUMNS = (
    "iteration",
    "configuration_snapshot_id",
    "semantic_sha256",
    "resolved_config_path",
)
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
        _validate_reference_fields(
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


class IterationConfigLog:
    """One validated ECSV row per cumulative model-search iteration."""

    def __init__(self, table: QTable | None = None) -> None:
        self.table = _empty_table() if table is None else table
        _validate_table(self.table)

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read and verify a previously written iteration/configuration log."""
        table = QTable.read(path, format="ascii.ecsv")
        _validate_table(table, repository=path.parent)
        return cls(table)

    @staticmethod
    def path_for(resolved_config_path: Path) -> Path:
        """Return the fixed log path for a versioned resolved snapshot."""
        reference = ConfigurationSnapshotReference.from_resolved_config(
            resolved_config_path
        )
        return reference.repository / ITERATION_CONFIG_LOG_FILENAME

    def write(self, path: Path) -> None:
        """Atomically replace the persisted log after validating its snapshots."""
        if path.name != ITERATION_CONFIG_LOG_FILENAME:
            raise ValueError(
                f"IterationConfigLog must be written as "
                f"{ITERATION_CONFIG_LOG_FILENAME!r}."
            )
        _validate_table(self.table, repository=path.parent)
        _write_table_atomically(self.table, path)

    def append(
        self,
        iteration: int,
        snapshot: ConfigurationSnapshotReference,
    ) -> Self:
        """Return a log recording `snapshot` for one cumulative iteration."""
        row = {
            "iteration": iteration,
            "configuration_snapshot_id": snapshot.snapshot_id,
            "semantic_sha256": snapshot.semantic_sha256,
            "resolved_config_path": snapshot.resolved_config_path,
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
                f"Iteration {iteration} is already linked to a different "
                "configuration snapshot."
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
    ) -> list[ConfigurationSnapshotReference]:
        """Reconstruct portable snapshot references below `repository`."""
        return [
            ConfigurationSnapshotReference(
                repository=repository,
                snapshot_id=int(row["configuration_snapshot_id"]),
                semantic_sha256=str(row["semantic_sha256"]),
                resolved_config_path=str(row["resolved_config_path"]),
            )
            for row in self.table
        ]

    def __len__(self) -> int:
        """Return the total number of iterations recorded so far."""
        return len(self.table)


def _empty_table() -> QTable:
    """Create the stable ECSV schema used even before the first iteration."""
    return QTable(
        names=_COLUMNS,
        dtype=("i8", "i8", "U64", "U4096"),
    )


def _validate_table(table: QTable, repository: Path | None = None) -> None:
    """Validate log schema, ordering, references, and optional disk contents."""
    if tuple(table.colnames) != _COLUMNS:
        raise ValueError(
            f"IterationConfigLog columns must be {list(_COLUMNS)!r}; received "
            f"{table.colnames!r}."
        )
    iterations = [int(value) for value in table["iteration"]]
    if iterations != list(range(len(table))):
        raise ValueError(
            "IterationConfigLog iterations must be unique, contiguous, and start at 0."
        )
    for row in table:
        snapshot_id = int(row["configuration_snapshot_id"])
        semantic_sha256 = str(row["semantic_sha256"])
        resolved_config_path = str(row["resolved_config_path"])
        _validate_reference_fields(
            snapshot_id,
            semantic_sha256,
            resolved_config_path,
        )
        if repository is not None:
            reference = ConfigurationSnapshotReference.from_resolved_config(
                repository.joinpath(*PurePosixPath(resolved_config_path).parts)
            )
            if (
                reference.snapshot_id != snapshot_id
                or reference.semantic_sha256 != semantic_sha256
            ):
                raise ValueError(
                    f"Iteration {int(row['iteration'])} does not match its "
                    "resolved configuration snapshot."
                )


def _validate_reference_fields(
    snapshot_id: int,
    semantic_sha256: str,
    resolved_config_path: str,
) -> None:
    """Validate one portable snapshot reference stored in the log."""
    if (
        isinstance(snapshot_id, bool)
        or not isinstance(snapshot_id, Integral)
        or snapshot_id < 0
    ):
        raise ValueError("configuration_snapshot_id must be a nonnegative integer.")
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
