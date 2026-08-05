"""A table linking each search round to the archived config file used for it.

A config archive can change between separate `ModelIterator.run()` calls
(e.g. a user edits and resumes a paused search), but `AllModels` -- and
`Model.iteration` -- have no notion of *which* config file was in effect for
a given round. `IterationConfigLog` fills that gap: one row per iteration,
recording which `resolved_config_path` produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from astropy.table import QTable


class IterationConfigLog:
    """One row per search round, linking it to the config file used for it."""

    def __init__(self, table: QTable | None = None) -> None:
        self.table = QTable() if table is None else table

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read a previously written `IterationConfigLog` table."""
        return cls(QTable.read(path, format="ascii.ecsv"))

    def write(self, path: Path) -> None:
        """Write this table to `path`, alongside `AllModels` in a config archive."""
        self.table.write(path, format="ascii.ecsv", overwrite=True)

    def append(self, iteration: int, resolved_config_path: Path) -> Self:
        """Return a new `IterationConfigLog` recording one round's config path."""
        row = {
            "iteration": iteration,
            "resolved_config_path": str(resolved_config_path),
        }
        if not self.table.colnames:
            return type(self)(QTable(rows=[row]))
        table = self.table.copy()
        table.add_row(row)
        return type(self)(table)

    def __len__(self) -> int:
        """Total number of rounds recorded so far."""
        return len(self.table)
