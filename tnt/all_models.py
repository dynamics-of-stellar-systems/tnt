"""The growing table of every evaluated `Model`.

Backed by an astropy `QTable`: one row per evaluated `Model`, with a
column per potential-component parameter (value + unit, e.g. `"bh.m"`,
`"stars.ml"` -- always present, since a proposed point's parameters are
known before evaluation), boolean `orblib_done`/`weights_done` flags (see
`Model`'s docstring), and one column per `Model.chi2` key (e.g. `"chi2"`,
`"kinchi2"`) once at least one appended model has `weights_done`. A model
appended without `chi2` (evaluation didn't succeed) gets `nan` in every
existing chi2 column; a model appended with `chi2` before any chi2 column
exists yet adds those columns, backfilling every earlier row with `nan`.

`read`/`write` round-trip through ``ascii.ecsv``, which preserves each
column's unit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import astropy.units as au
import numpy as np
from astropy.table import QTable, Row
from unxt import Quantity

from tnt.model import Model


def _to_astropy_quantity(value: Quantity) -> au.Quantity:
    """Convert one `unxt.Quantity` potential parameter to an astropy `Quantity`."""
    return float(value.ustrip(value.unit)) * au.Unit(str(value.unit))


def _model_row(model: Model) -> dict[str, Any]:
    """Flatten one `Model` into a table row: parameters, flags, and chi2."""
    row: dict[str, Any] = {
        f"{component_name}.{parameter_name}": _to_astropy_quantity(value)
        for component_name, component in model.potential.components.items()
        for parameter_name, value in component.parameters.items()
    }
    row["iteration"] = model.iteration
    row["orblib_done"] = model.orblib_done
    row["weights_done"] = model.weights_done
    if model.chi2 is not None:
        row.update(model.chi2)
    return row


def _add_row(table: QTable, row: dict[str, Any]) -> QTable:
    """Add `row` to `table`, reconciling either side's missing chi2 columns."""
    for key in row:
        if key not in table.colnames:
            table[key] = np.full(len(table), np.nan)
    for key in table.colnames:
        if key not in row:
            row[key] = np.nan
    table.add_row(row)
    return table


class AllModels:
    """Every `Model` evaluated so far, backed by `io_settings.all_models_file`."""

    def __init__(self, table: QTable | None = None) -> None:
        self.table = QTable() if table is None else table

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read a previously written `AllModels` table."""
        return cls(QTable.read(path, format="ascii.ecsv"))

    def write(self, path: Path) -> None:
        """Write only this table; checkpoints should use `ModelSearchState`."""
        self.table.write(path, format="ascii.ecsv", overwrite=True)

    def append(self, model: Model) -> Self:
        """Return a new `AllModels` with `model` added."""
        row = _model_row(model)
        if not self.table.colnames:
            # `QTable(rows=[row])` (rather than `_add_row`'s reconciliation,
            # meant for chi2 columns appearing on a later row) is what
            # infers each column's real dtype/unit from `row`'s values.
            return type(self)(QTable(rows=[row]))
        return type(self)(_add_row(self.table.copy(), row))

    def __len__(self) -> int:
        """Total number of models evaluated so far."""
        return len(self.table)

    def n_iterations(self) -> int:
        """Number of `ModelIterator.run` search rounds reflected here.

        0 if empty, otherwise one more than the largest `Model.iteration`
        among the models held here. Lets `ModelIterator.run` assign cumulative
        iteration labels after resuming a previously written `AllModels`, and
        measure its per-call `stopping_criteria.n_new_iter` allowance from the
        resumed starting point, without a separately persisted counter.
        """
        if not len(self.table):
            return 0
        return int(self.table["iteration"].max()) + 1

    def has_successful_model(self) -> bool:
        """Whether at least one model completed orbit-weight solving."""
        if not len(self.table):
            return False
        return bool(np.any(self.table["weights_done"]))

    def best(self, which_chi2: str) -> Row:
        """The table row with the lowest value of the named chi2 column.

        Rows without a `weights_done` evaluation have `nan` in every chi2
        column, and are ignored here.

        Args:
            which_chi2: One of `Model.chi2`'s keys, e.g.
                `parameter_space_settings.which_chi2`.
        """
        if not len(self.table):
            raise ValueError("AllModels.best: no models have been evaluated yet.")
        if which_chi2 not in self.table.colnames:
            raise ValueError(f"AllModels.best: no model has a computed {which_chi2!r}.")
        try:
            index = int(np.nanargmin(self.table[which_chi2]))
        except ValueError as error:
            raise ValueError(
                f"AllModels.best: no model has a computed {which_chi2!r}."
            ) from error
        return self.table[index]
