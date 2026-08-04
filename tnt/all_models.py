"""The growing table of every evaluated `Model`.

Signature-only scaffold: every method raises `NotImplementedError`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from tnt.model import Model


class AllModels:
    """Every `Model` evaluated so far, backed by `io_settings.all_models_file`."""

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read a previously written `AllModels` table."""
        raise NotImplementedError

    def write(self, path: Path) -> None:
        """Write this table to `path`, e.g. `io_settings.all_models_file`."""
        raise NotImplementedError

    def append(self, model: Model) -> Self:
        """Return a new `AllModels` with `model` added."""
        raise NotImplementedError

    def best(self, which_chi2: str) -> Model:
        """The `Model` with the lowest value of the named chi2 metric.

        Args:
            which_chi2: One of the keys in each `Model.chi2`, e.g.
                `parameter_space_settings.which_chi2`.
        """
        raise NotImplementedError
