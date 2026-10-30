"""Parameter-generation results shared by generators and their consumers."""

from dataclasses import dataclass, field
from typing import Any

from unxt import Quantity

ParameterSet = dict[str, dict[str, Quantity]]
"""Complete parameter coordinates in their declared units, grouped by component."""


@dataclass(frozen=True)
class ParameterBatch:
    """Candidates and diagnostic data from a single generation call.

    Public generation returns finalized parameters and common proposal counts.
    Subclass generation supplies raw parameters and optional scientific data.
    No batch is cached by the generator between calls.
    """

    parameters: list[ParameterSet]
    diagnostics: dict[str, Any] = field(default_factory=dict)
