"""Locations for generator-owned diagnostics within a run's output folder."""

from pathlib import Path


def diagnostic_path(directory: Path, name: str, iteration: int | None = None) -> Path:
    """Create the output folder and name an optional iteration-specific figure."""
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "" if iteration is None else f"_iteration_{iteration:04d}"
    return directory / f"{name}{suffix}.png"
