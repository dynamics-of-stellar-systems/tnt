"""Read YAML configuration files."""

from pathlib import Path
from typing import Any, NamedTuple

import astropy.units as au
import unxt as u
import yaml
from unxt import AbstractUnitSystem, Quantity

from tnt import mge
from tnt.mge import AbstractMGE


class UnitSystems(NamedTuple):
    """Internal and display unit systems defined by a configuration."""

    internal: AbstractUnitSystem
    display: AbstractUnitSystem


def read_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        The parsed configuration as a dictionary. An empty file yields an
        empty dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If the YAML document's root is not a mapping.
    """
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError(
            f"Configuration file {config_path} must contain a YAML mapping."
        )

    return data


def build_unit_systems(config: dict[str, Any]) -> UnitSystems:
    """Build the internal and display unit systems from a configuration.

    The display unit system inherits all units from the internal unit
    system, then overrides them with any units given under
    ``units: display:``.

    Args:
        config: A configuration dictionary, as returned by `read_config`,
            containing a ``units: internal:`` mapping of dimension names to
            unit strings and, optionally, a ``units: display:`` mapping of
            overrides.

    Returns:
        A `UnitSystems` named tuple of the internal and display unit
        systems.

    Raises:
        KeyError: If the configuration has no ``units: internal:`` mapping.
    """
    units_config = config["units"]
    internal_units = units_config["internal"]
    display_overrides = units_config.get("display", {})

    internal = u.unitsystem(*internal_units.values())
    display = u.unitsystem(internal, *display_overrides.values())

    return UnitSystems(internal=internal, display=display)


def build_distance(config: dict[str, Any]) -> Quantity:
    """Build the distance to the galaxy from a configuration.

    Args:
        config: A configuration dictionary, as returned by `read_config`,
            containing a ``distance:`` value/unit string (e.g. ``"30.5
            Mpc"``).

    Returns:
        The distance as a `unxt.Quantity`.

    Raises:
        KeyError: If the configuration has no ``distance:`` entry.
    """
    return Quantity.from_(au.Quantity(config["distance"]))


def build_mges(
    config: dict[str, Any],
    unit_system: AbstractUnitSystem,
    base_dir: str | Path = ".",
) -> dict[str, AbstractMGE]:
    """Build the named MGEs from a configuration.

    Each MGE's kind (light or mass) is inferred from its file's declared
    units -- see `tnt.mge.read_mge`.

    Args:
        config: A configuration dictionary, as returned by `read_config`,
            containing an optional ``mges:`` mapping of unique identifiers
            to ECSV file paths.
        unit_system: The unit system to convert each MGE's columns into.
        base_dir: Directory that relative ECSV paths are resolved against
            (typically the config file's own directory).

    Returns:
        A dict mapping each identifier to its `LightMGE` or `MassMGE`.
    """
    base_dir = Path(base_dir)
    return {
        identifier: mge.read_mge(base_dir / path, unit_system)
        for identifier, path in config.get("mges", {}).items()
    }