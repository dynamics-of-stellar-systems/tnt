"""Runtime kinematics objects built from resolved TNT configuration data."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from unxt import AbstractUnitSystem

from tnt.kinematics.base import AbstractKinematics, Histogram, Histogram2D
from tnt.kinematics.bayes_losvd import BayesLOSVD
from tnt.kinematics.gauss_hermite import GaussHermite
from tnt.kinematics.proper_motions import ProperMotions
from tnt.mge import LightMGE, MassMGE
from tnt.spatial_binnings import ProjectedBinning
from tnt.validation import (
    ConfigMapping,
    _mapping,
    _required_string,
    _resolve_typed_reference,
    _string,
)

__all__ = [
    "AbstractKinematics",
    "BayesLOSVD",
    "GaussHermite",
    "Histogram",
    "Histogram2D",
    "ProperMotions",
    "build_kinematics",
]


def _kinematics_class_registry() -> dict[str, type[AbstractKinematics]]:
    """Derive the configured type registry from concrete subclasses."""
    registry: dict[str, type[AbstractKinematics]] = {}
    for cls in AbstractKinematics.__subclasses__():
        kind = getattr(cls, "_type", None)
        if not isinstance(kind, str) or not kind:
            raise TypeError(f"{cls.__name__}._type must be a non-empty string.")
        if kind in registry:
            raise ValueError(
                f"Duplicate kinematics type {kind!r} on "
                f"{registry[kind].__name__} and {cls.__name__}."
            )
        registry[kind] = cls
    return registry


_KINEMATICS_CLASSES = _kinematics_class_registry()


def build_kinematics(
    kinematic_data: Mapping[str, ConfigMapping],
    input_directory: str | Path,
    unit_system: AbstractUnitSystem,
    spatial_binnings: Mapping[str, ProjectedBinning],
    mges: Mapping[str, LightMGE | MassMGE] | None = None,
) -> dict[str, AbstractKinematics]:
    """Build named kinematics objects from resolved configuration data.

    Args:
        kinematic_data: Resolved ``kinematic_data`` registry.
        input_directory: Directory against which data filenames are resolved.
        unit_system: Internal unit system used by runtime arrays.
        spatial_binnings: Already-built named spatial-binning objects.
        mges: Already-built named MGEs. May be omitted when no data set refers
            to an MGE.

    Returns:
        A mapping from configured names to concrete kinematics objects.
    """
    directory = Path(input_directory)
    available_mges: Mapping[str, LightMGE | MassMGE] = {} if mges is None else mges
    built: dict[str, AbstractKinematics] = {}
    for name, settings_value in kinematic_data.items():
        path = f"kinematic_data.{name}"
        if not isinstance(name, str) or not name:
            raise ValueError("kinematic_data names must be non-empty strings.")
        settings = _mapping(settings_value, path)
        kind = _required_string(settings, "type", path)
        try:
            cls = _KINEMATICS_CLASSES[kind]
        except KeyError as error:
            allowed = ", ".join(sorted(_KINEMATICS_CLASSES))
            raise ValueError(
                f"Unsupported {path}.type {kind!r}; expected one of: {allowed}."
            ) from error
        filename = _required_string(settings, "data_file", path)
        binning_name = _required_string(settings, "binning", path)
        binning = _resolve_typed_reference(
            spatial_binnings,
            binning_name,
            f"{path}.binning",
            "spatial_binnings",
            ProjectedBinning,
        )
        mge = None
        if "mge" in settings:
            mge_name = _string(settings["mge"], f"{path}.mge")
            mge = _resolve_typed_reference(
                available_mges,
                mge_name,
                f"{path}.mge",
                "MGEs",
                (LightMGE, MassMGE),
            )
        data_file = Path(filename)
        if not data_file.is_absolute():
            data_file = directory / data_file
        built[name] = cls.from_config(
            name=name,
            settings=settings,
            data_file=data_file,
            binning=binning,
            mge=mge,
            unit_system=unit_system,
        )
    return built
