"""Runtime kinematics objects built from resolved TNT configuration data."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from tnt.kinematics.base import AbstractKinematics, Histogram, Histogram2D
from tnt.kinematics.bayes_losvd import BayesLOSVD
from tnt.kinematics.gauss_hermite import GaussHermite
from tnt.kinematics.proper_motions import ProperMotions
from tnt.kinematics.registry import (
    get_kinematics_class,
    kinematics_type_names,
)
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


def build_kinematics(
    kinematic_data: Mapping[str, ConfigMapping],
    input_directory: str | Path,
    spatial_binnings: Mapping[str, ProjectedBinning],
    mges: Mapping[str, LightMGE | MassMGE] | None = None,
) -> dict[str, AbstractKinematics]:
    """Build named kinematics objects from resolved configuration data.

    Each data set keeps the units its file declares; nothing is coerced into
    a shared unit system (see `tnt.units`' module docstring).

    Args:
        kinematic_data: Resolved ``kinematic_data`` registry.
        input_directory: Directory against which data filenames are resolved.
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
        cls = get_kinematics_class(kind)
        if cls is None:
            allowed = ", ".join(sorted(kinematics_type_names()))
            raise ValueError(
                f"Unsupported {path}.type {kind!r}; expected one of: {allowed}."
            )
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
        )
    return built
