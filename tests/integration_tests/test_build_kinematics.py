"""Build kinematics from the realistic resolved example configuration."""

from pathlib import Path

from tnt import Configuration
from tnt.kinematics import GaussHermite, build_kinematics
from tnt.mge import build_mges
from tnt.spatial_binnings import build_spatial_binnings
from tnt.units import resolve_system_distance


def test_build_kinematics_from_resolved_configuration(
    example_configuration_path: Path,
    tmp_path: Path,
) -> None:
    config = Configuration().read(example_configuration_path, workspace_root=tmp_path)
    resolved = config.as_dict()
    input_directory = resolved["io_settings"]["input_directory"]
    unit_system = config.unit_systems.internal
    distance = resolve_system_distance(resolved["system_attributes"])
    mges = build_mges(resolved["MGEs"], input_directory, unit_system, distance)
    binnings = build_spatial_binnings(
        resolved["spatial_binnings"],
        input_directory,
        unit_system,
        resolved["mge_settings"]["projected_mass_quad_order"],
    )

    kinematics = build_kinematics(
        resolved["kinematic_data"],
        input_directory,
        unit_system,
        binnings,
        mges,
    )

    observed = kinematics["kinset1"]
    assert isinstance(observed, GaussHermite)
    assert observed.mge is mges["mge_lum"]
    assert observed.binning is binnings["kinset1_binning"]
    assert observed.n_spatial_bins == binnings["kinset1_binning"].n_bins == 152
