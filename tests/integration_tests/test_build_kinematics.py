"""Build kinematics from the realistic resolved example configuration."""

from pathlib import Path

from tnt import Configuration
from tnt.kinematics import GaussHermite, build_kinematics
from tnt.mge import build_mges


def test_build_kinematics_from_resolved_configuration(
    example_configuration_path: Path,
    tmp_path: Path,
) -> None:
    config = Configuration().read(example_configuration_path, workspace_root=tmp_path)
    resolved = config.as_dict()
    input_directory = resolved["io_settings"]["input_directory"]
    unit_system = config.unit_systems.internal
    mges = build_mges(resolved["MGEs"], input_directory, unit_system)

    kinematics = build_kinematics(
        resolved["kinematic_data"],
        input_directory,
        unit_system,
        resolved["spatial_binnings"],
        mges,
    )

    observed = kinematics["kinset1"]
    assert isinstance(observed, GaussHermite)
    assert observed.mge is mges["mge_lum"]
    assert observed.binning is resolved["spatial_binnings"]["kinset1_binning"]
    assert observed.n_spatial_bins == 2
