"""Build the named inputs referenced by a full, realistic resolved configuration.

Combines `Configuration.read` with `tnt.mge.build_mges` and
`tnt.spatial_binnings.build_spatial_binnings`: unlike
test_configuration_session.py, which only checks that resolution succeeds,
this also reads the files each resolved configuration entry actually
references, using its own `io_settings.input_directory` and unit system.
"""

from pathlib import Path

from tnt import Configuration
from tnt.mge import LightMGE, build_mges
from tnt.spatial_binnings import ProjectedBinning, build_spatial_binnings
from tnt.units import resolve_system_distance


def test_build_named_inputs_from_resolved_configuration(
    example_configuration_path: Path,
    tmp_path: Path,
) -> None:
    config = Configuration().read(example_configuration_path, workspace_root=tmp_path)
    resolved = config.as_dict()
    input_directory = resolved["io_settings"]["input_directory"]
    unit_system = config.unit_systems.internal

    distance = resolve_system_distance(resolved["system_attributes"])
    mges = build_mges(resolved["MGEs"], input_directory, unit_system, distance)
    assert isinstance(mges["mge_lum"], LightMGE)

    binnings = build_spatial_binnings(
        resolved["spatial_binnings"],
        input_directory,
        unit_system,
        resolved["mge_settings"]["projected_mass_quad_order"],
    )
    binning = binnings["kinset1_binning"]
    assert isinstance(binning, ProjectedBinning)
    assert (binning.npix_x, binning.npix_y) == (58, 52)
