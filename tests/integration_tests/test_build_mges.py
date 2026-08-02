"""Build the MGEs referenced by a full, realistic resolved configuration.

Combines `Configuration.read` with `tnt.mge.build_mges`: unlike
test_configuration_session.py, which only checks that resolution succeeds,
this also reads the MGE file(s) the resolved configuration actually
references, using its own `io_settings.input_directory` and unit system.
"""

from pathlib import Path

from tnt import Configuration
from tnt.mge import LightMGE, build_mges


def test_build_mges_from_resolved_configuration(
    example_configuration_path: Path,
    tmp_path: Path,
) -> None:
    config = Configuration().read(example_configuration_path, workspace_root=tmp_path)
    resolved = config.as_dict()

    mges = build_mges(
        resolved["MGEs"],
        resolved["io_settings"]["input_directory"],
        config.unit_systems.internal,
    )

    assert isinstance(mges["mge_lum"], LightMGE)
