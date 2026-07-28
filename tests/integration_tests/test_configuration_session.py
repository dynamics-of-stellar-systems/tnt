"""Resolve a full, realistic user configuration end-to-end.

Unlike the synthetic per-feature configurations in
tests/unit_tests/test_configuration.py, this exercises `configuration_session`
against a complete example profile (NGC6278) covering every top-level
section at once, through the same lifecycle a real TNT execution uses.
"""

from pathlib import Path

import tnt

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_configuration_session_resolves_full_example_configuration(
    example_configuration_path: Path,
    tmp_path: Path,
) -> None:
    with tnt.configuration_session(
        example_configuration_path, workspace_root=tmp_path
    ) as config:
        assert isinstance(config, tnt.Configuration)

        resolved = config.as_dict()
        assert resolved["system_attributes"]["name"] == "NGC6278"
        assert resolved["potential"]["stars"]["mge"] == "mge_lum"
        assert resolved["kinematic_data"]["kinset1"]["mge"] == "mge_lum"
        assert Path(resolved["io_settings"]["input_directory"]) == FIXTURES_DIR

        portable = config.as_portable_dict()
        assert portable["io_settings"]["output_directory"] == "NGC6278_output"

    assert config.resolved_path is not None
    assert config.resolved_path.is_file()
    assert config.user_config_path.is_file()
    assert config.run_manifest_path.is_file()
