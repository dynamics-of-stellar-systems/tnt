"""Resolve a full, realistic user configuration end-to-end.

Unlike the synthetic per-feature configurations in
tests/unit_tests/test_configuration.py, this exercises `configuration_session`
against a complete example profile (NGC6278) covering every top-level
section at once, through the same lifecycle a real TNT execution uses.
"""

from pathlib import Path

import yaml

import tnt

CONFIGURATION_PATH = Path(__file__).with_name("configuration.yaml")
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_configuration_session_resolves_full_example_configuration(
    tmp_path: Path,
) -> None:
    # Point input_directory at the real fixtures directory (output_directory
    # stays under tmp_path, so nothing is written into the repo) by writing a
    # copy of the example configuration with that one field overridden --
    # nothing in this module actually reads from input_directory yet, but a
    # real, existing path here is still more honest than a placeholder.
    raw_config = yaml.safe_load(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    raw_config["io_settings"]["input_directory"] = str(FIXTURES_DIR)
    user_config_path = tmp_path / "configuration.yaml"
    user_config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    with tnt.configuration_session(
        user_config_path, workspace_root=tmp_path
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
