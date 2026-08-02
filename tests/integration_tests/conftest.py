"""Shared fixtures for the full-example-configuration integration tests."""

from pathlib import Path

import pytest
import yaml

CONFIGURATION_PATH = Path(__file__).with_name("configuration.yaml")
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def example_configuration_path(tmp_path: Path) -> Path:
    """A temp copy of the example configuration, with a real input_directory.

    io_settings.input_directory in the committed configuration.yaml is a
    placeholder that doesn't exist on disk. Tests that only resolve the
    configuration don't care, but tests that go on to read the files it
    references (e.g. building MGEs) need it to point somewhere real --
    so this always overrides it to `FIXTURES_DIR`, the directory the
    referenced MGE ships in.
    """
    raw_config = yaml.safe_load(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    raw_config["io_settings"]["input_directory"] = str(FIXTURES_DIR)
    path = tmp_path / "configuration.yaml"
    path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    return path
