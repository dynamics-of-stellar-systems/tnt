from pathlib import Path

import pytest
import unxt as u

from tnt import config

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_read_config():
    config_path = FIXTURES_DIR / "config.yaml"

    assert config.read_config(config_path)["units"] == {
        "internal": {
            "length": "kpc",
            "time": "Myr",
            "mass": "Msun",
            "angle": "rad",
            "power": "Lsun",
        },
        "display": {
            "angle": "arcsec",
            "speed": "km / s",
        },
    }


def test_build_unit_systems():
    config_dict = config.read_config(FIXTURES_DIR / "config.yaml")

    unit_systems = config.build_unit_systems(config_dict)

    assert unit_systems.internal == u.unitsystem("kpc", "Myr", "Msun", "rad", "Lsun")
    assert unit_systems.display == u.unitsystem(
        unit_systems.internal, "arcsec", "km / s"
    )
    assert unit_systems.display.angle == u.unit("arcsec")
    assert unit_systems.display.speed == u.unit("km / s")
    assert unit_systems.display.length == u.unit("kpc")


def test_build_unit_systems_without_display_overrides():
    config_dict = {"units": {"internal": {"length": "kpc", "time": "Myr"}}}

    unit_systems = config.build_unit_systems(config_dict)

    assert unit_systems.internal == unit_systems.display


def test_build_distance():
    config_dict = config.read_config(FIXTURES_DIR / "config.yaml")

    distance = config.build_distance(config_dict)

    assert distance == u.Quantity(30.5, "Mpc")


def test_build_distance_missing():
    with pytest.raises(KeyError):
        config.build_distance({})


def test_build_unit_systems_missing_units():
    with pytest.raises(KeyError):
        config.build_unit_systems({})


def test_read_config_empty_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("")

    assert config.read_config(config_path) == {}


def test_read_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.read_config(tmp_path / "missing.yaml")


def test_read_config_non_mapping(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- 1\n- 2\n")

    with pytest.raises(TypeError):
        config.read_config(config_path)