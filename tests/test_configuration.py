from pathlib import Path

import pytest
import yaml

from tnt.configuration import Configuration


def _write_user_config(path: Path, output_directory: Path, body: str = "") -> None:
    path.write_text(
        f"""
system_components:
  stars:
    type: triaxial_visible_component
    parameters:
      q:
        value: 0.7
    kinematics:
      observed:
        type: gauss_hermite
        data_file: observed.ecsv
{body}
system_parameters:
  ml:
    value: 5.0
orbit_library_settings:
  logrmin: -0.2
  logrmax: 2.0
io_settings:
  input_directory: input
  output_directory: {output_directory.as_posix()}
""".lstrip(),
        encoding="utf-8",
    )


def test_read_resolves_defaults_and_writes_snapshot(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""        with_pops: true
weight_solver_settings:
  counter_rotating_orbit_cut:
    enabled: true
""",
    )

    config = Configuration().read(user_path)

    expected_path = (
        output_directory / "config_repository" / "resolved_config.yaml"
    )
    assert config.resolved_path == expected_path
    assert expected_path.is_file()

    written = yaml.safe_load(expected_path.read_text(encoding="utf-8"))
    assert written == config.data
    assert "dynamic_object_defaults" not in written
    assert "kinematics_type_defaults" not in written
    assert written["cosmological_parameters"]["H0"] == 70.0
    assert written["io_settings"]["input_directory"] == str(
        (Path.cwd() / "input").resolve()
    )
    assert written["io_settings"]["output_directory"] == str(output_directory)
    assert written["system_components"]["stars"]["include"] is True
    parameter = written["system_components"]["stars"]["parameters"]["q"]
    assert parameter["fixed"] is False
    assert parameter["logarithmic"] is False
    kinematics = written["system_components"]["stars"]["kinematics"][
        "observed"
    ]
    assert kinematics["with_pops"] is True
    assert kinematics["histogram"]["sigma_extent"] == 3.0
    assert kinematics["histogram"]["bin_width_sigma_fraction"] == 0.1
    cut = written["weight_solver_settings"]["counter_rotating_orbit_cut"]
    assert cut["enabled"] is True
    assert cut["min_affected_apertures"] == 2


def test_explicit_histogram_replaces_derived_policy(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""        histogram:
          width: 1000.0
          center: 10.0
          bins: 101
""",
    )

    config = Configuration().read(user_path)
    histogram = config.data["system_components"]["stars"]["kinematics"][
        "observed"
    ]["histogram"]

    assert histogram == {"width": 1000.0, "center": 10.0, "bins": 101}


def test_read_requires_output_directory(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    user_path.write_text(
        "io_settings:\n  input_directory: input\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"io_settings\.output_directory must be a non-empty string",
    ):
        Configuration().read(user_path)


def test_print_requires_read_configuration() -> None:
    with pytest.raises(RuntimeError, match="No configuration has been read"):
        Configuration().print()
