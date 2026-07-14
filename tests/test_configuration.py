import logging
from pathlib import Path

import pytest
import yaml

from tnt.configuration import Configuration, configuration_session


def _write_user_config(
    path: Path,
    output_directory: Path,
    body: str = "",
    orbit_body: str = "",
) -> None:
    path.write_text(
        f"""
system_attributes:
  distance_mpc: 10.0
  name: test_system
system_components:
  stars:
    type: triaxial_visible_component
    parameters:
      q:
        value: 0.7
    mge:
      potential_file: potential.ecsv
      luminosity_file: luminosity.ecsv
    kinematics:
      observed:
        type: gauss_hermite
        data_file: observed.ecsv
        aperture_file: aperture.dat
        bin_file: bins.dat
{body}
system_parameters:
  ml:
    value: 5.0
orbit_library_settings:
  logrmin: -0.2
  logrmax: 2.0
{orbit_body}
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

    package_logger = logging.getLogger("tnt")
    logger_state = (
        list(package_logger.handlers),
        package_logger.level,
        package_logger.propagate,
    )

    config = Configuration().read(user_path)

    assert list(package_logger.handlers) == logger_state[0]
    assert package_logger.level == logger_state[1]
    assert package_logger.propagate is logger_state[2]

    expected_path = output_directory / "config_repository" / "resolved_config.yaml"
    assert config.resolved_path == expected_path
    assert expected_path.is_file()

    written = yaml.safe_load(expected_path.read_text(encoding="utf-8"))
    assert written == config.data
    assert "dynamic_object_defaults" not in written
    assert "kinematics_type_defaults" not in written
    assert written["cosmological_parameters"]["H0"] == 70.0
    assert written["logging_settings"] == {
        "file": {"enabled": True, "level": "DEBUG", "directory": "logs"},
        "console": {"enabled": True, "level": "INFO"},
    }
    assert written["io_settings"]["input_directory"] == str(
        (Path.cwd() / "input").resolve()
    )
    assert written["io_settings"]["output_directory"] == str(output_directory)
    assert written["system_components"]["stars"]["include"] is True
    parameter = written["system_components"]["stars"]["parameters"]["q"]
    assert parameter["fixed"] is False
    assert parameter["logarithmic"] is False
    kinematics = written["system_components"]["stars"]["kinematics"]["observed"]
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
    histogram = config.data["system_components"]["stars"]["kinematics"]["observed"][
        "histogram"
    ]

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


def test_read_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    user_path.write_text(
        "io_settings:\n  input_directory: one\n  input_directory: two\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate configuration key"):
        Configuration().read(user_path)


def test_read_rejects_unknown_nested_field(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""        data_flie: misspelled.ecsv
""",
    )

    with pytest.raises(
        ValueError,
        match=r"kinematics\.observed contains unknown field\(s\): data_flie",
    ):
        Configuration().read(user_path)

    assert not output_directory.exists()


def test_read_rejects_partial_explicit_histogram(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""        histogram:
          width: 1000.0
""",
    )

    with pytest.raises(
        ValueError,
        match="must define width, center, and bins together",
    ):
        Configuration().read(user_path)


def test_read_rejects_even_histogram_bin_count(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""        histogram:
          width: 1000.0
          center: 0.0
          bins: 100
""",
    )

    with pytest.raises(ValueError, match="bins must be a positive odd integer"):
        Configuration().read(user_path)


def test_read_rejects_orbit_grid_with_too_few_i2_values(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        orbit_body="  nI2: 3\n",
    )

    with pytest.raises(ValueError, match=r"orbit_library_settings\.nI2"):
        Configuration().read(user_path)


def test_read_rejects_invalid_tagged_threshold_mode(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""parameter_space_settings:
  generator_settings:
    delta_chi2_threshold:
      mode: unsupported
""",
    )

    with pytest.raises(
        ValueError,
        match=r"delta_chi2_threshold\.mode must be one of",
    ):
        Configuration().read(user_path)


def test_read_rejects_nonpositive_worker_count(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""execution_settings:
  orbit_workers: 0
""",
    )

    with pytest.raises(
        ValueError,
        match=r"execution_settings\.orbit_workers must be a positive integer",
    ):
        Configuration().read(user_path)


@pytest.mark.parametrize(
    ("logging_body", "message"),
    [
        (
            """logging_settings:
  console:
    level: VERBOSE
""",
            r"logging_settings\.console\.level must be one of",
        ),
        (
            """logging_settings:
  file:
    directory: ../outside
""",
            r"logging_settings\.file\.directory must stay within",
        ),
    ],
)
def test_read_rejects_invalid_logging_settings(
    tmp_path: Path,
    logging_body: str,
    message: str,
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory, body=logging_body)

    with pytest.raises(ValueError, match=message):
        Configuration().read(user_path)


def test_configuration_session_logs_preparation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory)

    with configuration_session(user_path) as config:
        logging.getLogger("tnt.test").debug("execution debug detail")
        assert config.resolved_path is not None

    logfiles = list((output_directory / "logs").glob("tnt-*.log"))
    assert len(logfiles) == 1
    logfile = logfiles[0].read_text(encoding="utf-8")
    terminal = capsys.readouterr().err

    assert f"User configuration loaded from {user_path}" in logfile
    assert f"Resolving configuration loaded from {user_path}" in logfile
    assert "Resolved configuration written to" in logfile
    assert "execution debug detail" in logfile
    assert "TNT configuration session completed" in logfile
    assert "User configuration loaded from" in terminal
    assert "execution debug detail" not in terminal


def test_configuration_session_logs_validation_failure(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""        data_flie: misspelled.ecsv
""",
    )

    with pytest.raises(ValueError, match="data_flie"):
        with configuration_session(user_path):
            pass

    logfiles = list((output_directory / "logs").glob("tnt-*.log"))
    assert len(logfiles) == 1
    logfile = logfiles[0].read_text(encoding="utf-8")
    assert "Configuration preparation failed" in logfile
    assert "data_flie" in logfile
    assert "Traceback" in logfile
    assert not (output_directory / "config_repository").exists()


def test_print_requires_read_configuration() -> None:
    with pytest.raises(RuntimeError, match="No configuration has been read"):
        Configuration().print()
