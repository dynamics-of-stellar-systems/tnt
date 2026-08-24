import logging
import sys
from pathlib import Path

import pytest
import yaml

from tnt.configuration import (
    Configuration,
    configuration_session,
)


def _write_user_config(
    path: Path,
    output_directory: Path,
    body: str = "",
    orbit_body: str = "",
    kinematics_type: str = "gauss_hermite",
) -> None:
    path.write_text(
        f"""
system_attributes:
  distance: {{value: 10.0, unit: "kpc"}}
  name: test_system
MGEs:
  light: luminosity.ecsv
spatial_binnings:
  observed:
    min_x: {{value: -29.5, unit: "arcsec"}}
    min_y: {{value: -26.5, unit: "arcsec"}}
    x_extent: {{value: 58.0, unit: "arcsec"}}
    y_extent: {{value: 52.0, unit: "arcsec"}}
    PA: {{value: 126.0, unit: "deg"}}
    bins_file: bins.npy
potential:
  stars:
    type: triaxial_light_mge
    mge: light
    parameters:
      q:
        value: 0.7
      ml:
        unit: "Msun / Lsun"
        value: 5.0
kinematic_data:
  observed:
    type: {kinematics_type}
    binning: observed
    mge: light
    data_file: observed.ecsv
{body}
orbit_library_settings:
  orbit_sampler:
    logrmin: -0.2
    logrmax: 2.0
{orbit_body}
io_settings:
  input_directory: input
  output_directory: {output_directory.as_posix()}
""".lstrip(),
        encoding="utf-8",
    )


def test_read_resolves_defaults_and_writes_run_bundle(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""weight_solver_settings:
  reattempt_failures: false
""",
    )

    package_logger = logging.getLogger("tnt")
    logger_state = (
        list(package_logger.handlers),
        package_logger.level,
        package_logger.propagate,
    )

    config = Configuration().read(user_path, workspace_root=tmp_path)

    assert list(package_logger.handlers) == logger_state[0]
    assert package_logger.level == logger_state[1]
    assert package_logger.propagate is logger_state[2]

    repository = output_directory / "config_repository"
    assert config.resolved_path is not None
    expected_path = config.resolved_path
    assert expected_path.parent.parent == repository / "runs"
    assert expected_path.parent.name == "0000"
    assert expected_path.name == "resolved_config.yaml"
    assert config.resolved_path == expected_path
    assert expected_path.is_file()

    written = yaml.safe_load(expected_path.read_text(encoding="utf-8"))
    assert written == config.portable_data
    assert config.workspace_root == tmp_path
    assert "dynamic_object_defaults" not in written
    assert "kinematics_type_defaults" not in written
    assert written["cosmological_parameters"]["H0"] == {
        "value": 70.0,
        "unit": "km / (s Mpc)",
    }
    assert written["units"] == {
        "internal": {
            "length": "kpc",
            "time": "Myr",
            "mass": "Msun",
            "angle": "rad",
            "power": "Lsun",
        },
        "display": {"angle": "arcsec", "speed": "km / s"},
    }
    assert config.unit_systems is not None
    assert written["logging_settings"] == {
        "file": {"enabled": True, "level": "DEBUG", "directory": "logs"},
        "console": {"enabled": True, "level": "INFO"},
    }
    assert written["io_settings"]["input_directory"] == "input"
    assert written["io_settings"]["output_directory"] == "output"
    assert config.data["io_settings"]["input_directory"] == str(tmp_path / "input")
    assert config.data["io_settings"]["output_directory"] == str(output_directory)
    assert written["potential"]["stars"]["include"] is True
    parameter = written["potential"]["stars"]["parameters"]["q"]
    assert parameter["fixed"] is False
    kinematics = written["kinematic_data"]["observed"]
    assert kinematics["binning"] == "observed"
    assert kinematics["mge"] == "light"
    assert kinematics["maximum_gh_order"] == 4
    assert kinematics["observational_errors"]["systematic_uncertainties"] == {
        "v": {"value": 0.0, "unit": "km / s"},
        "sigma": {"value": 0.0, "unit": "km / s"},
        "h3": 0.0,
        "h4": 0.0,
    }
    assert kinematics["histogram"]["sigma_extent"] == 3.0
    assert kinematics["histogram"]["bin_width_sigma_fraction"] == 0.1
    assert written["parameter_space_settings"]["potential_rescalings"] == {
        "enabled": False,
        "range_count": 10,
        "mass_scale_range": {"minimum": 0.1, "maximum": 10.0},
        "spacing": "logarithmic",
    }
    assert written["weight_solver_settings"]["reattempt_failures"] is False
    assert written["weight_solver_settings"]["maxiter_factor"] == 3

    assert config.source_path == user_path

    assert config.run_manifest_path is not None
    manifest_path = config.run_manifest_path
    assert manifest_path == repository / "runs" / "0000" / "run_manifest.yaml"
    assert config.run_manifest_path == manifest_path
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 3
    assert manifest["run_id"] == 0
    assert config.run_id == 0
    assert set(manifest["tnt"]) == {
        "version",
        "git_commit",
        "git_working_tree_dirty",
    }
    assert "unxt" in manifest["dependencies"]
    assert manifest["execution"]["workspace_root"] == str(tmp_path)
    assert set(manifest["configuration"]) == {
        "input_directory",
        "logfile",
        "output_directory",
        "resolved",
    }
    assert manifest["configuration"]["resolved"] == str(
        expected_path.relative_to(repository)
    )
    assert manifest["configuration"]["input_directory"] == str(tmp_path / "input")
    assert manifest["configuration"]["output_directory"] == str(output_directory)
    assert manifest["configuration"]["logfile"] is None
    assert manifest["randomness"] == {
        "configured_orbit_library_seed": -1,
        "effective_orbit_library_seed": None,
        "status": "pending_generation",
    }


def test_repository_archives_resolved_configuration_for_every_run(
    tmp_path: Path,
) -> None:
    first_user_path = tmp_path / "first.yaml"
    second_user_path = tmp_path / "second.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(first_user_path, output_directory)
    second_user_path.write_text(
        f"# Same configuration with different formatting.\n\n"
        f"{first_user_path.read_text(encoding='utf-8')}\n",
        encoding="utf-8",
    )

    first = Configuration().read(first_user_path, workspace_root=tmp_path)
    repeated = Configuration().read(first_user_path, workspace_root=tmp_path)
    reformatted = Configuration().read(second_user_path, workspace_root=tmp_path)

    resolved_paths = {
        first.resolved_path,
        repeated.resolved_path,
        reformatted.resolved_path,
    }
    assert len(resolved_paths) == 3
    repository = output_directory / "config_repository"
    run_directories = sorted((repository / "runs").iterdir())
    assert [path.name for path in run_directories] == [
        "0000",
        "0001",
        "0002",
    ]
    manifests = [path / "run_manifest.yaml" for path in run_directories]
    manifest_data = [
        yaml.safe_load(path.read_text(encoding="utf-8")) for path in manifests
    ]
    assert [manifest["run_id"] for manifest in manifest_data] == [0, 1, 2]
    resolved_data = [
        yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
        for path in run_directories
    ]
    assert resolved_data == [first.portable_data] * 3


def test_repository_preserves_resolved_configuration_for_each_run(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory)
    first = Configuration().read(user_path, workspace_root=tmp_path)

    user_path.write_text(
        user_path.read_text(encoding="utf-8").replace(
            "name: test_system",
            "name: changed_system",
        ),
        encoding="utf-8",
    )
    second = Configuration().read(user_path, workspace_root=tmp_path)

    assert first.resolved_path != second.resolved_path
    assert first.resolved_path is not None
    assert second.resolved_path is not None
    assert first.resolved_path.parent.name == "0000"
    assert second.resolved_path.parent.name == "0001"
    first_archived = yaml.safe_load(first.resolved_path.read_text(encoding="utf-8"))
    second_archived = yaml.safe_load(second.resolved_path.read_text(encoding="utf-8"))
    assert first_archived["system_attributes"]["name"] == "test_system"
    assert second_archived["system_attributes"]["name"] == "changed_system"


def test_repository_preserves_equivalent_unit_declarations_per_run(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory)
    first = Configuration().read(user_path, workspace_root=tmp_path)
    user_path.write_text(
        user_path.read_text(encoding="utf-8").replace(
            'distance: {value: 10.0, unit: "kpc"}',
            'distance: {value: 10000.0, unit: "pc"}',
        ),
        encoding="utf-8",
    )

    second = Configuration().read(user_path, workspace_root=tmp_path)

    assert first.resolved_path != second.resolved_path
    assert first.resolved_path is not None
    assert second.resolved_path is not None
    first_archived = yaml.safe_load(first.resolved_path.read_text(encoding="utf-8"))
    second_archived = yaml.safe_load(second.resolved_path.read_text(encoding="utf-8"))
    assert first_archived["system_attributes"]["distance"] == {
        "value": 10.0,
        "unit": "kpc",
    }
    assert second_archived["system_attributes"]["distance"] == {
        "value": 10000.0,
        "unit": "pc",
    }


def test_default_workspace_root_is_invoking_script_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_directory = tmp_path / "driver"
    script_directory.mkdir()
    script_path = script_directory / "run_model.py"
    script_path.write_text("# test entrypoint\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(script_path))

    user_path = tmp_path / "user.yaml"
    _write_user_config(user_path, Path("results"))

    config = Configuration().read(user_path)

    assert config.workspace_root == script_directory
    assert config.data["io_settings"]["input_directory"] == str(
        script_directory / "input"
    )
    assert config.data["io_settings"]["output_directory"] == str(
        script_directory / "results"
    )
    assert config.portable_data["io_settings"]["output_directory"] == "results"


def test_explicit_histogram_replaces_derived_policy(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""    histogram:
      width: {value: 1000.0, unit: "kpc / Myr"}
      center: {value: 10.0, unit: "kpc / Myr"}
      bins: 101
""",
    )

    config = Configuration().read(user_path, workspace_root=tmp_path)
    histogram = config.data["kinematic_data"]["observed"]["histogram"]

    assert histogram == {
        "width": {"value": 1000.0, "unit": "kpc / Myr"},
        "center": {"value": 10.0, "unit": "kpc / Myr"},
        "bins": 101,
    }


def test_read_preserves_explicit_quantity(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory)
    original = user_path.read_text(encoding="utf-8").replace(
        'distance: {value: 10.0, unit: "kpc"}',
        'distance: {value: 10.0, unit: "Mpc"}',
    )
    user_path.write_text(original, encoding="utf-8")

    config = Configuration().read(user_path, workspace_root=tmp_path)

    assert config.data["system_attributes"]["distance"] == {
        "value": 10.0,
        "unit": "Mpc",
    }
    assert config.resolved_path is not None
    resolved = yaml.safe_load(config.resolved_path.read_text(encoding="utf-8"))
    assert resolved["system_attributes"]["distance"] == {
        "value": 10.0,
        "unit": "Mpc",
    }


def test_read_rejects_incompatible_quantity_unit_before_writing(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory)
    invalid = user_path.read_text(encoding="utf-8").replace(
        'distance: {value: 10.0, unit: "kpc"}',
        'distance: {value: 10.0, unit: "Myr"}',
    )
    user_path.write_text(invalid, encoding="utf-8")

    with pytest.raises(ValueError, match=r"system_attributes\.distance\.unit"):
        Configuration().read(user_path, workspace_root=tmp_path)

    assert not output_directory.exists()


def test_read_rejects_unitful_bare_number_before_writing(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory)
    invalid = user_path.read_text(encoding="utf-8").replace(
        'distance: {value: 10.0, unit: "kpc"}',
        "distance: 10.0",
    )
    user_path.write_text(invalid, encoding="utf-8")

    with pytest.raises(
        TypeError,
        match=r"system_attributes\.distance.*must state their unit explicitly",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)

    assert not output_directory.exists()


def test_quantity_override_cannot_inherit_default_unit(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""cosmological_parameters:
  H0:
    value: 75.0
""",
    )

    with pytest.raises(
        ValueError,
        match=r"cosmological_parameters\.H0.*missing required field.*unit",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)

    assert not output_directory.exists()


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
        Configuration().read(user_path, workspace_root=tmp_path)


def test_read_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    user_path.write_text(
        "io_settings:\n  input_directory: one\n  input_directory: two\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate configuration key"):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_read_rejects_unknown_nested_field(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""    data_flie: misspelled.ecsv
""",
    )

    with pytest.raises(
        ValueError,
        match=r"kinematic_data\.observed contains unknown field\(s\): data_flie",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)

    assert not output_directory.exists()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            """parameter_space_settings:
  stopping_criteria:
    unexpected_option: 1
""",
            r"stopping_criteria contains unknown field\(s\): unexpected_option",
        ),
        (
            """execution_settings:
  unexpected_option: 1
""",
            r"execution_settings contains unknown field\(s\): unexpected_option",
        ),
        (
            """weight_solver_settings:
  unexpected_option: 1
""",
            r"weight_solver_settings contains unknown field\(s\): unexpected_option",
        ),
    ],
)
def test_read_rejects_unknown_preparation_field(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory, body=body)

    with pytest.raises(ValueError, match=message):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_read_rejects_partial_explicit_histogram(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""    histogram:
      width: {value: 1000.0, unit: "kpc / Myr"}
""",
    )

    with pytest.raises(
        ValueError,
        match="must define width, center, and bins together",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_read_defers_even_histogram_bin_count_to_runtime(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""    histogram:
      width: {value: 1000.0, unit: "kpc / Myr"}
      center: {value: 0.0, unit: "kpc / Myr"}
      bins: 100
""",
    )

    config = Configuration().read(user_path, workspace_root=tmp_path)

    assert config.data["kinematic_data"]["observed"]["histogram"]["bins"] == 100


def test_read_rejects_orbit_grid_with_too_few_i2_values(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        orbit_body="    nI2: 3\n",
    )

    with pytest.raises(
        ValueError, match=r"orbit_library_settings\.orbit_sampler\.nI2"
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


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
        Configuration().read(user_path, workspace_root=tmp_path)


def test_read_accepts_disabled_minimum_delta_chi2(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""parameter_space_settings:
  stopping_criteria:
    minimum_delta_chi2:
      enabled: false
""",
    )

    config = Configuration().read(user_path, workspace_root=tmp_path)

    assert config.data["parameter_space_settings"]["stopping_criteria"][
        "minimum_delta_chi2"
    ] == {"enabled": False, "mode": "absolute", "value": 0.5}


def test_read_rejects_nonpositive_target_model_count(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""parameter_space_settings:
  stopping_criteria:
    target_model_count: 0
""",
    )

    with pytest.raises(
        ValueError,
        match=r"stopping_criteria\.target_model_count must be positive",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_read_rejects_nonpositive_n_new_iter(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""parameter_space_settings:
  stopping_criteria:
    n_new_iter: 0
""",
    )

    with pytest.raises(
        ValueError,
        match=r"stopping_criteria\.n_new_iter must be positive",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_read_rejects_negative_minimum_delta_chi2(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""parameter_space_settings:
  stopping_criteria:
    minimum_delta_chi2:
      value: -0.5
""",
    )

    with pytest.raises(
        ValueError,
        match=r"minimum_delta_chi2\.value must not be negative",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_read_rejects_negative_generator_delta_chi2_threshold(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""parameter_space_settings:
  generator_settings:
    delta_chi2_threshold:
      value: -0.5
""",
    )

    with pytest.raises(
        ValueError,
        match=r"delta_chi2_threshold\.value must not be negative",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


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
        Configuration().read(user_path, workspace_root=tmp_path)


def test_read_rejects_enabled_reattempt_failures(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""weight_solver_settings:
  reattempt_failures: true
""",
    )

    with pytest.raises(
        ValueError,
        match=r"weight_solver_settings\.reattempt_failures must be false",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


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
        Configuration().read(user_path, workspace_root=tmp_path)


def test_gauss_hermite_sets_resolve_independent_orders_and_systematics(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""  secondary:
    type: gauss_hermite
    maximum_gh_order: 5
    observational_errors:
      systematic_uncertainties:
        v: {value: 1.0, unit: "kpc / Myr"}
        sigma: {value: 2.0, unit: "kpc / Myr"}
        h3: 0.03
        h4: 0.04
        h5: 0.05
    data_file: secondary.ecsv
    binning: observed
    mge: light
""",
    )

    config = Configuration().read(user_path, workspace_root=tmp_path)
    kinematics = config.data["kinematic_data"]

    assert kinematics["observed"]["maximum_gh_order"] == 4
    assert kinematics["secondary"]["maximum_gh_order"] == 5
    assert kinematics["secondary"]["observational_errors"][
        "systematic_uncertainties"
    ] == {
        "v": {"value": 1.0, "unit": "kpc / Myr"},
        "sigma": {"value": 2.0, "unit": "kpc / Myr"},
        "h3": 0.03,
        "h4": 0.04,
        "h5": 0.05,
    }


def test_read_defers_gauss_hermite_systematics_completeness_to_runtime(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""    maximum_gh_order: 5
""",
    )

    config = Configuration().read(user_path, workspace_root=tmp_path)

    systematics = config.data["kinematic_data"]["observed"][
        "observational_errors"
    ]["systematic_uncertainties"]
    assert "h5" not in systematics


def test_proper_motion_set_resolves_its_own_variance_scale(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""    observational_errors:
      variance_scale: 1.5
""",
        kinematics_type="proper_motions",
    )

    config = Configuration().read(user_path, workspace_root=tmp_path)
    kinematics = config.data["kinematic_data"]["observed"]

    assert "maximum_gh_order" not in kinematics
    assert kinematics["observational_errors"] == {"variance_scale": 1.5}


def test_read_defers_proper_motion_variance_scale_to_runtime(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""    observational_errors:
      variance_scale: 0.0
""",
        kinematics_type="proper_motions",
    )

    config = Configuration().read(user_path, workspace_root=tmp_path)

    assert config.data["kinematic_data"]["observed"]["observational_errors"] == {
        "variance_scale": 0.0
    }


def test_proper_motion_data_can_omit_mge_and_share_population_binning(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        kinematics_type="proper_motions",
        body="""population_data:
  stellar_population:
    data_file: populations.ecsv
    binning: observed
""",
    )
    user_data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
    del user_data["kinematic_data"]["observed"]["mge"]
    user_path.write_text(yaml.safe_dump(user_data, sort_keys=False), encoding="utf-8")

    config = Configuration().read(user_path, workspace_root=tmp_path)

    assert "mge" not in config.data["kinematic_data"]["observed"]
    assert config.data["population_data"]["stellar_population"]["binning"] == "observed"


@pytest.mark.parametrize(
    ("section", "field", "registry"),
    [
        ("potential", "mge", "MGEs"),
        ("kinematic_data", "mge", "MGEs"),
        ("kinematic_data", "binning", "spatial_binnings"),
    ],
)
def test_read_rejects_unknown_registry_reference(
    tmp_path: Path,
    section: str,
    field: str,
    registry: str,
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory)
    user_data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
    entry = "stars" if section == "potential" else "observed"
    user_data[section][entry][field] = "missing"
    user_path.write_text(yaml.safe_dump(user_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"unknown {registry} entry 'missing'"):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_read_rejects_unknown_population_binning_reference(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""population_data:
  stellar_population:
    data_file: populations.ecsv
    binning: missing
""",
    )

    with pytest.raises(
        ValueError,
        match=r"unknown spatial_binnings entry 'missing'",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_read_rejects_population_data_in_kinematics_file(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""population_data:
  stellar_population:
    data_file: observed.ecsv
    binning: observed
""",
    )

    with pytest.raises(
        ValueError,
        match="data_file must be separate from every kinematic_data file",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_light_mge_potential_requires_ml_parameter(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory)
    user_data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
    del user_data["potential"]["stars"]["parameters"]["ml"]
    user_path.write_text(yaml.safe_dump(user_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"parameters is missing required field: ml"):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_mass_mge_potential_rejects_ml_parameter(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory)
    user_data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
    user_data["potential"]["stars"]["type"] = "triaxial_mass_mge"
    user_path.write_text(yaml.safe_dump(user_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"parameters\.ml is invalid for a mass MGE"):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_read_rejects_invalid_potential_rescaling_range(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""parameter_space_settings:
  potential_rescalings:
    mass_scale_range:
      minimum: 2.0
      maximum: 1.0
""",
    )

    with pytest.raises(ValueError, match=r"minimum must not exceed maximum"):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_configuration_session_logs_preparation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory)

    with configuration_session(user_path, workspace_root=tmp_path) as config:
        logging.getLogger("tnt.test").debug("execution debug detail")
        assert config.resolved_path is not None

    logfiles = list((output_directory / "logs").glob("tnt-*.log"))
    assert len(logfiles) == 1
    logfile = logfiles[0].read_text(encoding="utf-8")
    assert config.run_manifest_path is not None
    manifest = yaml.safe_load(config.run_manifest_path.read_text(encoding="utf-8"))
    terminal = capsys.readouterr().err

    assert f"User configuration loaded from {user_path}" in logfile
    assert f"Resolving configuration loaded from {user_path}" in logfile
    assert "Resolved configuration preserved at" in logfile
    assert "execution debug detail" in logfile
    assert "TNT configuration session completed" in logfile
    assert "User configuration loaded from" in terminal
    assert "execution debug detail" not in terminal
    assert manifest["configuration"]["logfile"] == str(logfiles[0])


def test_configuration_session_logs_validation_failure(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""    data_flie: misspelled.ecsv
""",
    )

    with (
        pytest.raises(ValueError, match="data_flie"),
        configuration_session(user_path, workspace_root=tmp_path),
    ):
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
