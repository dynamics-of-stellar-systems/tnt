import logging
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tnt.configuration import (
    Configuration,
    configuration_session,
)
from tnt.configuration import validation as configuration_validation


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
    type: TriaxialLightMGEPotential
    mge: light
    parameters:
      theta:
        unit: "rad"
        value: 1.0
      phi:
        unit: "rad"
        value: 0.5
      psi:
        unit: "rad"
        value: 0.0
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


def test_read_resolves_defaults_without_allocating_a_run(tmp_path: Path) -> None:
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
    assert not repository.exists()
    written = config.as_portable_dict()
    assert written == config.portable_data
    assert config.workspace_root == tmp_path
    assert "dynamic_object_defaults" not in written
    assert "kinematics_type_defaults" not in written
    assert written["cosmological_parameters"]["H"] == {
        "value": 70.0,
        "unit": "km / (s Mpc)",
    }
    assert written["units"] == {
        "internal": {
            "length": "kpc",
            "time": "Myr",
            "mass": "Msun",
            "angle": "rad",
        },
        "display": {"angle": "arcsec", "speed": "km / s"},
    }
    assert config.unit_systems is not None
    assert written["logging_settings"] == {
        "file": {"enabled": True, "level": "DEBUG", "directory": "logs"},
        "console": {"enabled": True, "level": "INFO"},
    }
    assert written["numerics_settings"]["jax_enable_x64"] is True
    assert written["io_settings"]["input_directory"] == "input"
    assert written["io_settings"]["output_directory"] == "output"
    assert config.data["io_settings"]["input_directory"] == str(tmp_path / "input")
    assert config.data["io_settings"]["output_directory"] == str(output_directory)
    parameter = written["potential"]["stars"]["parameters"]["theta"]
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


def test_read_rejects_non_boolean_jax_precision_policy(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    _write_user_config(
        user_path,
        tmp_path / "output",
        body="""numerics_settings:
  jax_enable_x64: 1
""",
    )

    with pytest.raises(
        TypeError, match=r"numerics_settings\.jax_enable_x64 must be a boolean"
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


@pytest.mark.parametrize("enable_x64", [False, True])
def test_read_applies_jax_precision_policy_in_isolated_process(
    tmp_path: Path,
    enable_x64: bool,
) -> None:
    user_path = tmp_path / "user.yaml"
    _write_user_config(
        user_path,
        tmp_path / "output",
        body=f"""numerics_settings:
  jax_enable_x64: {str(enable_x64).lower()}
""",
    )
    probe = """
import sys

import jax
import tnt

from tnt.potential.nfw import _newtonian_gravitational_constant

assert jax.config.jax_enable_x64 is True
expected = sys.argv[3] == "true"
config = tnt.Configuration().read(sys.argv[1], workspace_root=sys.argv[2])
assert jax.config.jax_enable_x64 is expected
assert config.portable_data["numerics_settings"]["jax_enable_x64"] is expected
expected_dtype = "float64" if expected else "float32"
assert str(_newtonian_gravitational_constant().value.dtype) == expected_dtype
"""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            probe,
            str(user_path),
            str(tmp_path),
            str(enable_x64).lower(),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_read_rejects_conflicting_jax_precision_policies_in_one_process(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    _write_user_config(first_path, tmp_path / "first-output")
    _write_user_config(
        second_path,
        tmp_path / "second-output",
        body="""numerics_settings:
  jax_enable_x64: false
""",
    )
    probe = """
import sys

from tnt import Configuration

Configuration().read(sys.argv[1], workspace_root=sys.argv[3])
try:
    Configuration().read(sys.argv[2], workspace_root=sys.argv[3])
except RuntimeError as error:
    assert "cannot be changed to False in the same process" in str(error)
else:
    raise AssertionError("Conflicting JAX precision policies were accepted")
"""

    result = subprocess.run(
        [sys.executable, "-c", probe, str(first_path), str(second_path), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_repeated_reads_do_not_allocate_runs(tmp_path: Path) -> None:
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

    repository = output_directory / "config_repository"
    assert not repository.exists()
    assert repeated.portable_data == first.portable_data
    assert reformatted.portable_data == first.portable_data


def test_separate_reads_retain_independent_resolved_configurations(
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

    assert first.portable_data["system_attributes"]["name"] == "test_system"
    assert second.portable_data["system_attributes"]["name"] == "changed_system"
    assert not (output_directory / "config_repository").exists()


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
    assert config.portable_data["system_attributes"]["distance"] == {
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
  H:
    value: 75.0
""",
    )

    with pytest.raises(
        ValueError,
        match=r"cosmological_parameters\.H.*missing required field.*unit",
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

    with pytest.raises(ValueError, match=r"orbit_library_settings\.orbit_sampler\.nI2"):
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


def test_read_rejects_nonpositive_max_new_mods_per_iter(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""parameter_space_settings:
  stopping_criteria:
    max_new_mods_per_iter: 0
""",
    )

    with pytest.raises(
        ValueError,
        match=r"stopping_criteria\.max_new_mods_per_iter must be positive",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_validate_prior_accepts_a_well_formed_declaration() -> None:
    # A parameter's `prior` field is validated structurally by
    # `_validate_prior`, called from `_validate_parameters` -- exercised
    # directly here since the packaged fixture's `potential.stars.parameters`
    # already declares `ml` and can't be re-declared a second time in one
    # YAML document (see `_write_user_config`).
    configuration_validation._validate_prior(
        {"distribution": "Uniform", "args": [1.0, 9.0]},
        "potential.stars.parameters.ml.prior",
    )


def test_validate_prior_rejects_missing_args() -> None:
    with pytest.raises(
        ValueError,
        match=r"prior is missing required field\(s\): args",
    ):
        configuration_validation._validate_prior(
            {"distribution": "Uniform"}, "potential.stars.parameters.ml.prior"
        )


def test_read_rejects_malformed_prior_plugin_reference(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""parameter_space_settings:
  priors:
    dh_mass_fraction:
      plugin: "priors/mass_fraction.py"
""",
    )

    with pytest.raises(
        ValueError,
        match=r"priors\.dh_mass_fraction\.plugin must have the form",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_read_accepts_declared_prior_plugin(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(
        user_path,
        output_directory,
        body="""parameter_space_settings:
  priors:
    dh_mass_fraction:
      plugin: "priors/mass_fraction.py:mass_fraction"
""",
    )

    config = Configuration().read(user_path, workspace_root=tmp_path)

    assert config.data["parameter_space_settings"]["priors"] == {
        "dh_mass_fraction": {"plugin": "priors/mass_fraction.py:mass_fraction"}
    }


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

    systematics = config.data["kinematic_data"]["observed"]["observational_errors"][
        "systematic_uncertainties"
    ]
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

    with pytest.raises(
        ValueError, match=r"parameters is missing required field\(s\): ml"
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_mass_mge_potential_rejects_ml_parameter(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory)
    user_data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
    user_data["potential"]["stars"]["type"] = "TriaxialMassMGEPotential"
    user_data["potential"]["stars"]["parameters"]["mge_mass_scale"] = {
        "value": 1.0,
        "unit": "",
    }
    user_path.write_text(yaml.safe_dump(user_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"parameters has invalid field\(s\) for TriaxialMassMGEPotential: ml",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_potential_component_rejects_the_removed_include_key(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    _write_user_config(user_path, tmp_path / "output")
    user_data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
    user_data["potential"]["stars"]["include"] = True
    user_path.write_text(yaml.safe_dump(user_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"potential\.stars contains unknown field\(s\): include",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_potential_component_requires_parameters(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    _write_user_config(user_path, tmp_path / "output")
    user_data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
    del user_data["potential"]["stars"]["parameters"]
    user_path.write_text(yaml.safe_dump(user_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"potential\.stars is missing required field: parameters"
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def _oblate_stars(user_data: dict) -> dict:
    """Retarget the default `stars` component at an oblate axisymmetric MGE type."""
    stars = user_data["potential"]["stars"]
    stars["type"] = "OblateLightMGEPotential"
    return stars


def test_oblate_mge_potential_rejects_triaxial_viewing_angles(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory)
    user_data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
    _oblate_stars(user_data)  # keeps the default theta/phi/psi/ml
    user_path.write_text(yaml.safe_dump(user_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            r"parameters has invalid field\(s\) for "
            r"OblateLightMGEPotential: phi, psi, theta"
        ),
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_oblate_mge_potential_requires_inclination_parameter(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory)
    user_data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
    parameters = _oblate_stars(user_data)["parameters"]
    for angle in ("theta", "phi", "psi"):
        del parameters[angle]
    user_path.write_text(yaml.safe_dump(user_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"parameters is missing required field\(s\): inclination"
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_oblate_mge_potential_resolves_with_inclination(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    output_directory = tmp_path / "output"
    _write_user_config(user_path, output_directory)
    user_data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
    parameters = _oblate_stars(user_data)["parameters"]
    for angle in ("theta", "phi", "psi"):
        del parameters[angle]
    parameters["inclination"] = {"unit": "deg", "value": 90.0}
    user_path.write_text(yaml.safe_dump(user_data, sort_keys=False), encoding="utf-8")

    config = Configuration().read(user_path, workspace_root=tmp_path)

    stars = config.data["potential"]["stars"]
    assert stars["type"] == "OblateLightMGEPotential"
    assert stars["parameters"]["inclination"]["value"] == 90.0


def _native_stars(user_data: dict, type_name: str, parameters: dict) -> dict:
    """Retarget the default `stars` component at a native `galax` potential type."""
    stars = user_data["potential"]["stars"]
    stars["type"] = type_name
    stars.pop("mge", None)
    stars.pop("parameterization", None)
    stars["parameters"] = parameters
    return stars


def _write_native_stars_config(
    user_path: Path, output_directory: Path, type_name: str, parameters: dict
) -> None:
    _write_user_config(user_path, output_directory)
    user_data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
    _native_stars(user_data, type_name, parameters)
    user_path.write_text(yaml.safe_dump(user_data, sort_keys=False), encoding="utf-8")


def test_native_galax_potential_resolves_with_its_exact_parameter_set(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.yaml"
    _write_native_stars_config(
        user_path,
        tmp_path / "output",
        "PlummerPotential",
        {
            "m_tot": {"value": 5.0, "unit": "Msun"},
            "r_s": {"value": 1.0, "unit": "kpc"},
        },
    )

    config = Configuration().read(user_path, workspace_root=tmp_path)

    assert config.data["potential"]["stars"]["type"] == "PlummerPotential"


def test_native_galax_potential_rejects_a_missing_parameter(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    _write_native_stars_config(
        user_path,
        tmp_path / "output",
        "PlummerPotential",
        {"m_tot": {"value": 5.0, "unit": "Msun"}},
    )

    with pytest.raises(
        ValueError, match=r"parameters is missing required field\(s\): r_s"
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_native_galax_potential_rejects_an_unexpected_parameter(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    _write_native_stars_config(
        user_path,
        tmp_path / "output",
        "PlummerPotential",
        {
            "m_tot": {"value": 5.0, "unit": "Msun"},
            "r_s": {"value": 1.0, "unit": "kpc"},
            "q1": {"value": 0.9},
        },
    )

    with pytest.raises(
        ValueError,
        match=r"parameters has invalid field\(s\) for PlummerPotential: q1",
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_native_galax_potential_requires_fields_with_a_galax_default(
    tmp_path: Path,
) -> None:
    # TriaxialHernquistPotential's q1/q2 default to 1.0 in galax; TNT still
    # requires them so the model-table schema is complete and reproducible.
    user_path = tmp_path / "user.yaml"
    _write_native_stars_config(
        user_path,
        tmp_path / "output",
        "TriaxialHernquistPotential",
        {
            "m_tot": {"value": 5.0, "unit": "Msun"},
            "r_s": {"value": 1.0, "unit": "kpc"},
        },
    )

    with pytest.raises(
        ValueError, match=r"parameters is missing required field\(s\): q1, q2"
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_native_parameterization_rejects_a_missing_parameter(tmp_path: Path) -> None:
    user_path = tmp_path / "user.yaml"
    _write_user_config(user_path, tmp_path / "output")
    user_data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
    stars = _native_stars(user_data, "NFWPotential", {"c": {"value": 8.0}})
    stars["parameterization"] = "concentration_m200"
    user_path.write_text(yaml.safe_dump(user_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"parameters is missing required field\(s\): M_200"
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_native_parameterization_validates_against_its_own_schema(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.yaml"
    _write_user_config(user_path, tmp_path / "output")
    user_data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
    stars = _native_stars(
        user_data,
        "NFWPotential",
        {
            "c": {"value": 8.0},
            "M_200": {"value": 1.0e12, "unit": "Msun"},
            "r_s": {"value": 1.0, "unit": "kpc"},  # native field, not in this scheme
        },
    )
    stars["parameterization"] = "concentration_m200"
    user_path.write_text(yaml.safe_dump(user_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            r"invalid field\(s\) for NFWPotential with parameterization "
            r"'concentration_m200': r_s"
        ),
    ):
        Configuration().read(user_path, workspace_root=tmp_path)


def test_unrecognized_potential_type_defers_to_resolve_not_parameter_errors(
    tmp_path: Path,
) -> None:
    # A misspelled type must not be reported as "every parameter is extra";
    # the unknown-type error surfaces later, at resolve().
    user_path = tmp_path / "user.yaml"
    _write_native_stars_config(
        user_path,
        tmp_path / "output",
        "PlummerPotentail",
        {"m_tot": {"value": 5.0, "unit": "Msun"}},
    )

    config = Configuration().read(user_path, workspace_root=tmp_path)
    assert config.data["potential"]["stars"]["type"] == "PlummerPotentail"


def test_unimplemented_parameterization_defers_to_resolve_not_parameter_errors(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.yaml"
    _write_user_config(user_path, tmp_path / "output")
    user_data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
    stars = _native_stars(
        user_data,
        "NFWPotential",
        {"c": {"value": 8.0}, "M_200": {"value": 1.0e12, "unit": "Msun"}},
    )
    stars["parameterization"] = "not_a_real_scheme"
    user_path.write_text(yaml.safe_dump(user_data, sort_keys=False), encoding="utf-8")

    config = Configuration().read(user_path, workspace_root=tmp_path)
    assert config.data["potential"]["stars"]["parameterization"] == "not_a_real_scheme"


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
        assert config.logfile_path is not None

    logfiles = list((output_directory / "logs").glob("tnt-*.log"))
    assert len(logfiles) == 1
    logfile = logfiles[0].read_text(encoding="utf-8")
    terminal = capsys.readouterr().err

    assert f"User configuration loaded from {user_path}" in logfile
    assert f"Resolving configuration loaded from {user_path}" in logfile
    assert "Resolved configuration preserved at" not in logfile
    assert "execution debug detail" in logfile
    assert "TNT configuration session completed" in logfile
    assert "User configuration loaded from" in terminal
    assert "execution debug detail" not in terminal
    assert config.logfile_path == logfiles[0]
    assert not (output_directory / "config_repository").exists()


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
