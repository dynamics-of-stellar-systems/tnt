from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from astropy.table import QTable

from tnt.all_models import AllModels
from tnt.configuration_compatibility import (
    ConfigurationCompatibilityError,
    _critical_configuration,
    _different_paths,
    ensure_resume_compatible,
)
from tnt.run_config_log import RunManifestReference


def _write_inputs(directory: Path, suffix: str = "") -> None:
    directory.mkdir(parents=True)
    for filename in ("light.ecsv", "bins.npy", "kin.ecsv", "pop.ecsv"):
        (directory / filename).write_bytes(f"{filename}{suffix}".encode())


def _config(input_directory: Path) -> dict[str, object]:
    return {
        "units": {
            "internal": {
                "length": "kpc",
                "time": "Myr",
                "mass": "Msun",
                "angle": "rad",
                "power": "Lsun",
            },
            "display": {"angle": "arcsec"},
        },
        "cosmological_parameters": {
            "H0": {"value": 0.1, "unit": "1 / Myr"}
        },
        "system_attributes": {
            "name": "galaxy",
            "distance": {"value": 10.0, "unit": "kpc"},
        },
        "mge_settings": {
            "intrinsic_mass_quad_order": 10,
            "projected_mass_quad_order": 10,
        },
        "numerics_settings": {
            "model_comparison_relative_tolerance": 1.0e-10,
            "parameter_grid_relative_tolerance": 1.0e-6,
            "constraint_error_floors": {
                "total_mass": 1.0e-8,
                "intrinsic_mass": 1.0e-16,
            },
        },
        "MGEs": {"light": "light.ecsv"},
        "spatial_binnings": {
            "observed": {
                "min_x": {"value": -1.0, "unit": "rad"},
                "min_y": {"value": -1.0, "unit": "rad"},
                "x_extent": {"value": 2.0, "unit": "rad"},
                "y_extent": {"value": 2.0, "unit": "rad"},
                "PA": {"value": 0.0, "unit": "rad"},
                "bins_file": "bins.npy",
            }
        },
        "potential": {
            "stars": {
                "include": True,
                "type": "triaxial_light_mge",
                "mge": "light",
                "parameters": {
                    "q": {
                        "value": 0.8,
                        "fixed": False,
                        "latex_label": "q",
                        "generator_settings": {"lower_bound": 0.5},
                    },
                    "ml": {
                        "value": 5.0,
                        "unit": "Msun / Lsun",
                        "fixed": False,
                    },
                },
            }
        },
        "kinematic_data": {
            "observed": {
                "type": "gauss_hermite",
                "binning": "observed",
                "mge": "light",
                "data_file": "kin.ecsv",
                "maximum_gh_order": 4,
            }
        },
        "population_data": {
            "populations": {
                "binning": "observed",
                "data_file": "pop.ecsv",
            }
        },
        "orbit_library_settings": {"random_seed": 42, "accuracy": 1.0e-5},
        "weight_solver_settings": {"type": "NNLS", "regularisation": 0},
        "parameter_space_settings": {
            "generator_type": "GridSearch",
            "which_chi2": "kinchi2",
            "generator_settings": {"threshold": 1.0},
            "stopping_criteria": {"n_new_iter": 10},
        },
        "analysis_settings": {"cache": True},
        "logging_settings": {"console": {"level": "INFO"}},
        "io_settings": {
            "input_directory": str(input_directory),
            "output_directory": str(input_directory.parent / "output"),
            "all_models_file": "all_models.ecsv",
        },
        "execution_settings": {"orbit_workers": 1},
    }


def _run(
    tmp_path: Path,
    run_id: int,
    config: dict[str, object],
) -> RunManifestReference:
    repository = tmp_path / "config_repository"
    run_directory = repository / "runs" / f"{run_id:04d}"
    run_directory.mkdir(parents=True)
    resolved_path = run_directory / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest_path = run_directory / "run_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "manifest_version": 3,
                "run_id": run_id,
                "configuration": {
                    "resolved": resolved_path.relative_to(repository).as_posix()
                },
            }
        ),
        encoding="utf-8",
    )
    return RunManifestReference.from_run_manifest(manifest_path)


def test_operational_and_search_changes_are_compatible(tmp_path: Path) -> None:
    first_inputs = tmp_path / "input-one"
    second_inputs = tmp_path / "input-two"
    _write_inputs(first_inputs)
    _write_inputs(second_inputs)
    baseline = _config(first_inputs)
    changed = deepcopy(baseline)
    changed["units"]["display"] = {"angle": "deg"}
    changed["system_attributes"]["name"] = "renamed"
    changed["potential"]["stars"]["parameters"]["q"].update(
        {
            "value": 0.7,
            "fixed": True,
            "latex_label": "changed",
            "generator_settings": {"lower_bound": 0.2},
        }
    )
    changed["parameter_space_settings"] = {"generator_type": "SinglePoint"}
    changed["analysis_settings"] = {"cache": False}
    changed["logging_settings"] = {"console": {"level": "ERROR"}}
    changed["execution_settings"] = {"orbit_workers": "all_available"}
    changed["io_settings"] = {
        "input_directory": str(second_inputs),
        "output_directory": str(tmp_path / "moved-output"),
        "all_models_file": "renamed.ecsv",
    }

    assert _different_paths(
        _critical_configuration(baseline),
        _critical_configuration(changed),
    ) == []


def test_equivalent_declared_units_are_compatible(tmp_path: Path) -> None:
    input_directory = tmp_path / "input"
    _write_inputs(input_directory)
    baseline = _config(input_directory)
    baseline["kinematic_data"]["observed"].update(
        {
            "histogram": {
                "width": {"value": 1000.0, "unit": "km / s"},
                "center": {"value": 0.0, "unit": "km / s"},
                "bins": 101,
            },
            "observational_errors": {
                "systematic_uncertainties": {
                    "v": {"value": 0.0, "unit": "km / s"},
                    "sigma": {"value": 0.0, "unit": "km / s"},
                    "h3": 0.0,
                    "h4": 0.0,
                }
            },
        }
    )
    equivalent = deepcopy(baseline)
    equivalent["system_attributes"]["distance"] = {
        "value": 10000.0,
        "unit": "pc",
    }
    equivalent["spatial_binnings"]["observed"].update(
        {
            "min_x": {"value": -1000.0, "unit": "mrad"},
            "min_y": {"value": -1000.0, "unit": "mrad"},
            "x_extent": {"value": 2000.0, "unit": "mrad"},
            "y_extent": {"value": 2000.0, "unit": "mrad"},
        }
    )
    equivalent["kinematic_data"]["observed"]["histogram"].update(
        {
            "width": {"value": 1_000_000.0, "unit": "m / s"},
            "center": {"value": 0.0, "unit": "m / s"},
        }
    )
    systematics = equivalent["kinematic_data"]["observed"][
        "observational_errors"
    ]["systematic_uncertainties"]
    systematics["v"] = {"value": 0.0, "unit": "m / s"}
    systematics["sigma"] = {"value": 0.0, "unit": "m / s"}
    equivalent["potential"]["stars"]["parameters"]["ml"]["unit"] = (
        "solMass / solLum"
    )

    assert _different_paths(
        _critical_configuration(baseline),
        _critical_configuration(equivalent),
    ) == []


@pytest.mark.parametrize(
    ("section", "mutate", "expected_path"),
    [
        (
            "cosmological_parameters",
            lambda value: value["H0"].update(value=0.2),
            "critical_configuration.cosmological_parameters.H0",
        ),
        (
            "system_attributes",
            lambda value: value["distance"].update(value=11.0),
            "critical_configuration.system_attributes.distance",
        ),
        (
            "units",
            lambda value: value["internal"].update(length="pc"),
            "critical_configuration.units.internal.length",
        ),
        (
            "mge_settings",
            lambda value: value.update(intrinsic_mass_quad_order=12),
            "critical_configuration.mge_settings.intrinsic_mass_quad_order",
        ),
        (
            "MGEs",
            lambda value: value.update(light="different-light.ecsv"),
            "critical_configuration.MGEs.light",
        ),
        (
            "numerics_settings",
            lambda value: value.update(parameter_grid_relative_tolerance=2.0e-6),
            "critical_configuration.numerics_settings.parameter_grid_relative_tolerance",
        ),
        (
            "orbit_library_settings",
            lambda value: value.update(accuracy=2.0e-5),
            "critical_configuration.orbit_library_settings.accuracy",
        ),
        (
            "weight_solver_settings",
            lambda value: value.update(regularisation=1),
            "critical_configuration.weight_solver_settings.regularisation",
        ),
        (
            "spatial_binnings",
            lambda value: value["observed"]["PA"].update(value=1.0),
            "critical_configuration.spatial_binnings.observed.PA",
        ),
        (
            "kinematic_data",
            lambda value: value["observed"].update(maximum_gh_order=6),
            "critical_configuration.kinematic_data.observed.maximum_gh_order",
        ),
        (
            "potential",
            lambda value: value["stars"]["parameters"].update(extra={"value": 1.0}),
            "critical_configuration.potential.stars.parameters.extra",
        ),
    ],
)
def test_critical_configuration_changes_are_rejected(
    tmp_path: Path,
    section: str,
    mutate: object,
    expected_path: str,
) -> None:
    input_directory = tmp_path / "input"
    _write_inputs(input_directory)
    baseline = _config(input_directory)
    changed = deepcopy(baseline)
    mutate(changed[section])

    differences = _different_paths(
        _critical_configuration(baseline),
        _critical_configuration(changed),
        "critical_configuration",
    )

    assert expected_path in differences


def test_scientific_file_content_change_is_not_part_of_contract(tmp_path: Path) -> None:
    input_directory = tmp_path / "input"
    _write_inputs(input_directory)
    config = _config(input_directory)
    baseline = _critical_configuration(config)
    (input_directory / "kin.ecsv").write_bytes(b"changed")

    assert _critical_configuration(config) == baseline


def test_resume_rejects_incompatible_baseline_run(tmp_path: Path) -> None:
    input_directory = tmp_path / "input"
    _write_inputs(input_directory)
    historical_config = _config(input_directory)
    current_config = deepcopy(historical_config)
    current_config["numerics_settings"]["parameter_grid_relative_tolerance"] = 2.0e-6
    baseline_run = _run(tmp_path, 0, historical_config)

    with pytest.raises(
        ConfigurationCompatibilityError,
        match="numerics_settings.parameter_grid_relative_tolerance",
    ):
        ensure_resume_compatible(
            _critical_configuration(current_config),
            baseline_run,
            AllModels(),
            "kinchi2",
        )


def test_resume_rejects_unavailable_historical_chi2(tmp_path: Path) -> None:
    input_directory = tmp_path / "input"
    _write_inputs(input_directory)
    config = _config(input_directory)
    baseline_run = _run(tmp_path, 0, config)
    models = AllModels(
        QTable(
            {
                "stars.q": [0.8],
                "stars.ml": [5.0],
                "iteration": [0],
                "orblib_done": [True],
                "weights_done": [True],
                "chi2": [1.0],
            }
        )
    )

    with pytest.raises(ConfigurationCompatibilityError, match="no such column"):
        ensure_resume_compatible(
            _critical_configuration(config),
            baseline_run,
            models,
            "kinchi2",
        )


def test_resume_allows_negative_orbit_seed(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "input"
    _write_inputs(input_directory)
    config = _config(input_directory)
    config["orbit_library_settings"]["random_seed"] = -1
    baseline_run = _run(tmp_path, 0, config)

    ensure_resume_compatible(
        _critical_configuration(config),
        baseline_run,
        AllModels(),
        "kinchi2",
    )
