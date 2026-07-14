"""Validate resolved TNT configuration data without constructing objects."""

from __future__ import annotations

import math
from collections.abc import Collection
from typing import Any

ConfigDict = dict[str, Any]

_TOP_LEVEL_KEYS = {
    "analysis_settings",
    "cosmological_parameters",
    "execution_settings",
    "io_settings",
    "mge_settings",
    "numerics_settings",
    "orbit_library_settings",
    "parameter_space_settings",
    "system_attributes",
    "system_components",
    "system_parameters",
    "weight_solver_settings",
}
_COMPONENT_TYPES = {"nfw", "plummer", "triaxial_visible_component"}
_KINEMATICS_TYPES = {"bayes_losvd", "gauss_hermite", "proper_motions"}


def validate_resolved_configuration(config: ConfigDict) -> None:
    """Validate a fully merged configuration using data-only checks.

    The checks in this module deliberately avoid constructing scientific
    objects, reading observational files, or checking optional dependencies.

    Args:
        config: Fully merged configuration with dynamic defaults applied.

    Raises:
        TypeError: If a value has the wrong data type.
        ValueError: If a field is unknown, missing, or semantically invalid.
    """
    _reject_unknown_keys(config, _TOP_LEVEL_KEYS, "configuration")
    _require_keys(
        config,
        {"system_attributes", "system_components", "system_parameters"},
        "configuration",
    )

    _validate_cosmological_parameters(
        _mapping(config, "cosmological_parameters", "configuration")
    )
    _validate_mge_settings(_mapping(config, "mge_settings", "configuration"))
    _validate_numerics_settings(_mapping(config, "numerics_settings", "configuration"))
    _validate_system_attributes(_mapping(config, "system_attributes", "configuration"))
    _validate_components(_mapping(config, "system_components", "configuration"))
    _validate_parameters(
        _mapping(config, "system_parameters", "configuration"),
        "system_parameters",
        require_nonempty=True,
    )
    _validate_orbit_library_settings(
        _mapping(config, "orbit_library_settings", "configuration")
    )
    _validate_weight_solver_settings(
        _mapping(config, "weight_solver_settings", "configuration")
    )
    _validate_parameter_space_settings(
        _mapping(config, "parameter_space_settings", "configuration")
    )
    _validate_analysis_settings(_mapping(config, "analysis_settings", "configuration"))
    _validate_io_settings(_mapping(config, "io_settings", "configuration"))
    _validate_execution_settings(
        _mapping(config, "execution_settings", "configuration")
    )


def _validate_cosmological_parameters(settings: ConfigDict) -> None:
    path = "cosmological_parameters"
    _reject_unknown_keys(settings, {"H0"}, path)
    _require_keys(settings, {"H0"}, path)
    _positive_number(settings["H0"], f"{path}.H0")


def _validate_mge_settings(settings: ConfigDict) -> None:
    path = "mge_settings"
    _reject_unknown_keys(settings, {"axial_ratio_cap"}, path)
    _require_keys(settings, {"axial_ratio_cap"}, path)
    cap = _number(settings["axial_ratio_cap"], f"{path}.axial_ratio_cap")
    if not 0 < cap <= 1:
        raise ValueError(
            f"{path}.axial_ratio_cap must be greater than 0 and at most 1."
        )


def _validate_numerics_settings(settings: ConfigDict) -> None:
    path = "numerics_settings"
    _reject_unknown_keys(
        settings,
        {
            "constraint_error_floors",
            "model_comparison_relative_tolerance",
            "parameter_grid_relative_tolerance",
        },
        path,
    )
    _require_keys(
        settings,
        {
            "constraint_error_floors",
            "model_comparison_relative_tolerance",
            "parameter_grid_relative_tolerance",
        },
        path,
    )
    for key in (
        "model_comparison_relative_tolerance",
        "parameter_grid_relative_tolerance",
    ):
        _positive_number(settings.get(key), f"{path}.{key}")
    floors = _mapping(settings, "constraint_error_floors", path)
    _reject_unknown_keys(
        floors, {"intrinsic_mass", "total_mass"}, f"{path}.constraint_error_floors"
    )
    _require_keys(
        floors, {"intrinsic_mass", "total_mass"}, f"{path}.constraint_error_floors"
    )
    for key in ("intrinsic_mass", "total_mass"):
        _positive_number(floors[key], f"{path}.constraint_error_floors.{key}")


def _validate_system_attributes(attributes: ConfigDict) -> None:
    path = "system_attributes"
    _reject_unknown_keys(attributes, {"distance_mpc", "name"}, path)
    _require_keys(attributes, {"distance_mpc", "name"}, path)
    _positive_number(attributes["distance_mpc"], f"{path}.distance_mpc")
    _nonempty_string(attributes["name"], f"{path}.name")


def _validate_components(components: ConfigDict) -> None:
    path = "system_components"
    if not components:
        raise ValueError(f"{path} must contain at least one component.")
    for name, component_value in components.items():
        name = _dynamic_name(name, path)
        component_path = f"{path}.{name}"
        component = _require_mapping(component_value, component_path)
        _reject_unknown_keys(
            component,
            {"include", "kinematics", "mge", "parameters", "type"},
            component_path,
        )
        _require_keys(component, {"include", "type"}, component_path)
        component_type = _choice(
            component["type"],
            _COMPONENT_TYPES,
            f"{component_path}.type",
        )
        include = _boolean(component["include"], f"{component_path}.include")

        if "parameters" in component:
            _validate_parameters(
                _mapping(component, "parameters", component_path),
                f"{component_path}.parameters",
                require_nonempty=include,
            )
        elif include:
            raise ValueError(f"{component_path} is missing required field: parameters.")

        if "mge" in component:
            _validate_component_mge(
                _mapping(component, "mge", component_path),
                f"{component_path}.mge",
            )
        if "kinematics" in component:
            _validate_kinematics(
                _mapping(component, "kinematics", component_path),
                f"{component_path}.kinematics",
            )

        if include and component_type == "triaxial_visible_component":
            _require_keys(component, {"kinematics", "mge"}, component_path)
            if not component["kinematics"]:
                raise ValueError(f"{component_path}.kinematics must not be empty.")


def _validate_component_mge(mge: ConfigDict, path: str) -> None:
    _reject_unknown_keys(mge, {"luminosity_file", "potential_file"}, path)
    _require_keys(mge, {"luminosity_file", "potential_file"}, path)
    for key in ("luminosity_file", "potential_file"):
        _nonempty_string(mge[key], f"{path}.{key}")


def _validate_parameters(
    parameters: ConfigDict,
    path: str,
    *,
    require_nonempty: bool,
) -> None:
    if require_nonempty and not parameters:
        raise ValueError(f"{path} must contain at least one parameter.")
    for name, parameter_value in parameters.items():
        name = _dynamic_name(name, path)
        parameter_path = f"{path}.{name}"
        parameter = _require_mapping(parameter_value, parameter_path)
        _reject_unknown_keys(
            parameter,
            {
                "fixed",
                "generator_settings",
                "latex_label",
                "logarithmic",
                "value",
            },
            parameter_path,
        )
        _require_keys(
            parameter,
            {"fixed", "logarithmic", "value"},
            parameter_path,
        )
        _boolean(parameter["fixed"], f"{parameter_path}.fixed")
        _boolean(parameter["logarithmic"], f"{parameter_path}.logarithmic")
        value = _number(parameter["value"], f"{parameter_path}.value")
        if "latex_label" in parameter:
            _nonempty_string(parameter["latex_label"], f"{parameter_path}.latex_label")
        if "generator_settings" in parameter:
            _validate_parameter_generator_settings(
                _mapping(parameter, "generator_settings", parameter_path),
                f"{parameter_path}.generator_settings",
                value,
            )


def _validate_parameter_generator_settings(
    settings: ConfigDict,
    path: str,
    value: float,
) -> None:
    keys = {"lower_bound", "minimum_step", "step", "upper_bound"}
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    lower = _number(settings["lower_bound"], f"{path}.lower_bound")
    upper = _number(settings["upper_bound"], f"{path}.upper_bound")
    if lower > upper:
        raise ValueError(f"{path}.lower_bound must not exceed upper_bound.")
    if not lower <= value <= upper:
        raise ValueError(
            f"The parameter value at {path.rsplit('.', 1)[0]}.value must lie within its bounds."
        )
    _positive_number(settings["step"], f"{path}.step")
    _nonnegative_number(settings["minimum_step"], f"{path}.minimum_step")


def _validate_kinematics(kinematics: ConfigDict, path: str) -> None:
    for name, settings_value in kinematics.items():
        name = _dynamic_name(name, path)
        settings_path = f"{path}.{name}"
        settings = _require_mapping(settings_value, settings_path)
        _reject_unknown_keys(
            settings,
            {
                "aperture_file",
                "bin_file",
                "data_file",
                "histogram",
                "type",
                "warning_thresholds",
                "with_pops",
            },
            settings_path,
        )
        _require_keys(
            settings,
            {"aperture_file", "bin_file", "data_file", "type", "with_pops"},
            settings_path,
        )
        kinematics_type = _choice(
            settings["type"],
            _KINEMATICS_TYPES,
            f"{settings_path}.type",
        )
        _boolean(settings["with_pops"], f"{settings_path}.with_pops")
        for key in ("aperture_file", "bin_file", "data_file"):
            _nonempty_string(settings[key], f"{settings_path}.{key}")
        if "histogram" in settings:
            _validate_histogram(
                _mapping(settings, "histogram", settings_path),
                f"{settings_path}.histogram",
                kinematics_type,
            )
        if "warning_thresholds" in settings:
            if kinematics_type != "proper_motions":
                raise ValueError(
                    f"{settings_path}.warning_thresholds is only valid for "
                    "proper_motions."
                )
            _validate_proper_motion_warning_thresholds(
                _mapping(settings, "warning_thresholds", settings_path),
                f"{settings_path}.warning_thresholds",
            )


def _validate_histogram(
    histogram: ConfigDict,
    path: str,
    kinematics_type: str,
) -> None:
    explicit_keys = {"bins", "center", "width"}
    derived_keys = {
        "bin_width_sigma_fraction",
        "center",
        "oversampling_factor",
        "sigma_extent",
        "systemic_velocity",
        "width_scale",
    }
    _reject_unknown_keys(histogram, explicit_keys | derived_keys, path)
    if explicit_keys.issubset(histogram):
        if set(histogram) != explicit_keys:
            raise ValueError(
                f"{path} must contain only width, center, and bins when "
                "explicit histogram metadata is used."
            )
        _positive_number(histogram["width"], f"{path}.width")
        _number(histogram["center"], f"{path}.center")
        bins = _integer(histogram["bins"], f"{path}.bins")
        if bins <= 0 or bins % 2 == 0:
            raise ValueError(f"{path}.bins must be a positive odd integer.")
        return

    if kinematics_type == "gauss_hermite":
        allowed = {"bin_width_sigma_fraction", "center", "sigma_extent"}
        _reject_unknown_keys(histogram, allowed, path)
        _require_keys(histogram, allowed, path)
        _positive_number(histogram["sigma_extent"], f"{path}.sigma_extent")
        _positive_number(
            histogram["bin_width_sigma_fraction"],
            f"{path}.bin_width_sigma_fraction",
        )
        _number(histogram["center"], f"{path}.center")
    elif kinematics_type == "bayes_losvd":
        allowed = {"center", "oversampling_factor", "systemic_velocity", "width_scale"}
        _reject_unknown_keys(histogram, allowed, path)
        _require_keys(histogram, allowed, path)
        _positive_number(histogram["width_scale"], f"{path}.width_scale")
        _positive_number(
            histogram["oversampling_factor"],
            f"{path}.oversampling_factor",
        )
        _number(histogram["center"], f"{path}.center")
        _choice(
            histogram["systemic_velocity"],
            {"flux_weighted"},
            f"{path}.systemic_velocity",
        )
    else:
        raise ValueError(
            f"{path} for proper_motions must use explicit width, center, and bins."
        )


def _validate_proper_motion_warning_thresholds(
    thresholds: ConfigDict,
    path: str,
) -> None:
    keys = {"max_bin_width_sigma_ratio", "min_histogram_width_sigma_ratio"}
    _reject_unknown_keys(thresholds, keys, path)
    _require_keys(thresholds, keys, path)
    for key in keys:
        _positive_number(thresholds[key], f"{path}.{key}")


def _validate_orbit_library_settings(settings: ConfigDict) -> None:
    path = "orbit_library_settings"
    keys = {
        "accuracy",
        "dithering",
        "logrmax",
        "logrmin",
        "nE",
        "nI2",
        "nI3",
        "number_orbits",
        "orbital_periods",
        "quad_nph",
        "quad_nr",
        "quad_nth",
        "random_seed",
        "sampling",
        "starting_orbit",
    }
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    for key in (
        "nE",
        "nI3",
        "dithering",
        "quad_nph",
        "quad_nr",
        "quad_nth",
        "sampling",
        "starting_orbit",
    ):
        value = _integer(settings[key], f"{path}.{key}")
        if value <= 0:
            raise ValueError(f"{path}.{key} must be a positive integer.")
    n_i2 = _integer(settings["nI2"], f"{path}.nI2")
    if n_i2 < 4:
        raise ValueError(f"{path}.nI2 must be at least 4.")
    _integer(settings["number_orbits"], f"{path}.number_orbits")
    number_orbits = settings["number_orbits"]
    if number_orbits != -1 and number_orbits <= 0:
        raise ValueError(f"{path}.number_orbits must be -1 or a positive integer.")
    _integer(settings["random_seed"], f"{path}.random_seed")
    _positive_number(settings["orbital_periods"], f"{path}.orbital_periods")
    _positive_number(settings["accuracy"], f"{path}.accuracy")
    logrmin = _number(settings["logrmin"], f"{path}.logrmin")
    logrmax = _number(settings["logrmax"], f"{path}.logrmax")
    if logrmin >= logrmax:
        raise ValueError(f"{path}.logrmin must be less than logrmax.")


def _validate_weight_solver_settings(settings: ConfigDict) -> None:
    path = "weight_solver_settings"
    keys = {
        "GH_sys_err",
        "PM_sys_err_factor",
        "counter_rotating_orbit_cut",
        "lum_intr_rel_err",
        "maxiter_factor",
        "nnls_solver",
        "number_GH",
        "reattempt_failures",
        "regularisation",
        "sb_proj_rel_err",
        "type",
    }
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    _choice(settings["type"], {"NNLS"}, f"{path}.type")
    _choice(settings["nnls_solver"], {"cvxopt", "scipy"}, f"{path}.nnls_solver")
    _positive_number(settings["maxiter_factor"], f"{path}.maxiter_factor")
    _nonnegative_number(settings["regularisation"], f"{path}.regularisation")
    number_gh = _integer(settings["number_GH"], f"{path}.number_GH")
    if number_gh <= 0:
        raise ValueError(f"{path}.number_GH must be a positive integer.")
    _nonempty_string(settings["GH_sys_err"], f"{path}.GH_sys_err")
    for key in ("PM_sys_err_factor", "lum_intr_rel_err", "sb_proj_rel_err"):
        _nonnegative_number(settings[key], f"{path}.{key}")
    _boolean(settings["reattempt_failures"], f"{path}.reattempt_failures")
    _validate_counter_rotating_cut(
        _mapping(settings, "counter_rotating_orbit_cut", path),
        f"{path}.counter_rotating_orbit_cut",
    )


def _validate_counter_rotating_cut(settings: ConfigDict, path: str) -> None:
    keys = {
        "enabled",
        "h1_penalty_scale",
        "min_abs_observed_velocity_over_sigma",
        "min_affected_apertures",
        "min_orbit_velocity_difference_over_sigma",
        "require_opposite_velocity_sign",
    }
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    _boolean(settings["enabled"], f"{path}.enabled")
    _boolean(
        settings["require_opposite_velocity_sign"],
        f"{path}.require_opposite_velocity_sign",
    )
    for key in (
        "h1_penalty_scale",
        "min_abs_observed_velocity_over_sigma",
        "min_orbit_velocity_difference_over_sigma",
    ):
        _positive_number(settings[key], f"{path}.{key}")
    apertures = _integer(
        settings["min_affected_apertures"], f"{path}.min_affected_apertures"
    )
    if apertures <= 0:
        raise ValueError(f"{path}.min_affected_apertures must be positive.")


def _validate_parameter_space_settings(settings: ConfigDict) -> None:
    path = "parameter_space_settings"
    keys = {"generator_settings", "generator_type", "stopping_criteria", "which_chi2"}
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    _choice(settings["generator_type"], {"GridSearch"}, f"{path}.generator_type")
    _choice(
        settings["which_chi2"], {"chi2", "kinchi2", "kinmapchi2"}, f"{path}.which_chi2"
    )
    generator = _mapping(settings, "generator_settings", path)
    _reject_unknown_keys(
        generator, {"delta_chi2_threshold"}, f"{path}.generator_settings"
    )
    _require_keys(generator, {"delta_chi2_threshold"}, f"{path}.generator_settings")
    _validate_tagged_threshold(
        _mapping(generator, "delta_chi2_threshold", f"{path}.generator_settings"),
        f"{path}.generator_settings.delta_chi2_threshold",
        {"absolute", "fraction_of_sqrt_2n_observations"},
    )
    stopping = _mapping(settings, "stopping_criteria", path)
    _reject_unknown_keys(
        stopping,
        {"minimum_delta_chi2", "n_max_iter", "n_max_mods"},
        f"{path}.stopping_criteria",
    )
    _require_keys(
        stopping,
        {"minimum_delta_chi2", "n_max_iter", "n_max_mods"},
        f"{path}.stopping_criteria",
    )
    _validate_tagged_threshold(
        _mapping(stopping, "minimum_delta_chi2", f"{path}.stopping_criteria"),
        f"{path}.stopping_criteria.minimum_delta_chi2",
        {"absolute", "relative"},
    )
    for key in ("n_max_iter", "n_max_mods"):
        value = _integer(stopping[key], f"{path}.stopping_criteria.{key}")
        if value <= 0:
            raise ValueError(f"{path}.stopping_criteria.{key} must be positive.")


def _validate_tagged_threshold(
    threshold: ConfigDict,
    path: str,
    allowed_modes: set[str],
) -> None:
    _reject_unknown_keys(threshold, {"mode", "value"}, path)
    _require_keys(threshold, {"mode", "value"}, path)
    _choice(threshold["mode"], allowed_modes, f"{path}.mode")
    _nonnegative_number(threshold["value"], f"{path}.value")


def _validate_analysis_settings(settings: ConfigDict) -> None:
    path = "analysis_settings"
    _reject_unknown_keys(settings, {"kinematic_moments", "orbit_decomposition"}, path)
    _require_keys(settings, {"kinematic_moments", "orbit_decomposition"}, path)
    decomposition = _mapping(settings, "orbit_decomposition", path)
    decomposition_path = f"{path}.orbit_decomposition"
    _reject_unknown_keys(
        decomposition,
        {
            "cache",
            "circularity_thresholds",
            "component_naming",
            "write_component_weights",
        },
        decomposition_path,
    )
    _require_keys(
        decomposition,
        {
            "cache",
            "circularity_thresholds",
            "component_naming",
            "write_component_weights",
        },
        decomposition_path,
    )
    _choice(
        decomposition["component_naming"],
        {"bulge_disk"},
        f"{decomposition_path}.component_naming",
    )
    _boolean(decomposition["cache"], f"{decomposition_path}.cache")
    _boolean(
        decomposition["write_component_weights"],
        f"{decomposition_path}.write_component_weights",
    )
    thresholds = _mapping(decomposition, "circularity_thresholds", decomposition_path)
    threshold_path = f"{decomposition_path}.circularity_thresholds"
    threshold_keys = {
        "cold_min",
        "counter_rotating_cold_max",
        "counter_rotating_warm_max",
        "warm_min",
    }
    _reject_unknown_keys(thresholds, threshold_keys, threshold_path)
    _require_keys(thresholds, threshold_keys, threshold_path)
    values = {
        key: _number(thresholds[key], f"{threshold_path}.{key}")
        for key in threshold_keys
    }
    if any(not -1 <= value <= 1 for value in values.values()):
        raise ValueError(f"{threshold_path} values must lie between -1 and 1.")
    if not (
        values["counter_rotating_cold_max"]
        < values["counter_rotating_warm_max"]
        < values["warm_min"]
        < values["cold_min"]
    ):
        raise ValueError(
            f"{threshold_path} values must be strictly increasing from counter-rotating cold to cold."
        )
    moments = _mapping(settings, "kinematic_moments", path)
    moments_path = f"{path}.kinematic_moments"
    _reject_unknown_keys(moments, {"velocity_dispersion_method"}, moments_path)
    _require_keys(moments, {"velocity_dispersion_method"}, moments_path)
    _choice(
        moments["velocity_dispersion_method"],
        {"gaussian_fit"},
        f"{moments_path}.velocity_dispersion_method",
    )


def _validate_io_settings(settings: ConfigDict) -> None:
    path = "io_settings"
    keys = {"all_models_file", "input_directory", "output_directory"}
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    for key in keys:
        _nonempty_string(settings[key], f"{path}.{key}")


def _validate_execution_settings(settings: ConfigDict) -> None:
    path = "execution_settings"
    keys = {
        "external_chi2_workers",
        "model_processing_order",
        "orbit_family_integration_in_parallel",
        "orbit_workers",
        "weight_workers",
    }
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    for key in ("external_chi2_workers", "orbit_workers", "weight_workers"):
        _worker_count(settings[key], f"{path}.{key}")
    _choice(
        settings["model_processing_order"],
        {"model_by_model", "stage_by_stage"},
        f"{path}.model_processing_order",
    )
    _boolean(
        settings["orbit_family_integration_in_parallel"],
        f"{path}.orbit_family_integration_in_parallel",
    )


def _worker_count(value: Any, path: str) -> None:
    if value == "all_available":
        return
    workers = _integer(value, path)
    if workers <= 0:
        raise ValueError(f"{path} must be a positive integer or 'all_available'.")


def _mapping(mapping: ConfigDict, key: str, parent: str) -> ConfigDict:
    if key not in mapping:
        raise ValueError(f"{parent} is missing required field: {key}.")
    return _require_mapping(mapping[key], f"{parent}.{key}")


def _require_mapping(value: Any, path: str) -> ConfigDict:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a mapping.")
    return value


def _reject_unknown_keys(
    mapping: ConfigDict,
    allowed: Collection[str],
    path: str,
) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown field(s): {', '.join(unknown)}.")


def _require_keys(mapping: ConfigDict, required: set[str], path: str) -> None:
    missing = sorted(required - mapping.keys())
    if missing:
        raise ValueError(f"{path} is missing required field(s): {', '.join(missing)}.")


def _dynamic_name(value: Any, parent: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Names below {parent} must be non-empty strings.")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} must be a non-empty string.")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean.")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite.")
    return number


def _positive_number(value: Any, path: str) -> float:
    number = _number(value, path)
    if number <= 0:
        raise ValueError(f"{path} must be greater than zero.")
    return number


def _nonnegative_number(value: Any, path: str) -> float:
    number = _number(value, path)
    if number < 0:
        raise ValueError(f"{path} must not be negative.")
    return number


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer.")
    return value


def _choice(value: Any, allowed: set[str], path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string.")
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{path} must be one of: {choices}; got {value!r}.")
    return value
