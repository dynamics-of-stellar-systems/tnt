"""Validate resolved TNT configuration data without constructing objects.

Configuration validation may import runtime-family modules to read their static
registration metadata (`_type`, `_raw_dimensions`, and registry type names).
Import-time registration is allowed; validation must not instantiate
scientific objects, load scientific input data, or begin scientific execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tnt.kinematics.registry import kinematics_type_names
from tnt.parameter_generator import parameter_generator_required_settings
from tnt.potential.registry import (
    is_registered_component_type,
    parameter_schema_is_known,
    raw_parameter_dimensions,
)
from tnt.units import declared_quantity_value, validate_configuration_quantities
from tnt.validation import (
    _integer,
    _mapping,
    _nonnegative_number,
    _number,
    _positive_number,
    _reject_unknown_keys,
    _require_keys,
    _required_mapping,
    _string,
)

ConfigDict = dict[str, Any]

_TOP_LEVEL_KEYS = {
    "MGEs",
    "analysis_settings",
    "cosmological_parameters",
    "execution_settings",
    "io_settings",
    "kinematic_data",
    "logging_settings",
    "mge_settings",
    "numerics_settings",
    "orbit_library_settings",
    "parameter_space_settings",
    "population_data",
    "potential",
    "spatial_binnings",
    "system_attributes",
    "units",
    "weight_solver_settings",
}


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
        {
            "MGEs",
            "kinematic_data",
            "population_data",
            "potential",
            "spatial_binnings",
            "system_attributes",
            "units",
        },
        "configuration",
    )
    # _validate_potential's schema-specific checks (e.g. "ml is invalid for
    # a mass MGE potential") run before validate_configuration_quantities'
    # generic dimension check on purpose: both would reject the same
    # mistakenly-declared ml, but only this order gives the specific
    # message instead of a generic "unit not supported for this parameter"
    # one. Needs only mges, itself independent of everything else here.
    mges = _validate_mges(_required_mapping(config, "MGEs", "configuration"))
    _validate_potential(
        _required_mapping(config, "potential", "configuration"),
        mges,
    )
    validate_configuration_quantities(config)

    _validate_units(_required_mapping(config, "units", "configuration"))
    _validate_cosmological_parameters(
        _required_mapping(config, "cosmological_parameters", "configuration")
    )
    _validate_mge_settings(_required_mapping(config, "mge_settings", "configuration"))
    _validate_numerics_settings(
        _required_mapping(config, "numerics_settings", "configuration")
    )
    _validate_logging_settings(
        _required_mapping(config, "logging_settings", "configuration")
    )
    _validate_system_attributes(
        _required_mapping(config, "system_attributes", "configuration")
    )
    binnings = _validate_spatial_binnings(
        _required_mapping(config, "spatial_binnings", "configuration")
    )
    kinematic_data = _required_mapping(config, "kinematic_data", "configuration")
    kinematic_files = _validate_kinematics(
        kinematic_data,
        binnings,
        mges,
    )
    _validate_population_data(
        _required_mapping(config, "population_data", "configuration"),
        binnings,
        kinematic_files,
    )
    _validate_orbit_library_settings(
        _required_mapping(config, "orbit_library_settings", "configuration")
    )
    _validate_weight_solver_settings(
        _required_mapping(config, "weight_solver_settings", "configuration")
    )
    _validate_parameter_space_settings(
        _required_mapping(config, "parameter_space_settings", "configuration")
    )
    _validate_analysis_settings(
        _required_mapping(config, "analysis_settings", "configuration")
    )
    _validate_io_settings(_required_mapping(config, "io_settings", "configuration"))
    _validate_execution_settings(
        _required_mapping(config, "execution_settings", "configuration")
    )


def _validate_cosmological_parameters(settings: ConfigDict) -> None:
    path = "cosmological_parameters"
    _reject_unknown_keys(settings, {"H"}, path)
    _require_keys(settings, {"H"}, path)
    value = declared_quantity_value(settings["H"], "inverse_time", f"{path}.H")
    _positive_number(value, f"{path}.H.value")


def _validate_units(settings: ConfigDict) -> None:
    path = "units"
    _reject_unknown_keys(settings, {"display", "internal"}, path)
    _require_keys(settings, {"display", "internal"}, path)
    internal = _required_mapping(settings, "internal", path)
    internal_keys = {"angle", "length", "mass", "time"}
    _reject_unknown_keys(internal, internal_keys, f"{path}.internal")
    _require_keys(internal, internal_keys, f"{path}.internal")
    display = _required_mapping(settings, "display", path)
    _reject_unknown_keys(display, internal_keys | {"power", "speed"}, f"{path}.display")
    for section_name, section in (("internal", internal), ("display", display)):
        for key, value in section.items():
            _string(value, f"{path}.{section_name}.{key}")


def _validate_mge_settings(settings: ConfigDict) -> None:
    path = "mge_settings"
    keys = {"intrinsic_mass_quad_order", "projected_mass_quad_order"}
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    for key in keys:
        order = _integer(settings[key], f"{path}.{key}")
        if order <= 0:
            raise ValueError(f"{path}.{key} must be a positive integer.")


def _validate_numerics_settings(settings: ConfigDict) -> None:
    path = "numerics_settings"
    _reject_unknown_keys(
        settings,
        {
            "constraint_error_floors",
            "jax_enable_x64",
            "model_comparison_relative_tolerance",
            "parameter_grid_relative_tolerance",
        },
        path,
    )
    _require_keys(
        settings,
        {
            "constraint_error_floors",
            "jax_enable_x64",
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
    _boolean(settings["jax_enable_x64"], f"{path}.jax_enable_x64")
    floors = _required_mapping(settings, "constraint_error_floors", path)
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
    _reject_unknown_keys(attributes, {"distance", "name"}, path)
    _require_keys(attributes, {"distance", "name"}, path)
    distance = declared_quantity_value(
        attributes["distance"], "length", f"{path}.distance"
    )
    _positive_number(distance, f"{path}.distance.value")
    _string(attributes["name"], f"{path}.name")


def _validate_logging_settings(settings: ConfigDict) -> None:
    path = "logging_settings"
    _reject_unknown_keys(settings, {"console", "file"}, path)
    _require_keys(settings, {"console", "file"}, path)

    file_settings = _required_mapping(settings, "file", path)
    file_path = f"{path}.file"
    _reject_unknown_keys(file_settings, {"directory", "enabled", "level"}, file_path)
    _require_keys(file_settings, {"directory", "enabled", "level"}, file_path)
    _boolean(file_settings["enabled"], f"{file_path}.enabled")
    _logging_level(file_settings["level"], f"{file_path}.level")
    directory = _string(file_settings["directory"], f"{file_path}.directory")
    directory_path = Path(directory)
    if directory_path.is_absolute() or ".." in directory_path.parts:
        raise ValueError(
            f"{file_path}.directory must stay within io_settings.output_directory."
        )

    console_settings = _required_mapping(settings, "console", path)
    console_path = f"{path}.console"
    _reject_unknown_keys(console_settings, {"enabled", "level"}, console_path)
    _require_keys(console_settings, {"enabled", "level"}, console_path)
    _boolean(console_settings["enabled"], f"{console_path}.enabled")
    _logging_level(console_settings["level"], f"{console_path}.level")


def _validate_mges(mges: ConfigDict) -> set[str]:
    """Validate the named Multi-Gaussian Expansion (MGE) file registry."""
    path = "MGEs"
    names: set[str] = set()
    for name, entry in mges.items():
        name = _dynamic_name(name, path)
        entry_path = f"{path}.{name}"
        entry = _mapping(entry, entry_path)
        _reject_unknown_keys(entry, {"file", "major_axis_pa"}, entry_path)
        _require_keys(entry, {"file", "major_axis_pa"}, entry_path)
        _string(entry["file"], f"{entry_path}.file")
        declared_quantity_value(
            entry["major_axis_pa"], "angle", f"{entry_path}.major_axis_pa"
        )
        names.add(name)
    return names


def _validate_spatial_binnings(binnings: ConfigDict) -> set[str]:
    """Collect the names of reusable projected-plane binning definitions.

    Each binning's own fields are validated by `ProjectedBinning` itself
    (see `tnt.spatial_binnings.ProjectedBinning.from_settings`) rather than
    here -- this only collects names so other sections can validate their
    ``spatial_binnings`` references against a known registry.
    """
    path = "spatial_binnings"
    names: set[str] = set()
    for name in binnings:
        names.add(_dynamic_name(name, path))
    return names


def _validate_potential(potential: ConfigDict, mge_names: set[str]) -> None:
    """Validate the named potential components and their MGE references."""
    path = "potential"
    if not potential:
        raise ValueError(f"{path} must contain at least one component.")
    for name, component_value in potential.items():
        name = _dynamic_name(name, path)
        component_path = f"{path}.{name}"
        component = _mapping(component_value, component_path)
        _reject_unknown_keys(
            component,
            {"mge", "parameterization", "parameters", "type"},
            component_path,
        )
        _require_keys(component, {"type"}, component_path)
        component_type = _string(component["type"], f"{component_path}.type")
        parameterization: str | None = None
        if "parameterization" in component:
            parameterization = _string(
                component["parameterization"], f"{component_path}.parameterization"
            )

        if "parameters" in component:
            _validate_parameters(
                _required_mapping(component, "parameters", component_path),
                f"{component_path}.parameters",
                require_nonempty=True,
            )
        else:
            raise ValueError(f"{component_path} is missing required field: parameters.")

        is_mge_potential = is_registered_component_type(component_type)
        if is_mge_potential:
            _require_keys(component, {"mge"}, component_path)
            _validate_registry_reference(
                component["mge"],
                mge_names,
                f"{component_path}.mge",
                "MGEs",
            )
        elif "mge" in component:
            raise ValueError(
                f"{component_path}.mge is only valid for MGE potential types."
            )

        # Exact-name completeness check against the resolved type/
        # parameterization's full parameter set: a registered TNT component's
        # own `_raw_dimensions` (see tnt.potential.registry.register_component),
        # a registered non-native parameterization's raw schema, or -- the
        # common case -- a curated galax class's native constructor kwargs
        # (`tnt.potential.registry._SUPPORTED_GALAX_TYPES`). Every native field
        # must be declared, including ones with a galax constructor default, so
        # the model-table schema is complete and the model reproducible.
        # `parameter_schema_is_known` gates this: an unrecognized type or an
        # unimplemented parameterization is left to `resolve()`'s own, clearer
        # error rather than reported here as "every parameter is extra".
        parameters = component.get("parameters")
        parameter_names = set(parameters) if isinstance(parameters, dict) else set()
        if parameter_schema_is_known(component_type, parameterization):
            required = set(raw_parameter_dimensions(component_type, parameterization))
            schema_label = component_type
            if parameterization is not None:
                schema_label = (
                    f"{component_type} with parameterization {parameterization!r}"
                )
            extra = parameter_names - required
            if extra:
                names = ", ".join(sorted(extra))
                raise ValueError(
                    f"{component_path}.parameters has invalid field(s) for "
                    f"{schema_label}: {names}."
                )
            missing = required - parameter_names
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(
                    f"{component_path}.parameters is missing required "
                    f"field(s): {names}."
                )


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
        parameter = _mapping(parameter_value, parameter_path)
        _reject_unknown_keys(
            parameter,
            {
                "fixed",
                "generator_settings",
                "latex_label",
                "unit",
                "value",
            },
            parameter_path,
        )
        _require_keys(
            parameter,
            {"fixed", "value"},
            parameter_path,
        )
        _boolean(parameter["fixed"], f"{parameter_path}.fixed")
        value = _number(parameter["value"], f"{parameter_path}.value")
        if "latex_label" in parameter:
            _string(parameter["latex_label"], f"{parameter_path}.latex_label")
        if "generator_settings" in parameter:
            _validate_parameter_generator_settings(
                _required_mapping(parameter, "generator_settings", parameter_path),
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
            f"The parameter value at {path.rsplit('.', 1)[0]}.value must lie "
            "within its bounds."
        )
    _positive_number(settings["step"], f"{path}.step")
    _nonnegative_number(settings["minimum_step"], f"{path}.minimum_step")


def _validate_kinematics(
    kinematics: ConfigDict,
    binning_names: set[str],
    mge_names: set[str],
) -> set[Path]:
    path = "kinematic_data"
    data_files: set[Path] = set()
    for name, settings_value in kinematics.items():
        name = _dynamic_name(name, path)
        settings_path = f"{path}.{name}"
        settings = _mapping(settings_value, settings_path)
        _reject_unknown_keys(
            settings,
            {
                "binning",
                "data_file",
                "histogram",
                "maximum_gh_order",
                "mge",
                "observational_errors",
                "type",
                "warning_thresholds",
            },
            settings_path,
        )
        _require_keys(
            settings,
            {"binning", "data_file", "type"},
            settings_path,
        )
        _choice(
            settings["type"],
            set(kinematics_type_names()),
            f"{settings_path}.type",
        )
        filename = _string(settings["data_file"], f"{settings_path}.data_file")
        data_files.add(Path(filename))
        _validate_registry_reference(
            settings["binning"],
            binning_names,
            f"{settings_path}.binning",
            "spatial_binnings",
        )
        if "mge" in settings:
            _validate_registry_reference(
                settings["mge"],
                mge_names,
                f"{settings_path}.mge",
                "MGEs",
            )
        # Type-specific settings are validated when the concrete runtime
        # object is instantiated by tnt.kinematics.build_kinematics. This
        # preparation layer retains only generic schema and reference checks.
    return data_files


def _validate_population_data(
    populations: ConfigDict,
    binning_names: set[str],
    kinematic_files: set[Path],
) -> None:
    """Validate population data sets and their reusable binning references."""
    path = "population_data"
    for name, settings_value in populations.items():
        name = _dynamic_name(name, path)
        settings_path = f"{path}.{name}"
        settings = _mapping(settings_value, settings_path)
        _reject_unknown_keys(settings, {"binning", "data_file"}, settings_path)
        _require_keys(settings, {"binning", "data_file"}, settings_path)
        filename = _string(settings["data_file"], f"{settings_path}.data_file")
        if Path(filename) in kinematic_files:
            raise ValueError(
                f"{settings_path}.data_file must be separate from every "
                "kinematic_data file."
            )
        _validate_registry_reference(
            settings["binning"],
            binning_names,
            f"{settings_path}.binning",
            "spatial_binnings",
        )


def _validate_registry_reference(
    value: Any,
    known_names: set[str],
    path: str,
    registry: str,
) -> None:
    name = _string(value, path)
    if name not in known_names:
        raise ValueError(f"{path} references unknown {registry} entry {name!r}.")


def _validate_orbit_library_settings(settings: ConfigDict) -> None:
    path = "orbit_library_settings"
    keys = {
        "accuracy",
        "dithering",
        "n_stored_timesteps",
        "orbit_sampler",
        "orbital_periods",
        "quad_nph",
        "quad_nr",
        "quad_nth",
        "random_seed",
        "starting_orbit",
    }
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    for key in (
        "quad_nph",
        "quad_nr",
        "quad_nth",
        "n_stored_timesteps",
        "starting_orbit",
    ):
        value = _integer(settings[key], f"{path}.{key}")
        if value <= 0:
            raise ValueError(f"{path}.{key} must be a positive integer.")
    _validate_orbit_sampler(
        _required_mapping(settings, "orbit_sampler", path), f"{path}.orbit_sampler"
    )
    _validate_dithering(
        _required_mapping(settings, "dithering", path), f"{path}.dithering"
    )
    _integer(settings["random_seed"], f"{path}.random_seed")
    _positive_number(settings["orbital_periods"], f"{path}.orbital_periods")
    _positive_number(settings["accuracy"], f"{path}.accuracy")


_ORBIT_SAMPLER_TYPES = {"Grid", "Random"}


def _validate_orbit_sampler(settings: ConfigDict, path: str) -> None:
    """Validate an `orbit_library_settings.orbit_sampler` entry.

    `logrmin`/`logrmax` (the radial log-extent orbits are sampled within)
    are required for every scheme. Only `Grid`'s further fields (`nE`/
    `nI2`/`nI3`) are otherwise known and checked here -- `Random`'s own
    settings are still undecided, mirroring
    `weight_solver_settings.nnls_solver`.
    """
    _require_keys(settings, {"type", "logrmin", "logrmax"}, path)
    sampler_type = _choice(settings["type"], _ORBIT_SAMPLER_TYPES, f"{path}.type")
    logrmin = _number(settings["logrmin"], f"{path}.logrmin")
    logrmax = _number(settings["logrmax"], f"{path}.logrmax")
    if logrmin >= logrmax:
        raise ValueError(f"{path}.logrmin must be less than logrmax.")
    if sampler_type != "Grid":
        return
    keys = {"type", "logrmin", "logrmax", "nE", "nI2", "nI3"}
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    for key in ("nE", "nI3"):
        value = _integer(settings[key], f"{path}.{key}")
        if value <= 0:
            raise ValueError(f"{path}.{key} must be a positive integer.")
    n_i2 = _integer(settings["nI2"], f"{path}.nI2")
    if n_i2 < 4:
        raise ValueError(f"{path}.nI2 must be at least 4.")


_DITHERING_TYPES = {"Cubic"}


def _validate_dithering(settings: ConfigDict, path: str) -> None:
    """Validate an `orbit_library_settings.dithering` entry.

    Only `Cubic`'s field (`n_dither`) is known and checked here; there's no
    other dithering scheme yet.
    """
    _require_keys(settings, {"type"}, path)
    dithering_type = _choice(settings["type"], _DITHERING_TYPES, f"{path}.type")
    if dithering_type != "Cubic":
        return
    keys = {"type", "n_dither"}
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    value = _integer(settings["n_dither"], f"{path}.n_dither")
    if value <= 0:
        raise ValueError(f"{path}.n_dither must be a positive integer.")


def _validate_weight_solver_settings(settings: ConfigDict) -> None:
    path = "weight_solver_settings"
    keys = {
        "lum_intr_rel_err",
        "maxiter_factor",
        "nnls_solver",
        "reattempt_failures",
        "regularisation",
        "sb_proj_rel_err",
        "type",
    }
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    _choice(settings["type"], {"NNLS"}, f"{path}.type")
    # TNT will use a JAX-native NNLS solver. The specific implementation is
    # undecided, so this only checks that a name was given; tighten to a
    # `_choice` over the real options once chosen.
    _string(settings["nnls_solver"], f"{path}.nnls_solver")
    _positive_number(settings["maxiter_factor"], f"{path}.maxiter_factor")
    _nonnegative_number(settings["regularisation"], f"{path}.regularisation")
    for key in ("lum_intr_rel_err", "sb_proj_rel_err"):
        _nonnegative_number(settings[key], f"{path}.{key}")
    reattempt_failures = _boolean(
        settings["reattempt_failures"], f"{path}.reattempt_failures"
    )
    if reattempt_failures:
        raise ValueError(
            f"{path}.reattempt_failures must be false until retry behavior is "
            "implemented."
        )


def _validate_parameter_space_settings(settings: ConfigDict) -> None:
    path = "parameter_space_settings"
    keys = {
        "generator_settings",
        "generator_type",
        "potential_rescalings",
        "stopping_criteria",
        "which_chi2",
    }
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    required_settings_by_type = parameter_generator_required_settings()
    generator_type = _choice(
        settings["generator_type"],
        set(required_settings_by_type),
        f"{path}.generator_type",
    )
    _choice(
        settings["which_chi2"], {"chi2", "kinchi2", "kinmapchi2"}, f"{path}.which_chi2"
    )
    generator = _required_mapping(settings, "generator_settings", path)
    required_generator_keys = required_settings_by_type[generator_type]
    # Not `_reject_unknown_keys`: recursive default-merging can't remove a
    # mapping key, so a user who overrides `generator_type` away from the
    # packaged default's ("GridSearch") still inherits its
    # `generator_settings.delta_chi2_threshold` unless they redeclare that
    # exact key with a real value -- an empty `generator_settings: {}`
    # override doesn't clear it. Keys beyond what this `generator_type`
    # requires are therefore tolerated rather than rejected.
    _require_keys(generator, required_generator_keys, f"{path}.generator_settings")
    if "delta_chi2_threshold" in required_generator_keys:
        _validate_tagged_threshold(
            _required_mapping(
                generator, "delta_chi2_threshold", f"{path}.generator_settings"
            ),
            f"{path}.generator_settings.delta_chi2_threshold",
            {"absolute", "fraction_of_sqrt_2n_observations"},
        )
    stopping = _required_mapping(settings, "stopping_criteria", path)
    _reject_unknown_keys(
        stopping,
        {"minimum_delta_chi2", "n_new_iter", "target_model_count"},
        f"{path}.stopping_criteria",
    )
    _require_keys(
        stopping,
        {"minimum_delta_chi2", "n_new_iter", "target_model_count"},
        f"{path}.stopping_criteria",
    )
    _validate_tagged_threshold(
        _required_mapping(stopping, "minimum_delta_chi2", f"{path}.stopping_criteria"),
        f"{path}.stopping_criteria.minimum_delta_chi2",
        {"absolute", "relative"},
        with_enabled=True,
    )
    for key in ("n_new_iter", "target_model_count"):
        value = _integer(stopping[key], f"{path}.stopping_criteria.{key}")
        if value <= 0:
            raise ValueError(f"{path}.stopping_criteria.{key} must be positive.")

    _validate_potential_rescalings(
        _required_mapping(settings, "potential_rescalings", path),
        f"{path}.potential_rescalings",
    )


def _validate_potential_rescalings(settings: ConfigDict, path: str) -> None:
    keys = {
        "enabled",
        "mass_scale_range",
        "range_count",
        "spacing",
    }
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    _boolean(settings["enabled"], f"{path}.enabled")

    range_count = _integer(settings["range_count"], f"{path}.range_count")
    if range_count <= 0:
        raise ValueError(f"{path}.range_count must be positive.")

    mass_scale_range = _required_mapping(settings, "mass_scale_range", path)
    range_path = f"{path}.mass_scale_range"
    _reject_unknown_keys(mass_scale_range, {"maximum", "minimum"}, range_path)
    _require_keys(mass_scale_range, {"maximum", "minimum"}, range_path)
    minimum = _positive_number(mass_scale_range["minimum"], f"{range_path}.minimum")
    maximum = _positive_number(mass_scale_range["maximum"], f"{range_path}.maximum")
    if minimum > maximum:
        raise ValueError(f"{range_path}.minimum must not exceed maximum.")
    _choice(settings["spacing"], {"linear", "logarithmic"}, f"{path}.spacing")


def _validate_tagged_threshold(
    threshold: ConfigDict,
    path: str,
    allowed_modes: set[str],
    *,
    with_enabled: bool = False,
) -> None:
    keys = {"mode", "value"}
    if with_enabled:
        keys.add("enabled")
    _reject_unknown_keys(threshold, keys, path)
    _require_keys(threshold, keys, path)
    if with_enabled:
        _boolean(threshold["enabled"], f"{path}.enabled")
    _choice(threshold["mode"], allowed_modes, f"{path}.mode")
    _nonnegative_number(threshold["value"], f"{path}.value")


def _validate_analysis_settings(settings: ConfigDict) -> None:
    path = "analysis_settings"
    _reject_unknown_keys(settings, {"kinematic_moments", "orbit_decomposition"}, path)
    _require_keys(settings, {"kinematic_moments", "orbit_decomposition"}, path)
    decomposition = _required_mapping(settings, "orbit_decomposition", path)
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
    thresholds = _required_mapping(
        decomposition, "circularity_thresholds", decomposition_path
    )
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
            f"{threshold_path} values must be strictly increasing from "
            "counter-rotating cold to cold."
        )
    moments = _required_mapping(settings, "kinematic_moments", path)
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
        _string(settings[key], f"{path}.{key}")


def _validate_execution_settings(settings: ConfigDict) -> None:
    path = "execution_settings"
    keys = {
        "model_processing_order",
        "orbit_workers",
        "weight_workers",
    }
    _reject_unknown_keys(settings, keys, path)
    _require_keys(settings, keys, path)
    for key in ("orbit_workers", "weight_workers"):
        _worker_count(settings[key], f"{path}.{key}")
    _choice(
        settings["model_processing_order"],
        {"model_by_model", "stage_by_stage"},
        f"{path}.model_processing_order",
    )


def _worker_count(value: Any, path: str) -> None:
    if value == "all_available":
        return
    workers = _integer(value, path)
    if workers <= 0:
        raise ValueError(f"{path} must be a positive integer or 'all_available'.")


def _dynamic_name(value: Any, parent: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Names below {parent} must be non-empty strings.")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean.")
    return value


def _choice(value: Any, allowed: set[str], path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string.")
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{path} must be one of: {choices}; got {value!r}.")
    return value


def _logging_level(value: Any, path: str) -> str:
    return _choice(
        value,
        {"CRITICAL", "DEBUG", "ERROR", "INFO", "WARNING"},
        path,
    )
