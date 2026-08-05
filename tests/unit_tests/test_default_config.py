from importlib.resources import files


def test_default_config_is_packaged() -> None:
    default_config = files("tnt.defaults").joinpath("default_config.yaml")
    config_text = default_config.read_text(encoding="utf-8")

    assert default_config.is_file()
    assert "orbit_library_settings:" in config_text
    assert '  H0: {value: 70.0, unit: "km / (s Mpc)"}' in config_text
    assert "units:" in config_text
    assert (
        "mge_settings:\n  intrinsic_mass_quad_order: 10\n"
        "  projected_mass_quad_order: 10"
    ) in config_text
    assert "numerics_settings:" in config_text
    assert "  model_comparison_relative_tolerance: 1.0e-10" in config_text
    assert "  parameter_grid_relative_tolerance: 1.0e-6" in config_text
    assert "    total_mass: 1.0e-8" in config_text
    assert "    intrinsic_mass: 1.0e-16" in config_text
    assert "dynamic_object_defaults:" in config_text
    assert "  potential:\n    include: true" in config_text
    assert "  parameter:\n    fixed: false\n    logarithmic: false" in config_text
    assert "MGEs: {}" in config_text
    assert "spatial_binnings: {}" in config_text
    assert "potential: {}" in config_text
    assert "kinematic_data: {}" in config_text
    assert "population_data: {}" in config_text
    assert "kinematics_type_defaults:" in config_text
    assert '        v: {value: 0.0, unit: "km / s"}' in config_text
    assert '        sigma: {value: 0.0, unit: "km / s"}' in config_text
    assert '      center: {value: 0.0, unit: "km / s"}' in config_text
    assert "      sigma_extent: 3.0" in config_text
    assert "      bin_width_sigma_fraction: 0.1" in config_text
    assert "      width_scale: 1.0" in config_text
    assert "      oversampling_factor: 1.0" in config_text
    assert '      systemic_velocity: "flux_weighted"' in config_text
    assert "      max_bin_width_sigma_ratio: 0.25" in config_text
    assert "      min_histogram_width_sigma_ratio: 5.0" in config_text
    assert "analysis_settings:" in config_text
    assert "      cold_min: 0.8" in config_text
    assert "      warm_min: 0.25" in config_text
    assert "      counter_rotating_warm_max: -0.25" in config_text
    assert "      counter_rotating_cold_max: -0.8" in config_text
    assert '    component_naming: "bulge_disk"' in config_text
    assert "    cache: true" in config_text
    assert "    write_component_weights: false" in config_text
    assert '    velocity_dispersion_method: "gaussian_fit"' in config_text
    assert "  logrmin:" not in config_text
    assert "  logrmax:" not in config_text
    assert "  random_seed: -1" in config_text
    assert "    delta_chi2_threshold:" in config_text
    assert '      mode: "fraction_of_sqrt_2n_observations"' in config_text
    assert "    minimum_delta_chi2:" in config_text
    assert '      mode: "absolute"' in config_text
    assert "    min_delta_chi2_abs:" not in config_text
    assert "    min_delta_chi2_rel:" not in config_text
    assert "    threshold_del_chi2_abs:" not in config_text
    assert "    threshold_del_chi2_as_frac_of_sqrt2nobs:" not in config_text
    assert "  potential_rescalings:" in config_text
    assert "    enabled: false" in config_text
    assert "    range_count: 10" in config_text
    assert '    spacing: "logarithmic"' in config_text
    assert '  model_processing_order: "model_by_model"' in config_text
    assert "  orbit_family_integration_in_parallel: false" in config_text
    assert "  model_strategy:" not in config_text
    assert "  orbit_libraries_in_parallel:" not in config_text
    assert "counter_rotating_orbit_cut" not in config_text
