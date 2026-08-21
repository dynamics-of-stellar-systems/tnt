"""Build population data from the realistic resolved example configuration."""

from pathlib import Path

import jax.numpy as jnp

from tnt import Configuration
from tnt.populations import Populations, build_populations
from tnt.spatial_binnings import build_spatial_binnings


def test_build_populations_from_resolved_configuration(
    example_configuration_path: Path,
    tmp_path: Path,
) -> None:
    config = Configuration().read(example_configuration_path, workspace_root=tmp_path)
    resolved = config.as_dict()
    input_directory = resolved["io_settings"]["input_directory"]
    unit_system = config.unit_systems.internal
    binnings = build_spatial_binnings(
        resolved["spatial_binnings"],
        input_directory,
        unit_system,
        resolved["mge_settings"]["projected_mass_quad_order"],
    )

    populations = build_populations(
        resolved["population_data"],
        input_directory,
        unit_system,
        binnings,
    )

    observed = populations["stellar_populations"]
    assert isinstance(observed, Populations)
    assert observed.binning is binnings["kinset1_binning"]
    assert observed.property_names == ("age", "metallicity")
    assert observed.n_spatial_bins == binnings["kinset1_binning"].n_bins == 152
    age, age_uncertainty = observed.values_and_uncertainties("age")
    assert jnp.allclose(age.ustrip("Myr"), jnp.full(152, 10000.0))
    assert jnp.allclose(age_uncertainty.ustrip("Myr"), jnp.full(152, 500.0))
