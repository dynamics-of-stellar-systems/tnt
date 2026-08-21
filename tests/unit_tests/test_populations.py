from pathlib import Path

import astropy.units as au
import jax
import jax.numpy as jnp
import pytest
import unxt as u
from astropy.table import QTable

from tnt.populations import Populations, build_populations
from tnt.spatial_binnings import ProjectedBinning


def _unit_system() -> u.AbstractUnitSystem:
    return u.unitsystem("kpc", "Myr", "Msun", "rad", "Lsun")


def _binning() -> ProjectedBinning:
    return ProjectedBinning(
        min_x=u.Quantity(0.0, "rad"),
        min_y=u.Quantity(0.0, "rad"),
        x_extent=u.Quantity(1.0, "rad"),
        y_extent=u.Quantity(1.0, "rad"),
        PA=u.Quantity(0.0, "rad"),
        bins=jnp.array([[1, 2], [0, 2]]),
        quad_order=2,
    )


def _settings(data_file: Path) -> dict[str, str]:
    return {"data_file": data_file.name, "binning": "observed"}


def _write_populations(path: Path) -> None:
    table = QTable()
    table["bin_id"] = [1, 2]
    table["age"] = [10.0, 8.0] * au.Gyr
    table["dage"] = [500.0, 750.0] * au.Myr
    table["metallicity"] = [0.1, -0.2]
    table["dmetallicity"] = [0.01, 0.02]
    table.write(path, format="ascii.ecsv")


def test_build_populations_reads_pairs_and_converts_units(tmp_path: Path) -> None:
    data_file = tmp_path / "populations.ecsv"
    _write_populations(data_file)
    binning = _binning()

    result = build_populations(
        {"stars": _settings(data_file)},
        tmp_path,
        _unit_system(),
        {"observed": binning},
    )["stars"]

    assert isinstance(result, Populations)
    assert result.name == "stars"
    assert result.data_file == data_file
    assert result.binning is binning
    assert result.n_spatial_bins == 2
    assert result.n_properties == 2
    assert result.property_names == ("age", "metallicity")
    age, age_uncertainty = result.values_and_uncertainties("age")
    assert age.unit == u.unit("Myr")
    assert jnp.allclose(age.ustrip("Myr"), jnp.array([10000.0, 8000.0]))
    assert jnp.allclose(age_uncertainty.ustrip("Myr"), jnp.array([500.0, 750.0]))
    metallicity, metallicity_uncertainty = result.values_and_uncertainties(
        "metallicity"
    )
    assert metallicity.unit == u.unit("")
    assert jnp.allclose(metallicity.ustrip(""), jnp.array([0.1, -0.2]))
    assert jnp.allclose(metallicity_uncertainty.ustrip(""), jnp.array([0.01, 0.02]))
    assert jax.tree_util.tree_leaves(result)


def test_build_populations_without_entries_returns_empty_mapping(
    tmp_path: Path,
) -> None:
    assert build_populations({}, tmp_path, _unit_system(), {}) == {}


def test_populations_reject_unknown_property_lookup(tmp_path: Path) -> None:
    data_file = tmp_path / "populations.ecsv"
    _write_populations(data_file)
    result = build_populations(
        {"stars": _settings(data_file)},
        tmp_path,
        _unit_system(),
        {"observed": _binning()},
    )["stars"]

    with pytest.raises(KeyError, match="Unknown population property 'colour'"):
        result.values_and_uncertainties("colour")


def test_build_populations_rejects_unknown_binning_before_reading_file(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "missing.ecsv")

    with pytest.raises(ValueError, match="unknown spatial_binnings entry 'observed'"):
        build_populations({"stars": settings}, tmp_path, _unit_system(), {})


def test_build_populations_rejects_non_projected_binning(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "missing.ecsv")

    with pytest.raises(TypeError, match="must resolve to ProjectedBinning"):
        build_populations(
            {"stars": settings},
            tmp_path,
            _unit_system(),
            {"observed": "not-a-binning"},  # type: ignore[dict-item]
        )


def test_build_populations_rejects_mge_setting(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "missing.ecsv")
    settings["mge"] = "light"

    with pytest.raises(ValueError, match="unknown field.*mge"):
        build_populations(
            {"stars": settings},
            tmp_path,
            _unit_system(),
            {"observed": _binning()},
        )


def test_populations_require_value_uncertainty_pairs(tmp_path: Path) -> None:
    data_file = tmp_path / "populations.ecsv"
    table = QTable({"bin_id": [1, 2], "age": [10.0, 8.0]})
    table.write(data_file, format="ascii.ecsv")

    with pytest.raises(ValueError, match="unpaired column.*age"):
        build_populations(
            {"stars": _settings(data_file)},
            tmp_path,
            _unit_system(),
            {"observed": _binning()},
        )


def test_populations_require_equivalent_pair_units(tmp_path: Path) -> None:
    data_file = tmp_path / "populations.ecsv"
    table = QTable()
    table["bin_id"] = [1, 2]
    table["age"] = [10.0, 8.0] * au.Gyr
    table["dage"] = [1.0, 1.0] * au.kpc
    table.write(data_file, format="ascii.ecsv")

    with pytest.raises(au.UnitConversionError, match="must have equivalent units"):
        build_populations(
            {"stars": _settings(data_file)},
            tmp_path,
            _unit_system(),
            {"observed": _binning()},
        )


def test_populations_require_positive_uncertainties(tmp_path: Path) -> None:
    data_file = tmp_path / "populations.ecsv"
    table = QTable(
        {"bin_id": [1, 2], "metallicity": [0.1, -0.2], "dmetallicity": [0.0, 0.1]}
    )
    table.write(data_file, format="ascii.ecsv")

    with pytest.raises(ValueError, match="dmetallicity must contain only positive"):
        build_populations(
            {"stars": _settings(data_file)},
            tmp_path,
            _unit_system(),
            {"observed": _binning()},
        )


def test_populations_reject_duplicate_bin_ids(tmp_path: Path) -> None:
    data_file = tmp_path / "populations.ecsv"
    table = QTable(
        {"bin_id": [1, 1], "metallicity": [0.1, -0.2], "dmetallicity": [0.1, 0.1]}
    )
    table.write(data_file, format="ascii.ecsv")

    with pytest.raises(ValueError, match="spatial bin IDs must be positive and unique"):
        build_populations(
            {"stars": _settings(data_file)},
            tmp_path,
            _unit_system(),
            {"observed": _binning()},
        )


def test_populations_reject_bin_ids_absent_from_binning(tmp_path: Path) -> None:
    data_file = tmp_path / "populations.ecsv"
    table = QTable(
        {"bin_id": [1, 3], "metallicity": [0.1, -0.2], "dmetallicity": [0.1, 0.1]}
    )
    table.write(data_file, format="ascii.ecsv")

    with pytest.raises(ValueError, match="absent from the referenced binning: 3"):
        build_populations(
            {"stars": _settings(data_file)},
            tmp_path,
            _unit_system(),
            {"observed": _binning()},
        )
