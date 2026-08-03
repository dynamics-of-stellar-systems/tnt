from pathlib import Path

import astropy.units as au
import jax.numpy as jnp
import numpy as np
import pytest
import unxt as u
from astropy.table import QTable

from tnt.kinematics import (
    BayesLOSVD,
    GaussHermite,
    Histogram2D,
    ProperMotions,
    build_kinematics,
)


def _unit_system() -> u.AbstractUnitSystem:
    return u.unitsystem("kpc", "Myr", "Msun", "rad", "Lsun")


def _common_settings(data_file: Path, kind: str) -> dict[str, object]:
    return {
        "type": kind,
        "data_file": data_file.name,
        "binning": "observed",
    }


def _write_gauss_hermite(path: Path) -> None:
    table = QTable()
    table["vbin_id"] = [1, 2]
    table["v"] = [100.0, -50.0] * au.km / au.s
    table["dv"] = [3.0, 4.0] * au.km / au.s
    table["sigma"] = [120.0, 80.0] * au.km / au.s
    table["dsigma"] = [5.0, 6.0] * au.km / au.s
    table["h3"] = [0.1, -0.1]
    table["dh3"] = [0.02, 0.03]
    table["h4"] = [0.01, -0.02]
    table["dh4"] = [0.04, 0.05]
    table.write(path, format="ascii.ecsv")


def test_build_gauss_hermite_converts_units_and_applies_systematics(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "gh.ecsv"
    _write_gauss_hermite(data_file)
    settings = _common_settings(data_file, "gauss_hermite")
    settings.update(
        {
            "maximum_gh_order": 4,
            "observational_errors": {
                "systematic_uncertainties": {
                    "v": 1.0,
                    "sigma": 2.0,
                    "h3": 0.01,
                    "h4": 0.02,
                }
            },
            "histogram": {
                "sigma_extent": 3.0,
                "bin_width_sigma_fraction": 0.1,
                "center": 0.0,
            },
        }
    )

    result = build_kinematics(
        {"gh": settings},
        tmp_path,
        _unit_system(),
        {"observed": "shared-binning"},
    )["gh"]

    assert isinstance(result, GaussHermite)
    assert result.binning == "shared-binning"
    assert result.velocity.unit == u.unit("kpc / Myr")
    assert result.coefficients.shape == (2, 2)
    assert result.histogram.bins % 2 == 1
    km_s_to_internal = au.Unit("km / s").to(au.Unit("kpc / Myr"))
    expected_dv = np.sqrt(np.array([3.0, 4.0]) ** 2 + (1.0 / km_s_to_internal) ** 2)
    assert jnp.allclose(
        result.velocity_uncertainty.ustrip("km / s"), expected_dv
    )
    values, uncertainties = result.observed_values_and_uncertainties()
    assert values.shape == uncertainties.shape == (2, 4)


def test_gauss_hermite_adds_missing_higher_order_with_systematic(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "gh.ecsv"
    _write_gauss_hermite(data_file)
    settings = _common_settings(data_file, "gauss_hermite")
    settings.update(
        {
            "maximum_gh_order": 5,
            "observational_errors": {
                "systematic_uncertainties": {
                    "v": 0.0,
                    "sigma": 0.0,
                    "h3": 0.0,
                    "h4": 0.0,
                    "h5": 0.03,
                }
            },
            "histogram": {"width": 1000.0, "center": 0.0, "bins": 101},
        }
    )

    result = build_kinematics(
        {"gh": settings}, tmp_path, _unit_system(), {"observed": object()}
    )["gh"]

    assert jnp.allclose(result.coefficients[:, 2], 0.0)
    assert jnp.allclose(result.coefficient_uncertainties[:, 2], 0.03)


def test_gauss_hermite_rejects_incomplete_systematics_at_construction(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "gh.ecsv"
    _write_gauss_hermite(data_file)
    settings = _common_settings(data_file, "gauss_hermite")
    settings.update(
        {
            "maximum_gh_order": 5,
            "observational_errors": {
                "systematic_uncertainties": {
                    "v": 0.0,
                    "sigma": 0.0,
                    "h3": 0.0,
                    "h4": 0.0,
                }
            },
            "histogram": {"width": 1000.0, "center": 0.0, "bins": 101},
        }
    )

    with pytest.raises(ValueError, match=r"missing required field\(s\): h5"):
        build_kinematics(
            {"gh": settings}, tmp_path, _unit_system(), {"observed": object()}
        )


def test_gauss_hermite_rejects_even_histogram_at_construction(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "gh.ecsv"
    _write_gauss_hermite(data_file)
    settings = _common_settings(data_file, "gauss_hermite")
    settings.update(
        {
            "maximum_gh_order": 4,
            "observational_errors": {
                "systematic_uncertainties": {
                    "v": 0.0,
                    "sigma": 0.0,
                    "h3": 0.0,
                    "h4": 0.0,
                }
            },
            "histogram": {"width": 1000.0, "center": 0.0, "bins": 100},
        }
    )

    with pytest.raises(ValueError, match="bins must be a positive odd integer"):
        build_kinematics(
            {"gh": settings}, tmp_path, _unit_system(), {"observed": object()}
        )


def _write_bayes_losvd(path: Path) -> None:
    table = QTable()
    table["binID_dynamite"] = [1, 2]
    table["bin_flux"] = [2.0, 1.0]
    losvds = np.array([[0.1, 0.6, 0.3], [0.4, 0.5, 0.1]])
    for index in range(3):
        table[f"losvd_{index}"] = losvds[:, index]
        table[f"dlosvd_{index}"] = [0.02, 0.03]
    table.meta = {
        "vcent": [-100.0, 0.0, 100.0],
        "dv": 100.0,
        "velocity_unit": "km / s",
    }
    table.write(path, format="ascii.ecsv")


def test_build_bayes_losvd_centers_systemic_velocity(tmp_path: Path) -> None:
    data_file = tmp_path / "bayes.ecsv"
    _write_bayes_losvd(data_file)
    settings = _common_settings(data_file, "bayes_losvd")
    settings["histogram"] = {
        "width_scale": 1.0,
        "oversampling_factor": 2.0,
        "center": 0.0,
        "systemic_velocity": "flux_weighted",
    }

    result = build_kinematics(
        {"bayes": settings},
        tmp_path,
        _unit_system(),
        {"observed": "shared-binning"},
    )["bayes"]

    assert isinstance(result, BayesLOSVD)
    assert result.losvd.shape == (2, 3)
    assert result.velocity_centers.unit == u.unit("kpc / Myr")
    flux_weighted_mean = jnp.sum(
        result.bin_flux * result.mean_velocity.ustrip(result.mean_velocity.unit)
    )
    assert jnp.isclose(flux_weighted_mean, 0.0, atol=1e-6)
    assert result.histogram.bins % 2 == 1


def _write_proper_motions(path: Path) -> None:
    distribution = np.ones((2, 3, 3), dtype=float)
    distribution[0, 1, 1] = 4.0
    np.savez(
        path,
        PM_2dhist=distribution,
        PM_2dhist_sigma=np.full((2, 3, 3), 0.2),
        binID_dynamite=np.array([1, 2]),
        nstarbin=np.array([20, 30]),
        velocity_unit=np.array("km / s"),
        vxrange=np.array(150.0),
        vyrange=np.array(120.0),
    )


def test_build_proper_motions_normalizes_and_scales_errors(tmp_path: Path) -> None:
    data_file = tmp_path / "pm.npz"
    _write_proper_motions(data_file)
    settings = _common_settings(data_file, "proper_motions")
    settings.update(
        {
            "observational_errors": {"variance_scale": 4.0},
            "warning_thresholds": {
                "max_bin_width_sigma_ratio": 10.0,
                "min_histogram_width_sigma_ratio": 0.1,
            },
        }
    )

    result = build_kinematics(
        {"pm": settings}, tmp_path, _unit_system(), {"observed": object()}
    )["pm"]

    assert isinstance(result, ProperMotions)
    assert isinstance(result.histogram, Histogram2D)
    assert jnp.allclose(jnp.sum(result.distribution, axis=(1, 2)), 1.0)
    assert jnp.allclose(result.normalization, jnp.array([12.0, 9.0]))
    assert jnp.isclose(result.uncertainty[0, 0, 0], 0.4 / 12.0)
    values, uncertainties = result.observed_values_and_uncertainties()
    assert values.shape == uncertainties.shape == (2, 9)


def test_proper_motions_rejects_variance_scale_at_construction(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "pm.npz"
    _write_proper_motions(data_file)
    settings = _common_settings(data_file, "proper_motions")
    settings.update(
        {
            "observational_errors": {"variance_scale": 0.0},
            "warning_thresholds": {
                "max_bin_width_sigma_ratio": 0.25,
                "min_histogram_width_sigma_ratio": 5.0,
            },
        }
    )

    with pytest.raises(ValueError, match="variance_scale must be greater than zero"):
        build_kinematics(
            {"pm": settings}, tmp_path, _unit_system(), {"observed": object()}
        )


def test_build_kinematics_rejects_unknown_binning_before_reading_file(
    tmp_path: Path,
) -> None:
    settings = _common_settings(tmp_path / "missing.ecsv", "gauss_hermite")

    with pytest.raises(ValueError, match="unknown spatial_binnings entry 'observed'"):
        build_kinematics({"gh": settings}, tmp_path, _unit_system(), {})
