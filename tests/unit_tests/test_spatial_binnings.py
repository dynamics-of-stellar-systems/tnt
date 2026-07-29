import numpy as np
import pytest
import unxt as u

from tnt.spatial_binnings import ProjectedBinning, build_spatial_binnings

_QUAD_ORDER = 3


def _internal_unit_system() -> u.AbstractUnitSystem:
    return u.unitsystem("kpc", "Myr", "Msun", "rad", "Lsun")


def _settings(**overrides):
    settings = {
        "min_x": {"value": -1.0, "unit": "rad"},
        "min_y": {"value": -2.0, "unit": "rad"},
        "x_extent": {"value": 3.0, "unit": "rad"},
        "y_extent": {"value": 4.0, "unit": "rad"},
        "PA": {"value": 0.5, "unit": "rad"},
        "bins_file": "bins.npy",
    }
    settings.update(overrides)
    return settings


def test_build_spatial_binnings_reads_each_named_binning(tmp_path):
    bins = np.array([[0, 1], [2, 0], [1, 1]])
    np.save(tmp_path / "bins.npy", bins)

    binnings = build_spatial_binnings(
        {"observed": _settings()}, tmp_path, _internal_unit_system(), _QUAD_ORDER
    )

    binning = binnings["observed"]
    assert isinstance(binning, ProjectedBinning)
    assert binning.min_x == u.Quantity(-1.0, "rad")
    assert binning.min_y == u.Quantity(-2.0, "rad")
    assert binning.x_extent == u.Quantity(3.0, "rad")
    assert binning.y_extent == u.Quantity(4.0, "rad")
    assert binning.PA == u.Quantity(0.5, "rad")
    assert np.array_equal(binning.bins, bins)


def test_build_spatial_binnings_without_entries_returns_empty_dict(tmp_path):
    assert (
        build_spatial_binnings({}, tmp_path, _internal_unit_system(), _QUAD_ORDER) == {}
    )


def test_from_settings_converts_declared_units(tmp_path):
    bins = np.zeros((2, 2), dtype=int)
    settings = _settings(
        min_x={"value": -3600.0, "unit": "arcsec"},
        PA={"value": 90.0, "unit": "deg"},
    )

    binning = ProjectedBinning.from_settings(
        settings, bins, _internal_unit_system(), _QUAD_ORDER
    )

    assert binning.min_x.unit == u.unit("rad")
    assert binning.min_x.ustrip("rad") == pytest.approx(-3600.0 * np.pi / 648000)
    assert binning.PA.ustrip("rad") == pytest.approx(np.pi / 2)


def test_from_settings_rejects_non_angle_unit(tmp_path):
    bins = np.zeros((2, 2), dtype=int)
    settings = _settings(x_extent={"value": 1.0, "unit": "kpc"})

    with pytest.raises(ValueError, match="not convertible"):
        ProjectedBinning.from_settings(
            settings, bins, _internal_unit_system(), _QUAD_ORDER
        )


def test_from_settings_rejects_unparseable_unit():
    bins = np.zeros((2, 2), dtype=int)
    settings = _settings(PA={"value": 1.0, "unit": "notaunit"})

    with pytest.raises(ValueError, match="did not parse as unit"):
        ProjectedBinning.from_settings(
            settings, bins, _internal_unit_system(), _QUAD_ORDER
        )


def test_from_settings_rejects_missing_field():
    bins = np.zeros((2, 2), dtype=int)
    settings = _settings()
    del settings["PA"]

    with pytest.raises(ValueError, match="missing required field: PA"):
        ProjectedBinning.from_settings(
            settings, bins, _internal_unit_system(), _QUAD_ORDER
        )


@pytest.mark.parametrize(
    "malformed",
    [
        {"value": 1.0},
        {"unit": "rad"},
        {"value": 1.0, "unit": "rad", "extra": 1},
        1.0,
    ],
)
def test_from_settings_rejects_malformed_quantity_mapping(malformed):
    bins = np.zeros((2, 2), dtype=int)
    settings = _settings(PA=malformed)

    with pytest.raises(ValueError, match="must be a mapping with exactly"):
        ProjectedBinning.from_settings(
            settings, bins, _internal_unit_system(), _QUAD_ORDER
        )


def test_from_settings_rejects_non_numeric_value():
    bins = np.zeros((2, 2), dtype=int)
    settings = _settings(PA={"value": "not a number", "unit": "rad"})

    with pytest.raises(TypeError, match="PA.value must be a number"):
        ProjectedBinning.from_settings(
            settings, bins, _internal_unit_system(), _QUAD_ORDER
        )


def test_from_settings_rejects_non_finite_value():
    bins = np.zeros((2, 2), dtype=int)
    settings = _settings(PA={"value": float("nan"), "unit": "rad"})

    with pytest.raises(ValueError, match="PA.value must be finite"):
        ProjectedBinning.from_settings(
            settings, bins, _internal_unit_system(), _QUAD_ORDER
        )


@pytest.mark.parametrize("key", ["x_extent", "y_extent"])
@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_from_settings_rejects_non_positive_extent(key, bad_value):
    bins = np.zeros((2, 2), dtype=int)
    settings = _settings(**{key: {"value": bad_value, "unit": "rad"}})

    with pytest.raises(ValueError, match=f"{key} must be greater than zero"):
        ProjectedBinning.from_settings(
            settings, bins, _internal_unit_system(), _QUAD_ORDER
        )


def test_from_settings_rejects_non_2d_bins():
    bins = np.zeros((2, 2, 2), dtype=int)

    with pytest.raises(ValueError, match="must be a 2D"):
        ProjectedBinning.from_settings(
            _settings(), bins, _internal_unit_system(), _QUAD_ORDER
        )


def test_from_settings_rejects_non_integer_bins():
    bins = np.zeros((2, 2), dtype=float)

    with pytest.raises(TypeError, match="integer dtype"):
        ProjectedBinning.from_settings(
            _settings(), bins, _internal_unit_system(), _QUAD_ORDER
        )


def test_from_settings_rejects_negative_bin_ids():
    bins = np.array([[0, -1], [1, 0]])

    with pytest.raises(ValueError, match="negative bin ID"):
        ProjectedBinning.from_settings(
            _settings(), bins, _internal_unit_system(), _QUAD_ORDER
        )


@pytest.mark.parametrize(
    ("bins_shape", "expected_npix_x", "expected_npix_y"),
    [((5, 3), 5, 3), ((2, 7), 2, 7)],
)
def test_projected_binning_infers_npix_from_bins_shape(
    tmp_path, bins_shape, expected_npix_x, expected_npix_y
):
    bins = np.zeros(bins_shape, dtype=int)
    np.save(tmp_path / "bins.npy", bins)

    binnings = build_spatial_binnings(
        {"observed": _settings()}, tmp_path, _internal_unit_system(), _QUAD_ORDER
    )

    binning = binnings["observed"]
    assert binning.npix_x == expected_npix_x
    assert binning.npix_y == expected_npix_y


def test_angular_to_physical_converts_spatial_fields():
    bins = np.zeros((2, 2), dtype=int)
    binning = ProjectedBinning.from_settings(
        _settings(), bins, _internal_unit_system(), _QUAD_ORDER
    )
    distance = u.Quantity(30.5, "Mpc")

    physical = binning.angular_to_physical(distance)

    assert physical.min_x.unit == u.unit("Mpc")
    assert physical.min_y.unit == u.unit("Mpc")
    assert physical.x_extent.unit == u.unit("Mpc")
    assert physical.y_extent.unit == u.unit("Mpc")
    for angular, converted in (
        (binning.min_x, physical.min_x),
        (binning.min_y, physical.min_y),
        (binning.x_extent, physical.x_extent),
        (binning.y_extent, physical.y_extent),
    ):
        assert converted.ustrip("Mpc") == pytest.approx(
            distance.ustrip("Mpc") * angular.ustrip("rad")
        )


def test_angular_to_physical_leaves_pa_and_bins_unchanged():
    bins = np.array([[0, 1], [2, 0]])
    binning = ProjectedBinning.from_settings(
        _settings(), bins, _internal_unit_system(), _QUAD_ORDER
    )
    distance = u.Quantity(30.5, "Mpc")

    physical = binning.angular_to_physical(distance)

    assert physical.PA == binning.PA
    assert np.array_equal(physical.bins, binning.bins)
