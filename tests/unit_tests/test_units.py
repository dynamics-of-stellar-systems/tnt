import ast
from pathlib import Path

import pytest
import unxt as u

import tnt.units
from tnt.units import (
    build_unit_systems,
    declared_quantity,
    validate_dimension,
)


def _unit_settings() -> dict:
    return {
        "internal": {
            "length": "kpc",
            "time": "Myr",
            "mass": "Msun",
            "angle": "rad",
        },
        "display": {
            "angle": "arcsec",
            "speed": "km / s",
        },
    }


def test_build_unit_systems_inherits_internal_display_units() -> None:
    systems = build_unit_systems(_unit_settings())

    assert systems.internal[u.dimension("length")] == u.unit("kpc")
    assert systems.display[u.dimension("length")] == u.unit("kpc")
    assert systems.display[u.dimension("angle")] == u.unit("arcsec")
    assert systems.display[u.dimension("speed")] == u.unit("km / s")


@pytest.mark.parametrize(
    ("settings_update", "error"),
    [
        ({"internal": {"length": "Myr"}}, r"units\.internal\.length"),
        ({"display": {"speed": "not_a_unit"}}, r"units\.display\.speed"),
    ],
)
def test_build_unit_systems_rejects_invalid_units(
    settings_update: dict, error: str
) -> None:
    settings = _unit_settings()
    for section, values in settings_update.items():
        settings[section].update(values)

    with pytest.raises(ValueError, match=error):
        build_unit_systems(settings)


def test_build_unit_systems_requires_every_internal_dimension() -> None:
    settings = _unit_settings()
    del settings["internal"]["angle"]

    with pytest.raises(ValueError, match=r"units\.internal.*angle"):
        build_unit_systems(settings)


def test_build_unit_systems_rejects_power_in_internal() -> None:
    settings = _unit_settings()
    settings["internal"]["power"] = "Lsun"

    with pytest.raises(ValueError, match=r"units\.internal.*power"):
        build_unit_systems(settings)


def test_build_unit_systems_accepts_power_display_override() -> None:
    settings = _unit_settings()
    settings["display"]["power"] = "erg / s"

    systems = build_unit_systems(settings)

    assert systems.display[u.dimension("power")] == u.unit("erg / s")


def test_declared_quantity_keeps_its_declared_unit() -> None:
    quantity = declared_quantity(
        {"value": 2.5, "unit": "Mpc"},
        "length",
        "system_attributes.distance",
    )

    assert quantity.unit == u.unit("Mpc")
    assert quantity.ustrip("Mpc") == pytest.approx(2.5)


def test_declared_quantity_rejects_bare_number() -> None:
    with pytest.raises(TypeError, match="must state their unit explicitly"):
        declared_quantity(2.5, "length", "system_attributes.distance")


def test_declared_quantity_rejects_sequence_shorthand() -> None:
    with pytest.raises(TypeError, match="mapping containing value and unit"):
        declared_quantity(
            [2.5, "Mpc"], "length", "system_attributes.distance"
        )


def test_validate_dimension_accepts_equivalent_unit() -> None:
    validate_dimension(u.unit("km / s"), "speed", "kinematic_data.x")


def test_validate_dimension_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="must describe speed"):
        validate_dimension(u.unit("kpc"), "speed", "kinematic_data.x")


def test_units_module_imports_no_runtime_family() -> None:
    """`tnt.units` is a low-level primitive: it must not import a TNT
    runtime-family package.

    `tnt.mge` / `tnt.kinematics` / `tnt.spatial_binnings` / `tnt.potential`
    all import `tnt.units`, so an import the other way -- at module level or
    lazily inside a function -- is a cycle. It is the reason whole-config
    quantity validation lives in `tnt.configuration.validation`, not here.
    Checked by parsing this module's own import statements rather than
    `sys.modules`, since importing any `tnt` submodule first runs
    `tnt/__init__.py`, which pulls in the whole package.
    """
    forbidden = ("tnt.potential", "tnt.mge", "tnt.kinematics", "tnt.spatial_binnings")
    tree = ast.parse(Path(tnt.units.__file__).read_text(encoding="utf-8"))

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.append(node.module)

    offending = [
        name
        for name in imported
        for family in forbidden
        if name == family or name.startswith(f"{family}.")
    ]
    assert not offending, offending
