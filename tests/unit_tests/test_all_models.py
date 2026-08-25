"""Unit tests for the unit-aware `AllModels` table."""

from __future__ import annotations

import pytest
from unxt import Quantity

from tnt.all_models import AllModels
from tnt.model import Model
from tnt.potential import Potential


def _unsolved_model(mass: Quantity, iteration: int) -> Model:
    return Model(
        potential=Potential(components={}),
        raw_parameters={"bh": {"m_tot": mass}},
        orblib_done=False,
        weights_done=False,
        weights=None,
        chi2=None,
        iteration=iteration,
    )


def test_append_converts_equivalent_declared_units_to_existing_column_unit() -> None:
    mass = Quantity(5.0, "Msun")
    models = AllModels().append(_unsolved_model(mass, 0))
    models = models.append(_unsolved_model(Quantity(float(mass.ustrip("kg")), "kg"), 1))

    stored_masses = models.table["bh.m_tot"].to_value("Msun")
    assert stored_masses == pytest.approx([5.0, 5.0])
