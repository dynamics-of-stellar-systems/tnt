"""Run ModelIterator against the realistic resolved example configuration.

Unlike tests/unit_tests/test_model_iterator.py (which fakes every
collaborator), this builds the `ModelIterator` via the real
`from_configuration`, against the real `Configuration`,
`SinglePointParameterGenerator`, `AllModels`, `RunConfigLog`, and
`build_potential` -- potential construction (including MGE composite
components and non-native parameterizations) is real end to end. Only
`Potential.generate_orbit_library`, `build_weight_solver`,
`build_orbit_sampler`, and `build_orbit_dithering` are faked, since orbit
integration and weight solving are still unimplemented (see
tnt.model_iterator's module docstring). `generate_orbit_library` never
calls `to_galax()`, so the MGE composite types' still-`NotImplementedError`
`to_galax()` is never reached here either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
from unxt import Quantity

import tnt.model_iterator as model_iterator_module
from tnt import Configuration
from tnt.model_iterator import ModelIterator
from tnt.model_search_state import ModelSearchState
from tnt.potential import Potential, _nfw_concentration_m200
from tnt.run_config_log import RunConfigLog

# ---------------------------------------------------------------------------
# Fakes standing in for OrbitLibrary / AbstractWeightSolver -- orbit
# integration and weight solving, unlike potential construction, are still
# unimplemented. See tests/unit_tests/test_model_iterator.py for the same
# pattern.
# ---------------------------------------------------------------------------


class FakeOrbitLibrary:
    def __init__(self, mass: float) -> None:
        self.mass = mass

    def rescaled(self, mass_scale: float) -> FakeOrbitLibrary:
        return FakeOrbitLibrary(self.mass * mass_scale)


class _FakeWeightResult(NamedTuple):
    weights: str
    chi2: dict[str, float]


class FakeWeightSolver:
    """`chi2["kinchi2"] == 10.0 / orbit_library.mass`, so the mass is recoverable."""

    def solve(
        self, orbit_library: FakeOrbitLibrary, kinematic_data: Any
    ) -> _FakeWeightResult:
        value = 10.0 / orbit_library.mass
        return _FakeWeightResult(
            weights="weights", chi2={"chi2": value, "kinchi2": value}
        )


def _fake_orbit_integration_and_weight_solving(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake only what's still unimplemented: orbit integration and weight solving.

    `build_potential`/`Potential` stay real -- see this module's docstring
    for why the MGE composite types' unimplemented `to_galax()` is never
    reached even so.
    """

    def fake_generate_orbit_library(
        self: Potential, settings: Any, sampler: Any, dithering: Any
    ) -> FakeOrbitLibrary:
        del self, settings, sampler, dithering
        return FakeOrbitLibrary(mass=1.0)

    monkeypatch.setattr(
        Potential, "generate_orbit_library", fake_generate_orbit_library
    )
    monkeypatch.setattr(
        model_iterator_module,
        "build_weight_solver",
        lambda settings: FakeWeightSolver(),
    )
    monkeypatch.setattr(
        model_iterator_module, "build_orbit_sampler", lambda settings: "fake-sampler"
    )
    monkeypatch.setattr(
        model_iterator_module,
        "build_orbit_dithering",
        lambda settings: "fake-dithering",
    )


def test_model_iterator_runs_against_the_resolved_example_configuration(
    example_configuration_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Configuration().read(example_configuration_path, workspace_root=tmp_path)
    resolved = config.as_dict()
    parameter_space_settings = resolved["parameter_space_settings"]
    assert parameter_space_settings["generator_type"] == "SinglePoint"

    captured_settings: list[Any] = []
    real_build_potential = model_iterator_module.build_potential

    def spying_build_potential(
        settings: Any, mges: Any, unit_system: Any, cosmological_parameters: Any
    ) -> Potential:
        captured_settings.append(settings)
        return real_build_potential(
            settings, mges, unit_system, cosmological_parameters
        )

    monkeypatch.setattr(
        model_iterator_module, "build_potential", spying_build_potential
    )
    _fake_orbit_integration_and_weight_solving(monkeypatch)

    assert config.run_manifest_path is not None
    iterator = ModelIterator.from_configuration(
        resolved, config.unit_systems.internal, config.run_manifest_path
    )

    models, config_log = iterator.run()

    # SinglePoint always proposes the same point, so the first round's
    # potential_rescalings sweep (range_count=10, none landing on 1.0) puts
    # 11 models straight past stopping_criteria.target_model_count=3. This is
    # a soft target checked only before each round, so the round that crosses
    # it still completes and the loop then stops after that one round.
    assert parameter_space_settings["stopping_criteria"]["target_model_count"] == 3
    assert len(models) == 11
    assert models.n_iterations() == 1
    assert len(config_log) == 1
    assert config.run_id is not None
    assert config_log.table["run_id"][0] == config.run_id
    output = Path(resolved["io_settings"]["output_directory"])
    models_path = output / resolved["io_settings"]["all_models_file"]
    log_path = RunConfigLog.path_for(config.run_manifest_path)
    ModelSearchState(models, config_log).write(models_path, log_path)
    restored_state = ModelSearchState.read(models_path, log_path)
    assert len(restored_state.all_models) == 11
    assert len(restored_state.run_config_log) == 1
    assert restored_state.run_config_log.table["run_id"][0] == config.run_id

    masses = sorted(10.0 / value for value in models.table["kinchi2"])
    expected_masses = sorted([1.0, *np.geomspace(0.1, 10.0, 10)])
    assert masses == pytest.approx(expected_masses)

    best = models.best("kinchi2")
    assert best["kinchi2"] == pytest.approx(min(models.table["kinchi2"]))

    # _settings_with_parameters overlaid SinglePoint's proposed values onto
    # the shared runtime potential settings without disturbing anything else.
    # ModelIterator converts the preserved declarations into internal parameter
    # coordinates once for both the generator and potential construction.
    stars_ml = captured_settings[0]["stars"]["parameters"]["ml"]
    assert stars_ml["value"] == pytest.approx(5.0)
    assert stars_ml["fixed"] is False
    assert stars_ml["generator_settings"]["upper_bound"] == pytest.approx(9.0)


def test_model_iterator_reports_real_potential_in_its_own_parameterization(
    example_configuration_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`raw_potential_parameters` reports every component correctly, real end to end.

    Unlike the test above (focused on the search loop itself, with a spy on
    `build_potential`), this test's focus is `dh`'s non-native
    `concentration_m200` parameterization -- confirming `AllModels`' table
    reports it (`dh.c`/`dh.M_200`) rather than galax's native `dh.m`/
    `dh.r_s`, correctly recomputed after every `potential_rescalings` variant
    despite `_nfw_concentration_m200_inverse` having no closed form.
    """
    config = Configuration().read(example_configuration_path, workspace_root=tmp_path)
    resolved = config.as_dict()
    _fake_orbit_integration_and_weight_solving(monkeypatch)

    assert config.run_manifest_path is not None
    iterator = ModelIterator.from_configuration(
        resolved, config.unit_systems.internal, config.run_manifest_path
    )
    models, config_log = iterator.run()

    assert len(models) == 11
    output = Path(resolved["io_settings"]["output_directory"])
    models_path = output / resolved["io_settings"]["all_models_file"]
    log_path = RunConfigLog.path_for(config.run_manifest_path)
    ModelSearchState(models, config_log).write(models_path, log_path)

    table = models.table
    assert {"bh.m_tot", "bh.r_s", "dh.c", "dh.M_200", "stars.ml"}.issubset(
        set(table.colnames)
    )
    assert "dh.m" not in table.colnames
    assert "dh.r_s" not in table.colnames

    unit_system = config.unit_systems.internal
    h0 = resolved["cosmological_parameters"]["H0"]
    r_s_values = []
    m_over_mass_scale_values = []
    c_values = []
    for row in table:
        mass_scale = 10.0 / row["kinchi2"]
        assert row["bh.m_tot"].to_value("Msun") == pytest.approx(5.0 * mass_scale)
        assert row["bh.r_s"].to_value("kpc") == pytest.approx(1.0e-3)
        assert row["stars.ml"].to_value("Msun / Lsun") == pytest.approx(
            5.0 * mass_scale
        )

        # Feed the table's own reported (c, M_200) back through the forward
        # conversion -- there's no closed form for the inverse, so this
        # self-consistency check (rather than an independently derivable
        # expected value) is what actually confirms the round trip worked.
        native = _nfw_concentration_m200(
            {
                "c": Quantity(float(row["dh.c"].value), ""),
                "M_200": Quantity(row["dh.M_200"].to_value("Msun"), "Msun"),
            },
            unit_system,
            {"H0": h0},
        )
        r_s_values.append(float(native["r_s"].ustrip("kpc")))
        m_over_mass_scale_values.append(float(native["m"].ustrip("Msun")) / mass_scale)
        c_values.append(float(row["dh.c"].value))

    # rescale() holds r_s fixed and scales m linearly with mass_scale
    # (native-space invariants); the raw (c, M_200) round trip must
    # reproduce both exactly.
    assert max(r_s_values) == pytest.approx(min(r_s_values), rel=1e-5)
    assert max(m_over_mass_scale_values) == pytest.approx(
        min(m_over_mass_scale_values), rel=1e-5
    )
    # Concentration genuinely changes across mass scales -- holding c fixed
    # would be the wrong (but easy-to-accidentally-implement) shortcut.
    assert len({round(c, 6) for c in c_values}) > 1


def test_model_iterator_rejects_stage_by_stage_before_runtime_construction(
    example_configuration_path: Path,
    tmp_path: Path,
) -> None:
    config = Configuration().read(example_configuration_path, workspace_root=tmp_path)
    resolved = config.as_dict()
    resolved["execution_settings"]["model_processing_order"] = "stage_by_stage"
    assert config.run_manifest_path is not None

    with pytest.raises(
        NotImplementedError,
        match=r"model_processing_order='stage_by_stage' is not implemented",
    ):
        ModelIterator.from_configuration(
            resolved, config.unit_systems.internal, config.run_manifest_path
        )
