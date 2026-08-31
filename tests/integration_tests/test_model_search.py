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
tnt.model_iterator's module docstring). `Potential.to_galax` (including the
MGE composite types') is real and implemented; the faked
`generate_orbit_library` never calls it during a `ModelIterator.run()`, so
most tests here never reach it either --
`test_potential_to_galax_succeeds_against_the_resolved_example_configuration`
is the exception, calling it directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from unxt import Quantity

import tnt.model_iterator as model_iterator_module
from tnt import Configuration
from tnt.all_models import AllModels
from tnt.configuration.compatibility import ConfigurationCompatibilityError
from tnt.model_iterator import ModelIterator
from tnt.model_search_state import ModelSearchState
from tnt.potential import Potential, _nfw_concentration_m200, build_potential
from tnt.run_config_log import (
    RUN_IDS_WITHOUT_ITERATIONS_METADATA_KEY,
    TOTAL_RUNS_METADATA_KEY,
    RunConfigLog,
)
from tnt.units import resolve_cosmological_parameters

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


class EmptyParameterGenerator:
    """Propose no model so the run completes without an iteration."""

    def generate_parameters(self, all_models: Any) -> list[Any]:
        del all_models
        return []


def _fake_orbit_integration_and_weight_solving(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake only what's still unimplemented: orbit integration and weight solving.

    `build_potential`/`Potential` stay real -- see this module's docstring
    for why `to_galax()` is never reached even so.
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

    captured_parameter_values: list[Any] = []
    real_build_potential = model_iterator_module.build_potential

    def spying_build_potential(
        resolved: Any,
        parameter_values: Any,
        cosmological_parameters: Any,
    ) -> Potential:
        captured_parameter_values.append(parameter_values)
        return real_build_potential(
            resolved, parameter_values, cosmological_parameters
        )

    monkeypatch.setattr(
        model_iterator_module, "build_potential", spying_build_potential
    )
    _fake_orbit_integration_and_weight_solving(monkeypatch)

    output = Path(resolved["io_settings"]["output_directory"])
    repository = output / "config_repository"
    assert not repository.exists()
    iterator = ModelIterator.from_configuration(config)
    assert not repository.exists()

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
    assert iterator.run_id == 0
    assert iterator.run_manifest is not None
    assert config_log.table["run_id"][0] == iterator.run_id
    manifest_path = iterator.run_manifest.absolute_run_manifest_path
    resolved_path = iterator.run_manifest.absolute_resolved_config_path
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    archived = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert archived == config.portable_data
    assert manifest["manifest_version"] == 3
    assert manifest["run_id"] == 0
    assert manifest["configuration"]["logfile"] is None
    assert manifest["configuration"]["resolved"] == "runs/0000/resolved_config.yaml"
    # Full manifest shape, not just the fields the rest of this test happens to
    # use -- restores the coverage test_configuration.py had before archiving
    # moved out of Configuration.read() (see the PR 37 run-boundary audit).
    assert set(manifest["tnt"]) == {"version", "git_commit", "git_working_tree_dirty"}
    assert "unxt" in manifest["dependencies"]
    assert manifest["execution"]["workspace_root"] == str(tmp_path)
    assert set(manifest["configuration"]) == {
        "input_directory",
        "logfile",
        "output_directory",
        "resolved",
    }
    assert manifest["configuration"]["input_directory"] == (
        resolved["io_settings"]["input_directory"]
    )
    assert manifest["configuration"]["output_directory"] == str(output)
    assert manifest["randomness"] == {
        "configured_orbit_library_seed": 4242,
        "effective_orbit_library_seed": 4242,
        "status": "fixed",
    }
    models_path = output / resolved["io_settings"]["all_models_file"]
    log_path = RunConfigLog.path_for(manifest_path)
    ModelSearchState(models, config_log).write(models_path, log_path)
    restored_state = ModelSearchState.read(models_path, log_path)
    assert len(restored_state.all_models) == 11
    assert len(restored_state.run_config_log) == 1
    assert restored_state.run_config_log.table["run_id"][0] == iterator.run_id

    masses = sorted(10.0 / value for value in models.table["kinchi2"])
    expected_masses = sorted([1.0, *np.geomspace(0.1, 10.0, 10)])
    assert masses == pytest.approx(expected_masses)

    best = models.best("kinchi2")
    assert best["kinchi2"] == pytest.approx(min(models.table["kinchi2"]))

    # SinglePoint proposes ml as a Quantity in its declared unit.
    # ModelIterator passes proposed values separately, so resolved config keeps
    # fixed/generator_settings unmodified (see tnt.potential's module docstring).
    stars_ml_settings = resolved["potential"]["stars"]["parameters"]["ml"]
    assert stars_ml_settings["fixed"] is False
    assert stars_ml_settings["generator_settings"]["upper_bound"] == pytest.approx(9.0)
    stars_ml_value = captured_parameter_values[0]["stars"]["ml"]
    assert stars_ml_value.ustrip("Msun / Lsun") == pytest.approx(5.0)

    # The same iterator may start another run. This configuration has already
    # exceeded its soft model target, so the second call adds no iteration but
    # still passes resume compatibility and receives its own archive.
    resumed_models, resumed_log = iterator.run(models, config_log)
    assert resumed_models is models
    assert resumed_log is config_log
    assert iterator.run_id == 1
    assert sorted(path.name for path in (repository / "runs").iterdir()) == [
        "0000",
        "0001",
    ]


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

    iterator = ModelIterator.from_configuration(config)
    models, config_log = iterator.run()

    assert len(models) == 11
    assert iterator.run_manifest is not None
    output = Path(resolved["io_settings"]["output_directory"])
    models_path = output / resolved["io_settings"]["all_models_file"]
    log_path = RunConfigLog.path_for(iterator.run_manifest.absolute_run_manifest_path)
    ModelSearchState(models, config_log).write(models_path, log_path)

    table = models.table
    assert {"bh.m_tot", "bh.r_s", "dh.c", "dh.M_200", "stars.ml"}.issubset(
        set(table.colnames)
    )
    assert "dh.m" not in table.colnames
    assert "dh.r_s" not in table.colnames

    cosmological_parameters = resolve_cosmological_parameters(
        resolved["cosmological_parameters"]
    )
    r_s_values = []
    m_over_mass_scale_values = []
    c_values = []
    for row in table:
        mass_scale = 10.0 / row["kinchi2"]
        assert row["bh.m_tot"].to_value("Msun") == pytest.approx(1.0e5 * mass_scale)
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
            cosmological_parameters,
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


def test_potential_to_galax_succeeds_against_the_resolved_example_configuration(
    example_configuration_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Potential.to_galax()` real end to end, against the realistic config.

    Every other test in this module fakes `generate_orbit_library` (per this
    module's docstring), so `to_galax()` -- including the MGE composite
    types' deprojection-to-`galax` assembly -- is otherwise never actually
    called here, even though potential construction up to that point is
    real. This calls it directly on a `Potential` built from the real
    resolved configuration's own proposed point, confirming it produces a
    working `galax` potential rather than just that its unit-tested pieces
    each work in isolation.
    """
    config = Configuration().read(example_configuration_path, workspace_root=tmp_path)
    unit_system = config.unit_systems.internal
    _fake_orbit_integration_and_weight_solving(monkeypatch)

    iterator = ModelIterator.from_configuration(config)
    (parameters,) = iterator.parameter_generator.generate_parameters(AllModels())
    potential = build_potential(
        iterator.resolved_potential,
        parameters,
        iterator.cosmological_parameters,
    )

    galax_potential = potential.to_galax(unit_system)

    xyz = Quantity(jnp.array([5.0, -3.0, 2.0]), "kpc")
    t = Quantity(0.0, "Myr")
    speed2 = unit_system["length"] ** 2 / unit_system["time"] ** 2
    value = galax_potential.potential(xyz, t).ustrip(speed2)
    assert np.isfinite(value)
    assert value < 0.0  # bound system: potential is negative everywhere


def test_model_iterator_rejects_stage_by_stage_before_runtime_construction(
    example_configuration_path: Path,
    tmp_path: Path,
) -> None:
    config = Configuration().read(example_configuration_path, workspace_root=tmp_path)
    config.data["execution_settings"]["model_processing_order"] = "stage_by_stage"

    with pytest.raises(
        NotImplementedError,
        match=r"model_processing_order='stage_by_stage' is not implemented",
    ):
        ModelIterator.from_configuration(config)

    output = Path(config.data["io_settings"]["output_directory"])
    assert not (output / "config_repository").exists()


def test_from_configuration_rejects_an_unread_configuration() -> None:
    with pytest.raises(
        RuntimeError,
        match="Configuration must be read before construction",
    ):
        ModelIterator.from_configuration(Configuration())


def test_invalid_runtime_owned_configuration_is_not_archived(
    example_configuration_path: Path,
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(example_configuration_path.read_text(encoding="utf-8"))
    raw["spatial_binnings"]["kinset1_binning"]["min_x"] = 5.0
    example_configuration_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    config = Configuration().read(example_configuration_path, workspace_root=tmp_path)
    output = Path(config.data["io_settings"]["output_directory"])

    with pytest.raises(TypeError, match=r"spatial_binnings.*min_x"):
        ModelIterator.from_configuration(config)

    assert not (output / "config_repository").exists()


def test_each_zero_iteration_call_is_archived_and_reported_as_a_run(
    example_configuration_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Configuration().read(example_configuration_path, workspace_root=tmp_path)
    _fake_orbit_integration_and_weight_solving(monkeypatch)
    iterator = ModelIterator.from_configuration(config)
    iterator.parameter_generator = EmptyParameterGenerator()

    models, config_log = iterator.run()

    assert len(models) == 0
    assert len(config_log) == 0
    assert iterator.run_manifest is not None
    assert iterator.run_manifest.run_id == 0

    models, config_log = iterator.run(models, config_log)

    assert len(models) == 0
    assert len(config_log) == 0
    assert iterator.run_manifest is not None
    assert iterator.run_manifest.run_id == 1
    output = Path(config.data["io_settings"]["output_directory"])
    models_path = output / config.data["io_settings"]["all_models_file"]
    log_path = RunConfigLog.path_for_repository(iterator.configuration_repository)
    ModelSearchState(models, config_log).write(models_path, log_path)
    assert config_log.table.meta[TOTAL_RUNS_METADATA_KEY] == 2
    assert config_log.table.meta[RUN_IDS_WITHOUT_ITERATIONS_METADATA_KEY] == [0, 1]


def test_incompatible_resume_is_rejected_before_allocating_a_run(
    example_configuration_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_orbit_integration_and_weight_solving(monkeypatch)
    first_config = Configuration().read(
        example_configuration_path,
        workspace_root=tmp_path,
    )
    first_iterator = ModelIterator.from_configuration(first_config)
    models, config_log = first_iterator.run()

    raw = yaml.safe_load(example_configuration_path.read_text(encoding="utf-8"))
    raw["system_attributes"]["distance"]["value"] = 40.0
    example_configuration_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    changed_config = Configuration().read(
        example_configuration_path,
        workspace_root=tmp_path,
    )
    resumed_iterator = ModelIterator.from_configuration(changed_config)

    with pytest.raises(
        ConfigurationCompatibilityError,
        match=r"system_attributes\.distance",
    ):
        resumed_iterator.run(models, config_log)

    runs_directory = first_iterator.configuration_repository / "runs"
    assert [path.name for path in runs_directory.iterdir()] == ["0000"]
    assert resumed_iterator.run_manifest is None
