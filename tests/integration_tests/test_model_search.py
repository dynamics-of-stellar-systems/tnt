"""Run ModelIterator against the realistic resolved example configuration.

Unlike tests/unit_tests/test_model_iterator.py (which fakes every
collaborator), this builds the `ModelIterator` via the real
`from_configuration`, against the real `Configuration`,
`SinglePointParameterGenerator`, `AllModels`, and `RunConfigLog`.
Only `build_potential`, `build_weight_solver`, `build_orbit_sampler`, and
`build_orbit_dithering` are faked, since potential construction, orbit
integration, and weight solving are still unimplemented (see
tnt.model_iterator's module docstring).

`tnt.potential` imports `galax.potential` at module level, and this venv's
installed `galax`/`equinox` versions are mutually incompatible
(`ImportError: cannot import name '_has_dataclass_init' from
'equinox._module'`) -- an unrelated, pre-existing environment issue. Stub
out `galax`/`galax.potential` before importing anything from `tnt` that
would pull in that chain, so this test can run regardless. Remove this
stub once the real dependency conflict is fixed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, NamedTuple

if "galax" not in sys.modules:
    _fake_galax = types.ModuleType("galax")
    _fake_galax_potential = types.ModuleType("galax.potential")

    class _FakeAbstractPotentialBase:
        pass

    _fake_galax_potential.AbstractPotentialBase = _FakeAbstractPotentialBase
    _fake_galax.potential = _fake_galax_potential
    sys.modules["galax"] = _fake_galax
    sys.modules["galax.potential"] = _fake_galax_potential

import numpy as np
import pytest

import tnt.model_iterator as model_iterator_module
from tnt import Configuration
from tnt.model_iterator import ModelIterator
from tnt.run_config_log import RunConfigLog

# ---------------------------------------------------------------------------
# Fakes standing in for Potential / OrbitLibrary / AbstractWeightSolver --
# see tests/unit_tests/test_model_iterator.py for the same pattern.
# ---------------------------------------------------------------------------


class FakePotential:
    def __init__(self, mass: float = 1.0) -> None:
        self.mass = mass
        self.components: dict[str, Any] = {}

    def generate_orbit_library(
        self,
        settings: Any,
        sampler: Any,
        dithering: Any,
    ) -> Any:
        return FakeOrbitLibrary(self.mass)

    def rescale(self, mass_scale: float) -> FakePotential:
        return FakePotential(mass=self.mass * mass_scale)


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

    def fake_build_potential(settings: Any, mges: Any) -> FakePotential:
        captured_settings.append(settings)
        return FakePotential(mass=1.0)

    monkeypatch.setattr(model_iterator_module, "build_potential", fake_build_potential)
    monkeypatch.setattr(
        model_iterator_module,
        "build_weight_solver",
        lambda settings: FakeWeightSolver(),
    )
    # generate_orbit_library is faked (via build_potential above), so the
    # real orbit_sampler/orbit_dithering objects are never touched -- but
    # build_orbit_sampler/build_orbit_dithering themselves still need
    # faking, since they're still unimplemented too.
    monkeypatch.setattr(
        model_iterator_module, "build_orbit_sampler", lambda settings: "fake-sampler"
    )
    monkeypatch.setattr(
        model_iterator_module,
        "build_orbit_dithering",
        lambda settings: "fake-dithering",
    )

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
    log_path = RunConfigLog.path_for(config.run_manifest_path)
    config_log.write(log_path)
    restored_log = RunConfigLog.read(log_path)
    assert len(restored_log) == 1
    assert restored_log.table["run_id"][0] == config.run_id

    masses = sorted(10.0 / value for value in models.table["kinchi2"])
    expected_masses = sorted([1.0, *np.geomspace(0.1, 10.0, 10)])
    assert masses == pytest.approx(expected_masses)

    best = models.best("kinchi2")
    assert best["kinchi2"] == pytest.approx(min(models.table["kinchi2"]))

    # _settings_with_parameters overlaid SinglePoint's proposed values onto
    # the real resolved potential settings without disturbing anything else
    # (units are already stripped by this point -- configuration resolution
    # converts every declared unit into TNT's internal units and drops it).
    stars_ml = captured_settings[0]["stars"]["parameters"]["ml"]
    assert stars_ml["value"] == pytest.approx(5.0)
    assert stars_ml["fixed"] is False
    assert stars_ml["generator_settings"]["upper_bound"] == pytest.approx(9.0)


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
