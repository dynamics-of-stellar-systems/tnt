"""Support overlays and traces collected from the actual Optimistix solve."""

from pathlib import Path

import jax
import numpy as np
import pytest

from tnt.all_models import AllModels
from tnt.model import Model
from tnt.parameter_generator import build_parameter_generator
from tnt.potential import Potential
from tnt.prior_support_points_diagnostic_plot import (
    _build_support_points_diagnostics_figure,
    plot_prior_support_points_diagnostics,
)
from tnt.priors import Prior
from tnt.support_points import optimize_support_points


def _generator(single_parameter=False):
    potential = {
        "halo": {
            "type": "NFWPotential",
            "parameters": {
                "m": {
                    "fixed": False,
                    "unit": "Msun",
                    "prior": {"distribution": "LogUniform", "args": [1e10, 1e13]},
                },
                "r_s": {
                    "fixed": False,
                    "unit": "kpc",
                    "prior": {"distribution": "Uniform", "args": [5.0, 25.0]},
                },
            },
        }
    }
    if single_parameter:
        potential["halo"]["parameters"]["r_s"].update(fixed=True, value=15.0)
    return build_parameter_generator(
        {
            "generator_type": "PriorSupportPoints",
            "generator_settings": {
                "seed": 1,
                "num_samples": 200,
                "num_warmup": 120,
                "target_accept_prob": 0.8,
                "max_iterations": 15,
            },
            "stopping_criteria": {"max_new_mods_per_iter": 4},
        },
        potential,
        Prior(potential, {}, {}),
    )


@pytest.mark.parametrize("tolerance", [1e-12, 1e3])
def test_history_records_exact_solver_steps_without_changing_solution(tolerance):
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(70, 2))
    initial, fixed = rng.normal(size=(4, 2)), rng.normal(size=(2, 2))
    plain, plain_info = optimize_support_points(
        reference, initial, fixed_points=fixed, max_iterations=12, tolerance=tolerance
    )
    traced, info = optimize_support_points(
        reference,
        initial,
        fixed_points=fixed,
        max_iterations=12,
        tolerance=tolerance,
        record_history=True,
    )
    np.testing.assert_allclose(traced, plain, atol=1e-12)
    assert plain_info.history is None
    assert len(info.history.objective) == info.iterations + 1
    assert len(info.history.maximum_displacement) == info.iterations
    assert len(info.history.rms_displacement) == info.iterations
    assert info.history.objective[0] == info.initial_objective
    assert info.history.objective[-1] == pytest.approx(info.objective, abs=1e-12)
    assert info.history.maximum_displacement[-1] == pytest.approx(info.full_step)
    assert np.all(
        np.array(info.history.rms_displacement)
        <= np.array(info.history.maximum_displacement) + 1e-15
    )


def test_corner_overlays_existing_and_new_points_and_plots_actual_history():
    generator = _generator()
    previous, _ = generator.generate_with_diagnostics(AllModels())
    history = AllModels()
    for point in previous:
        history = history.append(
            Model(
                potential=Potential(components={}),
                valid_potential=True,
                raw_parameters=point,
                orblib_done=False,
                weights_done=False,
                weights=None,
                chi2=None,
                iteration=0,
            )
        )
    fig, candidates, info = _build_support_points_diagnostics_figure(
        generator, jax.random.PRNGKey(2), history
    )
    for gid, points in (
        ("support-points-scatter", candidates),
        ("fixed-support-points-scatter", previous),
    ):
        collections = [
            artist
            for ax in fig.axes
            for artist in ax.collections
            if artist.get_gid() == gid
        ]
        assert len(collections) == 1
        expected = [
            [
                float(point["halo"]["m"].ustrip("Msun")),
                float(point["halo"]["r_s"].ustrip("kpc")),
            ]
            for point in points
        ]
        np.testing.assert_allclose(collections[0].get_offsets(), expected)
    objective = next(
        ax for ax in fig.axes if ax.get_gid() == "support-objective-history"
    )
    np.testing.assert_array_equal(
        objective.lines[0].get_ydata(), info["optimization"].history.objective
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        bbox = ax.get_tightbbox(renderer)
        if bbox is not None:
            assert bbox.x0 >= 0 and bbox.y0 >= 0
            assert bbox.x1 <= fig.bbox.width and bbox.y1 <= fig.bbox.height


@pytest.mark.parametrize("single_parameter", [False, True])
def test_plot_saves_and_returns_the_generated_design(tmp_path: Path, single_parameter):
    path = tmp_path / "support.png"
    candidates, info = plot_prior_support_points_diagnostics(
        _generator(single_parameter), path, jax.random.PRNGKey(2)
    )
    assert path.stat().st_size > 0
    assert len(candidates) == 4
    assert info["optimization"].history is not None


def test_all_fixed_plot_has_valid_history_and_readable_panels(tmp_path: Path):
    generator = build_parameter_generator(
        {
            "generator_type": "PriorSupportPoints",
            "generator_settings": {
                "seed": 1,
                "num_samples": 200,
                "num_warmup": 120,
                "target_accept_prob": 0.8,
            },
            "stopping_criteria": {"max_new_mods_per_iter": 2},
        },
        {},
        Prior({}, {}, {}),
    )
    figure, points, info = _build_support_points_diagnostics_figure(
        generator, jax.random.PRNGKey(2), AllModels()
    )
    assert len(points) == 2
    assert info["optimization"].history.objective == (0.0, 0.0)
    assert figure.get_size_inches()[0] >= 10.0
    figure.savefig(tmp_path / "fixed.png")
