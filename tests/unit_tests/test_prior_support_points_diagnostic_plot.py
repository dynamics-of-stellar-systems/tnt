"""Support overlays and traces collected from the actual Optimistix solve."""

from pathlib import Path

import numpy as np
import pytest

from tnt.all_models import AllModels
from tnt.model import Model
from tnt.parameter_generator import build_parameter_generator
from tnt.parameter_generator.diagnostics.support_points import (
    build_support_points_diagnostics_figure,
    plot_prior_support_points_diagnostics,
)
from tnt.potential import Potential
from tnt.priors import Prior
from tnt.support_points import optimize_support_points


def _generator(
    single_parameter=False,
    output_directory=None,
    constant=False,
    plot_diagnostics=False,
    generator_type="PriorSupportPoints",
):
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
    if constant:
        potential["halo"]["parameters"]["m"]["prior"] = {
            "distribution": "Delta",
            "args": [1e12],
        }
    return build_parameter_generator(
        {
            "generator_type": generator_type,
            "generator_settings": {
                "plot_diagnostics": plot_diagnostics,
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
        output_directory=output_directory,
    )


def _render(generator, history):
    _generated_batch = generator.generate_parameters(history, record_history=True)
    candidates, info = _generated_batch.parameters, _generated_batch.diagnostics
    return build_support_points_diagnostics_figure(candidates, info), candidates, info


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
    _generated_batch = generator.generate_parameters(AllModels())
    previous, _ = _generated_batch.parameters, _generated_batch.diagnostics
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
    fig, candidates, info = _render(generator, history)
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
    _generated_batch = _generator(single_parameter).generate_parameters(
        AllModels(), record_history=True
    )
    candidates, info = _generated_batch.parameters, _generated_batch.diagnostics
    plot_prior_support_points_diagnostics(candidates, info, path)
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
    figure, points, info = _render(generator, AllModels())
    assert len(points) == 1
    assert info["new_support_points"] == 2
    assert info["proposals"]["returned"] == 1
    assert info["optimization"].history.objective == (0.0, 0.0)
    assert figure.get_size_inches()[0] >= 10.0
    figure.savefig(tmp_path / "fixed.png")


def test_default_output_paths_preserve_each_iteration(tmp_path: Path):
    generator = _generator(output_directory=tmp_path, plot_diagnostics=True)
    _generated_batch = generator.generate_parameters(AllModels())
    first, _ = _generated_batch.parameters, _generated_batch.diagnostics
    history = AllModels()
    for point in first:
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
    first_path = (
        generator.output_directory / "support_points_diagnostics_iteration_0000.png"
    )
    original = first_path.read_bytes()
    generator.generate_parameters(history)
    assert first_path.read_bytes() == original
    assert (
        generator.output_directory / "support_points_diagnostics_iteration_0001.png"
    ).stat().st_size > 0


def test_plot_overlays_only_returned_points_after_duplicate_filtering():
    generator = _generator(single_parameter=True, constant=True)
    figure, points, info = _render(generator, AllModels())
    assert len(points) == info["proposals"]["returned"] == 1
    assert info["new_support_points"] == 4
    assert info["proposals"]["duplicates_removed"] == 3
    rug = next(
        line
        for ax in figure.axes
        for line in ax.lines
        if line.get_gid() == "support-points-rug"
    )
    assert len(rug.get_xdata()) == 1
    ordinary = generator.generate_parameters(AllModels()).parameters
    assert len(ordinary) == len(points)
    assert float(ordinary[0]["halo"]["m"].ustrip("Msun")) == float(
        points[0]["halo"]["m"].ustrip("Msun")
    )


@pytest.mark.parametrize("kind", ["PriorSampler", "PriorSupportPoints"])
def test_enabled_plots_use_one_actual_batch_and_preserve_candidates(
    tmp_path, monkeypatch, kind
):
    import tnt.parameter_generator.prior_support_points as support_module
    from tnt.parameter_generator.diagnostics.prior import plot_prior_diagnostics

    plain = _generator(single_parameter=True, generator_type=kind)
    expected = plain.generate_parameters(AllModels()).parameters
    enabled = _generator(
        single_parameter=True,
        generator_type=kind,
        output_directory=tmp_path,
        plot_diagnostics=True,
    )
    sampled = []
    optimized = []
    sample = enabled.prior.sample_with_diagnostics
    optimize = support_module.optimize_support_points

    def capture_sample(*args, **kwargs):
        result = sample(*args, **kwargs)
        sampled.append(result)
        return result

    def capture_optimize(*args, **kwargs):
        optimized.append(kwargs.get("record_history"))
        return optimize(*args, **kwargs)

    monkeypatch.setattr(enabled.prior, "sample_with_diagnostics", capture_sample)
    monkeypatch.setattr(support_module, "optimize_support_points", capture_optimize)
    _generated_batch = enabled.generate_parameters(AllModels())
    actual, info = _generated_batch.parameters, _generated_batch.diagnostics
    assert len(sampled) == 1
    assert optimized == ([True] if kind == "PriorSupportPoints" else [])
    assert info["sample_batch"].samples is sampled[0][0]
    assert info["sample_batch"].diagnostics is sampled[0][1]
    np.testing.assert_array_equal(
        [float(p["halo"]["m"].ustrip("Msun")) for p in actual],
        [float(p["halo"]["m"].ustrip("Msun")) for p in expected],
    )
    if kind == "PriorSupportPoints":
        figure = build_support_points_diagnostics_figure(actual, info)
        prefix = "support_points_diagnostics"
        plot_prior_support_points_diagnostics(actual, info, tmp_path / "explicit.png")
    else:
        from tnt.parameter_generator.diagnostics.prior import (
            build_prior_diagnostics_figure,
        )

        figure = build_prior_diagnostics_figure(info["sample_batch"], iteration=0)
        prefix = "prior_diagnostics"
        plot_prior_diagnostics(info["sample_batch"], tmp_path / "explicit.png")
    assert "model-search iteration 0" in figure._suptitle.get_text()
    assert (
        tmp_path / "parameter_generator" / f"{prefix}_iteration_0000.png"
    ).stat().st_size > 0
    assert len(sampled) == 1  # Rendering previously returned data does no work.
    assert len(optimized) == (1 if kind == "PriorSupportPoints" else 0)


@pytest.mark.parametrize("kind", ["PriorSampler", "PriorSupportPoints"])
def test_disabled_plots_create_no_output_and_missing_enabled_output_fails_early(
    tmp_path, monkeypatch, kind
):
    disabled = _generator(generator_type=kind, output_directory=tmp_path)
    assert disabled.generate_parameters(AllModels()).parameters
    assert not (tmp_path / "parameter_generator").exists()
    enabled = _generator(generator_type=kind, plot_diagnostics=True)

    def must_not_sample(*args, **kwargs):
        pytest.fail("Missing output destination should be detected before sampling")

    monkeypatch.setattr(enabled.prior, "sample_with_diagnostics", must_not_sample)
    with pytest.raises(ValueError, match="output_directory is not configured"):
        enabled.generate_parameters(AllModels())


def test_support_renderer_requires_previously_recorded_data():
    generator = _generator(single_parameter=True)
    _generated_batch = generator.generate_parameters(AllModels())
    points, info = _generated_batch.parameters, _generated_batch.diagnostics
    with pytest.raises(ValueError, match="requires record_history=True"):
        build_support_points_diagnostics_figure(points, info)
