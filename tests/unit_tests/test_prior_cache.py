"""Cache lifecycle and resume tests with inexpensive deterministic sampling."""

import json
from copy import deepcopy
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from tnt.parameter_generator.prior_cache import PriorSampleCache
from tnt.priors import Prior, PriorDiagnostics, load_prior_plugins

SETTINGS = {
    "bh": {
        "type": "KeplerPotential",
        "parameters": {
            "m_tot": {
                "unit": "Msun",
                "fixed": False,
                "value": 1.0,
                "prior": {"distribution": "Uniform", "args": [1.0, 10.0]},
            }
        },
    }
}


def generator(path, *, seed=7, enabled=True, settings=None):
    settings = deepcopy(SETTINGS if settings is None else settings)
    prior = Prior(settings, {}, {})
    prior.calls = []

    def sample(key, *, num_samples, **kwargs):
        prior.calls.append(np.asarray(key))
        samples = {"bh.m_tot": np.asarray(jax.random.uniform(key, (num_samples,)))}
        return samples, PriorDiagnostics(
            np.zeros(num_samples, dtype=bool),
            {"bh.m_tot": float(num_samples)},
            {"bh.m_tot": 1.0},
            {"bh.m_tot": "Uniform"},
        )

    prior.sample_with_diagnostics = sample
    return SimpleNamespace(
        prior=prior,
        sample_cache=PriorSampleCache(),
        generator_settings={"cache_prior_samples": enabled},
        potential_settings=settings,
        output_directory=path,
        seed=seed,
        num_warmup=10,
        target_accept_prob=0.8,
        max_new_mods_per_iter=3,
    )


def _pool(g, *, size, consume, iteration=0):
    return g.sample_cache.get(
        g.prior,
        potential_settings=g.potential_settings,
        seed=g.seed,
        size=size,
        num_warmup=g.num_warmup,
        target_accept_prob=g.target_accept_prob,
        output_directory=g.output_directory,
        enabled=g.generator_settings["cache_prior_samples"],
        reservation_size=g.max_new_mods_per_iter if consume else None,
        iteration=iteration,
    )


def reserve(g):
    batch, reservation = _pool(g, size=8, consume=True)
    return batch.samples["bh.m_tot"][reservation["start"] : reservation["stop"]]


def test_resume_and_exhaustion_match_uninterrupted_sequence(tmp_path):
    uninterrupted = generator(tmp_path / "whole")
    expected = [reserve(uninterrupted) for _ in range(7)]
    assert [len(x) for x in expected] == [3, 3, 2, 3, 3, 2, 3]
    assert len(uninterrupted.prior.calls) == 3
    assert not np.array_equal(expected[0], expected[3])
    for index in range(7):
        # Reconstruct the entire Prior/generator on every resume. No AllModels
        # progress is required: a crash after reservation must skip those draws.
        resumed = generator(tmp_path / "resumed")
        np.testing.assert_array_equal(reserve(resumed), expected[index])
        assert len(resumed.prior.calls) == int(index in (0, 3, 6))


def test_reference_reused_without_consumption_and_roundtrip_diagnostics(tmp_path):
    first = generator(tmp_path)
    batch, reservation = _pool(first, size=8, consume=False)
    assert reservation is None
    reconstructed = generator(tmp_path)
    cached, _ = _pool(reconstructed, size=8, consume=False)
    assert not reconstructed.prior.calls
    np.testing.assert_array_equal(batch.samples["bh.m_tot"], cached.samples["bh.m_tot"])
    np.testing.assert_array_equal(
        batch.diagnostics.diverging, cached.diagnostics.diverging
    )
    assert batch.diagnostics.ess == cached.diagnostics.ess
    assert (
        batch.diagnostics.sample_distributions
        == cached.diagnostics.sample_distributions
    )


@pytest.mark.parametrize("change", ["seed", "prior", "units", "warmup", "accept"])
def test_cache_invalidation_starts_new_sequence(tmp_path, change):
    reserve(generator(tmp_path))
    modified = generator(tmp_path)
    if change == "seed":
        modified.seed += 1
    elif change == "prior":
        modified.potential_settings["bh"]["parameters"]["m_tot"]["prior"]["args"][1] = (
            20.0
        )
    elif change == "units":
        modified.potential_settings["bh"]["parameters"]["m_tot"]["unit"] = "kg"
    elif change == "warmup":
        modified.num_warmup += 1
    else:
        modified.target_accept_prob = 0.9
    _, reservation = _pool(modified, size=8, consume=True)
    assert reservation == {"round": 0, "start": 0, "stop": 3, "reserved": 3}
    assert len(modified.prior.calls) == 1


def test_plugin_source_changes_identity(tmp_path):
    path = tmp_path / "plugin.py"
    path.write_text("def plugin(context):\n    pass\n")
    config = {"test": {"plugin": "plugin.py:plugin"}}
    first = Prior(SETTINGS, load_prior_plugins(config, tmp_path), {})
    path.write_text(
        "def plugin(context):\n    # a changed prior implementation\n    pass\n"
    )
    second = Prior(SETTINGS, load_prior_plugins(config, tmp_path), {})
    assert first.cache_identity() != second.cache_identity()
    assert first.cache_identity() == first.cache_identity()


def test_context_changes_identity():
    first = Prior(SETTINGS, {}, {}, cosmological_parameters={"test": 1.0})
    second = Prior(SETTINGS, {}, {}, cosmological_parameters={"test": 2.0})
    assert first.cache_identity() != second.cache_identity()


def test_memory_pool_and_opt_out(tmp_path):
    memory = generator(None)
    assert not np.array_equal(reserve(memory), reserve(memory))
    assert len(memory.prior.calls) == 1
    uncached = generator(tmp_path, enabled=False)
    a, _ = _pool(uncached, size=8, consume=True, iteration=0)
    b, _ = _pool(uncached, size=8, consume=True, iteration=1)
    assert len(uncached.prior.calls) == 2
    assert not np.array_equal(a.samples["bh.m_tot"], b.samples["bh.m_tot"])
    assert not list(tmp_path.iterdir())


def test_corrupt_cache_does_not_silently_rewind(tmp_path):
    reserve(generator(tmp_path))
    (tmp_path / "prior_samples.npz").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="Cannot resume prior cache"):
        reserve(generator(tmp_path))


def test_interrupted_publish_preserves_previous_reservation(tmp_path, monkeypatch):
    reserve(generator(tmp_path))
    import tnt.parameter_generator.prior_cache as cache

    with monkeypatch.context() as patch:
        patch.setattr(
            cache.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk"))
        )
        with pytest.raises(OSError, match="disk"):
            reserve(generator(tmp_path))
    resumed = generator(tmp_path)
    _, reservation = _pool(resumed, size=8, consume=True)
    assert reservation["start"] == 3
    assert not list(tmp_path.glob(".prior_samples-*"))


def test_actual_generator_reservation_logging_and_duplicate_consumption(
    tmp_path, caplog
):
    from tnt.all_models import AllModels
    from tnt.parameter_generator import build_parameter_generator

    settings = deepcopy(SETTINGS)
    settings["bh"]["parameters"]["m_tot"]["prior"] = {
        "distribution": "Delta",
        "args": [2.0],
    }
    prior = Prior(settings, {}, {})
    generator = build_parameter_generator(
        {
            "generator_type": "PriorSampler",
            "generator_settings": {
                "seed": 7,
                "num_samples": 12,
                "num_warmup": 10,
                "target_accept_prob": 0.8,
                "plot_diagnostics": False,
            },
            "stopping_criteria": {"max_new_mods_per_iter": 3},
        },
        settings,
        prior,
        output_directory=tmp_path,
    )
    with caplog.at_level("INFO", logger="tnt.parameter_generator"):
        first = generator.generate_parameters(AllModels(), run_id=17)
        second = generator.generate_parameters(AllModels(), run_id=18)
    assert first.diagnostics["proposals"]["duplicates_removed"] == 2
    assert len(first.parameters) == len(second.parameters) == 1
    assert second.diagnostics["reservation"]["start"] == 3
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "run 17, iteration 0, round 0, pool [0:3)" in messages
    assert "run 18, iteration 0, round 0, pool [3:6)" in messages
    assert (tmp_path / "parameter_generator" / "prior_samples.npz").is_file()


def test_real_context_identity_is_stable_across_processes(tmp_path):
    import subprocess
    import sys

    script = """
import jax.numpy as jnp
from unxt import Quantity, unitsystem
from tnt.mge import LightMGE
from tnt.priors import Prior
mge = LightMGE(
    I=Quantity(jnp.array([1000.]), 'Lsun / pc2'),
    sigma=Quantity(jnp.array([1.]), 'kpc'),
    q=Quantity(jnp.array([.8]), ''),
    PA_twist=Quantity(jnp.array([0.]), 'rad'),
    major_axis_pa=Quantity(0., 'deg'),
)
print(Prior({}, {}, {'stars': mge},
    cosmological_parameters={'H': Quantity(70., 'km/(s Mpc)')},
    unit_system=unitsystem('kpc', 'Myr', 'Msun', 'rad')).cache_identity())
"""
    results = [
        subprocess.run(
            [sys.executable, "-c", script], check=True, capture_output=True, text=True
        ).stdout
        for _ in range(2)
    ]
    assert results[0] == results[1]


def test_changed_closure_changes_prior_identity():
    def make_plugin(weight):
        def plugin(context):
            return weight

        return plugin

    first = Prior(SETTINGS, {"plugin": make_plugin(1.0)}, {})
    second = Prior(SETTINGS, {"plugin": make_plugin(2.0)}, {})
    assert first.cache_identity() != second.cache_identity()


def test_opaque_callable_still_works_with_caching_disabled(tmp_path):
    g = generator(tmp_path, enabled=False)
    g.prior._cache_plugins = None
    batch, _ = _pool(g, size=8, consume=True)
    assert len(batch.samples["bh.m_tot"]) == 8
    g.generator_settings["cache_prior_samples"] = True
    with pytest.raises(ValueError, match="cache_prior_samples: false"):
        _pool(g, size=8, consume=True)


def test_incomplete_array_archive_does_not_propose_declared_defaults(tmp_path):
    reserve(generator(tmp_path))
    path = tmp_path / "prior_samples.npz"
    with np.load(path, allow_pickle=False) as archive:
        remaining = {
            key: archive[key] for key in archive.files if not key.startswith("sample:")
        }
    np.savez(path, **remaining)
    with pytest.raises(ValueError, match="Cannot resume prior cache"):
        reserve(generator(tmp_path))


def test_two_generators_sharing_prior_have_independent_memory_pools():
    from tnt.all_models import AllModels
    from tnt.parameter_generator import build_parameter_generator

    g = generator(None)
    configuration = {
        "generator_type": "PriorSampler",
        "generator_settings": {
            "seed": g.seed,
            "num_samples": 8,
            "num_warmup": g.num_warmup,
            "target_accept_prob": g.target_accept_prob,
        },
        "stopping_criteria": {"max_new_mods_per_iter": 3},
    }
    first, second = [
        build_parameter_generator(configuration, SETTINGS, g.prior) for _ in range(2)
    ]
    assert first.sample_cache is not second.sample_cache
    a = first.generate_parameters(AllModels())
    b = second.generate_parameters(AllModels())
    next_a = first.generate_parameters(AllModels())
    assert a.diagnostics["reservation"]["start"] == 0
    assert b.diagnostics["reservation"]["start"] == 0
    assert next_a.diagnostics["reservation"]["start"] == 3
    assert len(g.prior.calls) == 2
    assert not hasattr(g.prior, "_sample_pools")


@pytest.mark.parametrize("key", ["fingerprint", "provenance"])
def test_missing_metadata_raises_recovery_error_without_resampling(tmp_path, key):
    reserve(generator(tmp_path))
    path = tmp_path / "prior_samples.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(arrays["metadata"]))
    del metadata[key]
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    np.savez(path, **arrays)
    original = path.read_bytes()
    resumed = generator(tmp_path)
    with pytest.raises(ValueError, match="Cannot resume prior cache"):
        reserve(resumed)
    assert not resumed.prior.calls
    assert path.read_bytes() == original
