"""One replaceable pool of constrained prior draws and pre-execution reservations."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import jax
import numpy as np

from tnt.parameter_generator.samples import PriorSampleBatch
from tnt.priors import Prior, PriorDiagnostics

_LOGGER = logging.getLogger(__name__)
_VERSION = 1


@lru_cache(maxsize=1)
def _software_identity() -> dict[str, str]:
    """Conservatively invalidate on implementation or numerical-stack changes."""
    root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {
        "tnt_source": digest.hexdigest(),
        **{
            name: importlib.metadata.version(name)
            for name in (
                "jax",
                "jaxlib",
                "numpy",
                "numpyro",
                "unxt",
                "equinox",
                "galax",
                "astropy",
                "scipy",
            )
        },
        "galax_source": (
            importlib.metadata.distribution("galax").read_text("direct_url.json") or ""
        ),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "devices": str([device.device_kind for device in jax.devices()]),
        "jax_enable_x64": str(jax.config.jax_enable_x64),
        "backend": jax.default_backend(),
    }


def _write(path: Path, state: dict[str, Any], batch: PriorSampleBatch) -> None:
    """Publish samples, diagnostics and cursor together; retain old data on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".prior_samples-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(
                stream,
                metadata=np.asarray(json.dumps(state)),
                diverging=np.asarray(batch.diagnostics.diverging),
                **{
                    f"sample:{name}": np.asarray(value)
                    for name, value in batch.samples.items()
                },
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path: Path) -> tuple[dict[str, Any], PriorSampleBatch]:
    """Read data only, never pickle; corruption is an error, not a silent rewind."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            state = json.loads(str(archive["metadata"]))
            samples = {
                name.removeprefix("sample:"): archive[name]
                for name in archive.files
                if name.startswith("sample:")
            }
            diagnostics = PriorDiagnostics(
                archive["diverging"],
                state["ess"],
                state["split_rhat"],
                state["sample_distributions"],
            )
        # Validate identity metadata here so malformed archives receive the
        # same recovery error as missing arrays, before reuse or invalidation.
        fingerprint = state["fingerprint"]
        provenance = state["provenance"]
        size = state["size"]
        if (
            not isinstance(fingerprint, str)
            or not isinstance(provenance, dict)
            or state["version"] != _VERSION
            or not isinstance(size, int)
            or size < 1
            or not 0 <= state["cursor"] <= size
            or state["round"] < 0
            or state["reserved"] < state["cursor"]
            or diagnostics.diverging.shape != (size,)
            or set(samples) != set(diagnostics.ess)
            or set(samples) != set(diagnostics.split_rhat)
            or any(len(value) != size for value in samples.values())
        ):
            raise ValueError("invalid pool metadata or array dimensions")
        return state, PriorSampleBatch(samples, diagnostics)
    except (ValueError, KeyError, TypeError, OSError, EOFError, BadZipFile) as error:
        raise ValueError(
            f"Cannot resume prior cache {path}: {error}. "
            "Restore the cache or explicitly remove it to begin a new sequence."
        ) from error


@dataclass(eq=False)
class PriorSampleCache:
    """Own one current pool independently of the Prior and generator ABC.

    The factory supplies this dependency to prior-based generators. Directly
    constructed generators get their own cache by default. One coordinator must
    own each output directory; concurrent reservations are unsupported.
    """

    _cached: tuple[dict[str, Any], PriorSampleBatch] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def get(
        self,
        prior: Prior,
        *,
        potential_settings: Mapping[str, Mapping[str, Any]],
        seed: int,
        size: int,
        num_warmup: int,
        target_accept_prob: float,
        output_directory: Path | None,
        enabled: bool = True,
        reservation_size: int | None = None,
        iteration: int = 0,
    ) -> tuple[PriorSampleBatch, dict[str, int] | None]:
        """Reuse a reference or persist a proposal reservation before execution.

        Without an output directory, state lives only on this cache object and
        cannot survive process restart. The Prior only supplies draws/identity.
        """
        consume = reservation_size is not None
        if not isinstance(enabled, bool):
            raise TypeError("generator_settings.cache_prior_samples must be a boolean")
        if not enabled:
            key = (
                jax.random.fold_in(jax.random.PRNGKey(seed), iteration)
                if consume
                else jax.random.PRNGKey(seed)
            )
            samples, diagnostics = prior.sample_with_diagnostics(
                key,
                num_samples=size,
                num_warmup=num_warmup,
                target_accept_prob=target_accept_prob,
            )
            return PriorSampleBatch(samples, diagnostics), None

        provenance = {
            "prior": prior.cache_identity(),
            "software": _software_identity(),
            "seed": seed,
            "size": size,
            "num_warmup": num_warmup,
            "target_accept_prob": target_accept_prob,
            "purpose": "proposals" if consume else "reference",
            "units": {
                f"{component}.{name}": parameter.get("unit", "")
                for component, settings in potential_settings.items()
                for name, parameter in settings.get("parameters", {}).items()
            },
        }
        fingerprint = hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()
        ).hexdigest()
        path = (
            output_directory / "prior_samples.npz"
            if output_directory is not None
            else None
        )
        # Disk is authoritative, including when another Prior instance has reserved.
        cached = (
            (_read(path) if path.exists() else None)
            if path is not None
            else self._cached
        )
        if cached is not None and cached[0]["fingerprint"] != fingerprint:
            _LOGGER.info(
                "Prior cache invalidated: prior, sampling settings or software changed."
            )
            cached = None
        if cached is None or (consume and cached[0]["cursor"] == size):
            round_index = 0 if cached is None else cached[0]["round"] + 1
            samples, diagnostics = prior.sample_with_diagnostics(
                (
                    jax.random.fold_in(jax.random.PRNGKey(seed), round_index)
                    if consume
                    else jax.random.PRNGKey(seed)
                ),
                num_samples=size,
                num_warmup=num_warmup,
                target_accept_prob=target_accept_prob,
            )
            batch = PriorSampleBatch(samples, diagnostics)
            state = {
                "version": _VERSION,
                "fingerprint": fingerprint,
                "provenance": provenance,
                "round": round_index,
                "cursor": 0,
                "size": size,
                "reserved": 0 if cached is None else cached[0]["reserved"],
                "ess": diagnostics.ess,
                "split_rhat": diagnostics.split_rhat,
                "sample_distributions": diagnostics.sample_distributions,
            }
        else:
            state, batch = cached
            state = dict(state)
        reservation = None
        if consume:
            start = state["cursor"]
            stop = min(size, start + reservation_size)
            state["cursor"] = stop
            state["reserved"] += stop - start
            reservation = {
                "round": state["round"],
                "start": start,
                "stop": stop,
                "reserved": state["reserved"],
            }
        # References only need writing when created/replaced.
        if path is not None and (consume or cached is None):
            _write(path, state, batch)
        self._cached = (state, batch)
        return batch, reservation
