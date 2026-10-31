"""Sequential prior support-point design orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np

from tnt.all_models import AllModels
from tnt.parameter_generator.base import AbstractParameterGenerator
from tnt.parameter_generator.registry import register_parameter_generator
from tnt.parameter_generator.samples import (
    PriorSampleBatch,
    _parameter_sets_from_samples,
)
from tnt.parameter_generator.types import ParameterBatch
from tnt.priors import Prior, PriorDiagnostics
from tnt.support_points import SupportPointDiagnostics, optimize_support_points


@register_parameter_generator
class PriorSupportPoints(AbstractParameterGenerator):
    """Augment existing locations with a batch of prior support points.

    Sampling and physical validation remain eager orchestration. The numerical
    kernel receives scaled arrays, with no knowledge of priors or model fits.
    Only unconstrained coordinates are supported. The iteration cap determines
    the new-point count; existing AllModels locations remain fixed.
    """

    _type: ClassVar[str] = "PriorSupportPoints"
    _diagnostic_plot_name: ClassVar[str] = "support_points_diagnostics"
    _required_generator_settings: ClassVar[frozenset[str]] = frozenset(
        {
            "num_samples",
            "num_warmup",
            "seed",
            "target_accept_prob",
        }
    )

    prior: Prior
    seed: int
    num_samples: int
    num_warmup: int
    target_accept_prob: float

    def _design_settings(self) -> tuple[int, int]:
        """Check static design settings before sampling or tracing."""
        settings = self.generator_settings
        count = self.max_new_mods_per_iter
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("max_new_mods_per_iter must be a positive integer")
        batch_size = settings.get("chunk_size", 1024)
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer")
        return count, batch_size

    def _existing_locations(
        self, all_models: AllModels, names: Sequence[str]
    ) -> np.ndarray:
        """Read all existing sample-site coordinates, converting declared units."""
        rows = np.empty((len(all_models), len(names)))
        if not len(all_models):
            return rows
        for index, site in enumerate(names):
            if site not in all_models.table.colnames:
                raise ValueError(f"AllModels is missing prior parameter column: {site}")
            component, parameter = site.split(".")
            unit = self.potential_settings[component]["parameters"][parameter].get(
                "unit", ""
            )
            try:
                rows[:, index] = au.Quantity(all_models.table[site]).to_value(unit)
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"Cannot read AllModels parameter {site} in unit {unit!r}"
                ) from error
        if not np.isfinite(rows).all():
            raise ValueError("AllModels contains non-finite prior parameter values")
        if len(rows) and len(np.unique(rows, axis=0)) != len(rows):
            raise ValueError(
                "AllModels contains duplicate existing parameter locations"
            )
        if len(rows) and not self._feasible_rows(rows, names).all():
            raise ValueError(
                "AllModels contains locations outside the current prior or geometry"
            )
        return rows

    def _feasible_rows(self, rows: np.ndarray, names: Sequence[str]) -> np.ndarray:
        """Evaluate the prior and geometry predicates for a batch of raw rows."""
        check = jax.jit(
            jax.vmap(
                lambda row: self.prior.feasibility(
                    {name: row[index] for index, name in enumerate(names)}
                )
            )
        )
        return np.asarray(check(jnp.asarray(rows))) & np.isfinite(rows).all(1)

    def _sample_cloud(
        self,
        count: int,
        batch_size: int,
        keep_samples: bool = False,
    ) -> tuple[list[str], np.ndarray, PriorDiagnostics, int, PriorSampleBatch | None]:
        """Draw a reference and separate initialization, retaining diagnostics."""
        draws, diagnostics = self.prior.sample_with_diagnostics(
            jax.random.PRNGKey(self.seed),
            num_samples=self.num_samples + count,
            num_warmup=self.num_warmup,
            target_accept_prob=self.target_accept_prob,
        )
        names = sorted(draws)
        if any(np.asarray(draws[name]).ndim != 1 for name in names):
            raise ValueError(
                "PriorSupportPoints currently requires scalar parameter sites"
            )
        rows = (
            np.column_stack([np.asarray(draws[name], dtype=float) for name in names])
            if names
            else np.empty((self.num_samples + count, 0))
        )
        valid = np.concatenate(
            [
                self._feasible_rows(rows[start : start + batch_size], names)
                for start in range(0, len(rows), batch_size)
            ]
        )
        discarded = int((~valid).sum())
        rows = rows[valid]
        if len(rows) <= count:
            raise ValueError("Too few feasible prior draws for the support design")
        batch = PriorSampleBatch(draws, diagnostics) if keep_samples else None
        return names, rows, diagnostics, discarded, batch

    def _coordinate_maps(
        self,
        rows: np.ndarray,
        names: Sequence[str],
        batch_size: int,
    ) -> tuple[np.ndarray, Callable[[jax.Array], jax.Array]]:
        """Encode the cloud and prepare a conditional, traceable inverse map."""

        def as_sites(row: jax.Array) -> dict[str, jax.Array]:
            return {name: row[index] for index, name in enumerate(names)}

        if names:

            def encode(row: jax.Array) -> jax.Array:
                mapped = self.prior.unconstrain(as_sites(row))
                return jnp.stack([mapped[name] for name in names])

            encode_batch = jax.jit(jax.vmap(encode))
            rows = np.concatenate(
                [
                    np.asarray(
                        encode_batch(jnp.asarray(rows[start : start + batch_size])),
                        dtype=float,
                    )
                    for start in range(0, len(rows), batch_size)
                ]
            )
            if not np.isfinite(rows).all():
                raise ValueError(
                    "Prior draws map to non-finite unconstrained coordinates; "
                    "check boundary values and numerical precision. "
                    "No clipping applied."
                )

        @jax.jit
        def decode(row: jax.Array) -> jax.Array:
            if not names:
                return row
            mapped = self.prior.constrain(as_sites(row))
            return jnp.stack([mapped[name] for name in names])

        return rows, decode

    def _optimize_cloud(
        self,
        rows: np.ndarray,
        count: int,
        names: Sequence[str],
        decode: Callable[[jax.Array], jax.Array],
        fixed: np.ndarray,
        record_history: bool = False,
    ) -> tuple[np.ndarray, SupportPointDiagnostics, dict[str, Any]]:
        """Standardize coordinates and invoke the prior-independent JAX solver."""
        reference, initial = rows[:-count], rows[-count:]
        active = np.ptp(reference, axis=0) > 0
        center = np.where(active, reference.mean(0), reference[0])
        scale = np.where(active, reference.std(0), 1.0)

        def unscale(point: jax.Array) -> jax.Array:
            return jnp.where(
                jnp.asarray(active),
                point * jnp.asarray(scale) + jnp.asarray(center),
                jnp.asarray(center),
            )

        controls = {
            key: self.generator_settings[key]
            for key in (
                "max_iterations",
                "tolerance",
                "chunk_size",
                "distance_epsilon",
            )
            if key in self.generator_settings
        }
        points, diagnostics = optimize_support_points(
            (reference - center) / scale,
            (initial - center) / scale,
            fixed_points=(fixed - center) / scale,
            record_history=record_history,
            **controls,
        )
        raw = np.asarray(
            jax.jit(jax.vmap(lambda point: decode(unscale(point))))(jnp.asarray(points))
        )
        metadata = {
            "reference_sample_size": len(reference),
            "site_names": tuple(names),
            "center": center,
            "scale": scale,
            "constant_dimensions": tuple(np.array(names)[~active]),
            "existing_support_points": len(fixed),
            "new_support_points": count,
        }
        return raw, diagnostics, metadata

    def _generate(
        self,
        all_models: AllModels,
        *,
        record_history: bool = False,
    ) -> ParameterBatch:
        """Resample the reference and optimize new points with history fixed."""
        count, batch_size = self._design_settings()
        keep_data = record_history
        names, rows, sampling, discarded, batch = self._sample_cloud(
            count, batch_size, keep_samples=keep_data
        )
        fixed = self._existing_locations(all_models, names)
        raw_fixed = fixed
        # Transform old and new locations through the same conditional model.
        rows, decode = self._coordinate_maps(rows, names, batch_size)
        if len(fixed):
            fixed, _ = self._coordinate_maps(fixed, names, batch_size)
        raw, optimization, diagnostics = self._optimize_cloud(
            rows, count, names, decode, fixed, keep_data
        )
        if not self._feasible_rows(raw, names).all():
            raise ValueError(
                "Mapped support points failed final feasibility validation; "
                "distribution transforms do not encode every geometry or plugin "
                "constraint. No candidates returned."
            )
        selected = {name: raw[:, index] for index, name in enumerate(names)}
        candidates = _parameter_sets_from_samples(selected, self.potential_settings)
        diagnostics.update(
            optimization=optimization,
            sampling=sampling,
            discarded_infeasible_draws=discarded,
        )
        if keep_data:
            diagnostics.update(
                sample_batch=batch,
                fixed_points={name: raw_fixed[:, i] for i, name in enumerate(names)},
                parameter_units={
                    f"{component}.{name}": spec.get("unit", "")
                    for component, settings in self.potential_settings.items()
                    for name, spec in settings.get("parameters", {}).items()
                },
            )
        return ParameterBatch(candidates, diagnostics)

    def _plot_diagnostics(self, batch: ParameterBatch, path: Path) -> None:
        from tnt.parameter_generator.diagnostics.support_points import (
            plot_prior_support_points_diagnostics,
        )

        plot_prior_support_points_diagnostics(batch.parameters, batch.diagnostics, path)
