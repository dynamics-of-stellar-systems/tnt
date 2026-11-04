"""Parameter proposals from fresh prior draws."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import equinox as eqx
import numpy as np

from tnt.all_models import AllModels
from tnt.parameter_generator.base import AbstractParameterGenerator
from tnt.parameter_generator.prior_cache import PriorSampleCache
from tnt.parameter_generator.registry import register_parameter_generator
from tnt.parameter_generator.samples import (
    _parameter_sets_from_samples,
)
from tnt.parameter_generator.types import ParameterBatch
from tnt.priors import Prior


@register_parameter_generator
class PriorSampler(AbstractParameterGenerator):
    """Proposes parameters by sampling from a `tnt.priors.Prior`.

    By default, num_samples draws form a reusable pool. Calls reserve up to
    max_new_mods_per_iter consecutive draws before execution. Exhaustion starts
    a new round keyed by the configured seed and persisted round number.
    Duplicate filtering may reduce a reservation; no replacements are drawn.
    With caching disabled, the sampling key folds in all_models.n_iterations()
    and each call draws num_samples fresh candidates before filtering/capping.
    Neither path uses fitted scores. Prior owns composition and MCMC selection.

    Every call logs diagnostics of its actual sampling batch. When
    `plot_diagnostics` is enabled, the generator also renders that same batch
    after shared proposal filtering, with a cumulative model-search iteration
    label. Plotting performs no additional sampling.
    """

    _type: ClassVar[str] = "PriorSampler"
    _diagnostic_plot_name: ClassVar[str] = "prior_diagnostics"
    _required_generator_settings: ClassVar[frozenset[str]] = frozenset(
        {"num_samples", "num_warmup", "seed", "target_accept_prob"}
    )

    prior: Prior
    sample_cache: PriorSampleCache = eqx.field(
        default_factory=PriorSampleCache,
        kw_only=True,
    )
    seed: int
    num_samples: int
    num_warmup: int
    target_accept_prob: float

    def _generate(
        self, all_models: AllModels, *, record_history: bool = False
    ) -> ParameterBatch:
        """Draw one batch and supply raw candidates and sampling data."""
        pool, reservation = self.sample_cache.get(
            self.prior,
            potential_settings=self.potential_settings,
            seed=self.seed,
            size=self.num_samples,
            num_warmup=self.num_warmup,
            target_accept_prob=self.target_accept_prob,
            output_directory=self.output_directory,
            enabled=self.generator_settings.get("cache_prior_samples", True),
            reservation_size=self.max_new_mods_per_iter,
            iteration=all_models.n_iterations(),
        )
        samples = pool.samples
        if reservation is not None:
            samples = {
                name: values[reservation["start"] : reservation["stop"]]
                for name, values in samples.items()
            }
        return ParameterBatch(
            _parameter_sets_from_samples(samples, self.potential_settings),
            {
                "sample_batch": pool,
                "sampling": pool.diagnostics,
                "reservation": reservation,
            },
        )

    def _plot_diagnostics(self, batch: ParameterBatch, path: Path) -> None:
        from tnt.parameter_generator.diagnostics.prior import plot_prior_diagnostics

        proposed = {}
        for site in batch.diagnostics["sample_batch"].samples:
            component, parameter = site.split(".")
            unit = self.potential_settings[component]["parameters"][parameter].get(
                "unit", ""
            )
            proposed[site] = np.array(
                [
                    float(point[component][parameter].ustrip(unit))
                    for point in batch.parameters
                ]
            )
        plot_prior_diagnostics(
            batch.diagnostics["sample_batch"],
            path,
            iteration=batch.diagnostics["iteration"],
            proposed_samples=proposed,
        )
