"""Parameter proposals from fresh prior draws."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import jax

from tnt.all_models import AllModels
from tnt.parameter_generator.base import AbstractParameterGenerator
from tnt.parameter_generator.registry import register_parameter_generator
from tnt.parameter_generator.samples import (
    PriorSampleBatch,
    _parameter_sets_from_samples,
)
from tnt.parameter_generator.types import ParameterBatch
from tnt.priors import Prior


@register_parameter_generator
class PriorSampler(AbstractParameterGenerator):
    """Proposes parameters by sampling from a `tnt.priors.Prior`.

    The sampling key folds in `all_models.n_iterations()` so successive
    iterations draw fresh reproducible batches. Shared finalization also reads
    existing parameter coordinates to remove duplicates. Neither stage uses
    fitted scores. Composition, factor detection and Predictive/MCMC selection
    belong to `Prior`.

    `num_samples` controls the batch drawn before filtering and capping to
    `max_new_mods_per_iter`. Configuration requires `num_samples > 100`,
    `num_warmup > 100` and `num_samples >= max_new_mods_per_iter`. Exact
    duplicates can reduce the returned batch; no replacements are sampled.

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
    seed: int
    num_samples: int
    num_warmup: int
    target_accept_prob: float

    def _generate(
        self, all_models: AllModels, *, record_history: bool = False
    ) -> ParameterBatch:
        """Draw one batch and supply raw candidates and sampling data."""
        key = jax.random.fold_in(
            jax.random.PRNGKey(self.seed), all_models.n_iterations()
        )
        samples, diagnostics = self.prior.sample_with_diagnostics(
            key,
            num_samples=self.num_samples,
            num_warmup=self.num_warmup,
            target_accept_prob=self.target_accept_prob,
        )
        return ParameterBatch(
            _parameter_sets_from_samples(samples, self.potential_settings),
            {
                "sample_batch": PriorSampleBatch(samples, diagnostics),
                "sampling": diagnostics,
            },
        )

    def _plot_diagnostics(self, batch: ParameterBatch, path: Path) -> None:
        from tnt.parameter_generator.diagnostics.prior import plot_prior_diagnostics

        plot_prior_diagnostics(
            batch.diagnostics["sample_batch"],
            path,
            iteration=batch.diagnostics["iteration"],
        )
