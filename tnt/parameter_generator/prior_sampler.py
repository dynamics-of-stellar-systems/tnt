"""Parameter proposals from fresh prior draws."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, ClassVar

import jax

from tnt.all_models import AllModels
from tnt.parameter_generator.base import AbstractParameterGenerator, ParameterSet
from tnt.parameter_generator.registry import register_parameter_generator
from tnt.parameter_generator.samples import (
    PriorSampleBatch,
    _format_batch_diagnostics_summary,
    _parameter_sets_from_samples,
)
from tnt.priors import Prior

_LOGGER = logging.getLogger(__name__)


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
    _required_generator_settings: ClassVar[frozenset[str]] = frozenset(
        {"num_samples", "num_warmup", "seed", "target_accept_prob"}
    )

    prior: Prior
    seed: int
    num_samples: int
    num_warmup: int
    target_accept_prob: float

    def generate_with_diagnostics(
        self, all_models: AllModels
    ) -> tuple[list[ParameterSet], dict[str, Any]]:
        """Sample once, finalize proposals and optionally render this batch."""
        # Folding in the cumulative iteration count (not conditioning on
        # fitted scores) is enough to stop a repeated call -- another
        # iteration, or a resumed run -- from redrawing an identical batch.
        iteration = all_models.n_iterations()
        path = self._diagnostic_plot_path("prior_diagnostics", iteration)
        key = jax.random.fold_in(jax.random.PRNGKey(self.seed), iteration)
        samples, diagnostics = self.prior.sample_with_diagnostics(
            key,
            num_samples=self.num_samples,
            num_warmup=self.num_warmup,
            target_accept_prob=self.target_accept_prob,
        )
        _LOGGER.info(
            "PriorSampler batch for iteration %d: %d/%d divergent draws, %s.",
            iteration,
            int(diagnostics.diverging.sum()),
            self.num_samples,
            _format_batch_diagnostics_summary(diagnostics),
        )
        candidates, selection = self._finalize_proposals(
            _parameter_sets_from_samples(samples, self.potential_settings), all_models
        )
        batch = PriorSampleBatch(samples, diagnostics)
        info = {"sample_batch": batch, "proposals": selection, "iteration": iteration}
        if path is not None:
            from tnt.parameter_generator.diagnostics.prior import plot_prior_diagnostics

            plot_prior_diagnostics(batch, path, iteration=iteration)
        return candidates, info

    def generate_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        """Return the same finalized candidates as the diagnostic interface."""
        return self.generate_with_diagnostics(all_models)[0]

    def _propose_free_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        return self.generate_with_diagnostics(all_models)[0]
