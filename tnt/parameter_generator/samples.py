"""Prior-draw conversion and sampling diagnostic summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

from unxt import Quantity

from tnt.parameter_generator.types import ParameterSet
from tnt.priors import PriorDiagnostics


class PriorSampleBatch(NamedTuple):
    """One actual generation batch and its unmodified sampling diagnostics."""

    samples: Mapping[str, Any]
    diagnostics: PriorDiagnostics


def _parameter_sets_from_samples(
    samples: Mapping[str, Any],
    potential_settings: Mapping[str, Mapping[str, Any]],
) -> list[ParameterSet]:
    """Convert `tnt.priors.Prior.sample`'s output into `ParameterSet`s.

    `samples` holds one array per non-fixed, prior-bearing parameter (dotted
    `"component.parameter"` site names -> bare, unit-stripped draws, see
    `Prior.sample`); fixed parameters aren't sample sites at all, so their
    declared `value` is filled in directly here instead. Every value is
    re-wrapped as a `Quantity` in its parameter's own declared `unit`
    (absent means dimensionless), matching `_declared_parameter_quantity`'s
    convention -- nothing here converts into a shared internal unit system.
    """
    num_samples = len(next(iter(samples.values()))) if samples else 1
    parameter_sets: list[ParameterSet] = []
    for index in range(num_samples):
        candidate: ParameterSet = {}
        for component_name, component in potential_settings.items():
            component_values: dict[str, Quantity] = {}
            for parameter_name, parameter in component.get("parameters", {}).items():
                site = f"{component_name}.{parameter_name}"
                value = samples[site][index] if site in samples else parameter["value"]
                component_values[parameter_name] = Quantity(
                    value, parameter.get("unit", "")
                )
            candidate[component_name] = component_values
        parameter_sets.append(candidate)
    return parameter_sets


def _format_batch_diagnostics_summary(diagnostics: PriorDiagnostics) -> str:
    """Summarize one batch's ESS/split-R-hat, naming any non-finite site.

    Plain `min`/`max` over `diagnostics.ess`/`.split_rhat` silently hide a
    `NaN` site behind a healthy-looking extremum whenever a finite value
    happens to come first in iteration order (`min([1.0, nan]) == 1.0`, not
    `nan`) -- and an intentionally constant/point-mass `prior` (e.g. a
    non-fixed parameter declared `Delta(...)`) genuinely produces `NaN` ESS
    and split-R-hat, not a bug. Reporting the worst *finite* value without
    checking for this would describe a batch with an undefined statistic as
    if every site were healthy.
    """
    names = sorted(diagnostics.ess)
    non_finite = [
        name
        for name in names
        if not (
            math.isfinite(diagnostics.ess[name])
            and math.isfinite(diagnostics.split_rhat[name])
        )
    ]
    finite_ess = [diagnostics.ess[name] for name in names if name not in non_finite]
    finite_rhat = [
        diagnostics.split_rhat[name] for name in names if name not in non_finite
    ]
    if not names:
        # No non-fixed, prior-bearing parameter at all -- a supported,
        # degenerate setup (see `tnt.parameter_generator.diagnostics.prior
        # .plot_prior_diagnostics`'s own zero-parameter handling).
        return "no non-fixed parameters to sample"
    if finite_ess:
        summary = (
            f"min ESS {min(finite_ess):.0f}, max split-R-hat {max(finite_rhat):.3f}"
        )
    else:
        summary = "no site has a finite ESS/split-R-hat"
    if non_finite:
        summary += (
            f"; non-finite ESS/split-R-hat at {len(non_finite)} site(s): "
            f"{', '.join(non_finite)}"
        )
    return summary
