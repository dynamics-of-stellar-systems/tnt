"""Build registered generators from resolved runtime settings."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tnt.parameter_generator.base import AbstractParameterGenerator
from tnt.parameter_generator.prior_sampler import PriorSampler
from tnt.parameter_generator.prior_support_points import PriorSupportPoints
from tnt.parameter_generator.registry import (
    get_parameter_generator_class,
    parameter_generator_type_names,
)
from tnt.priors import Prior


def build_parameter_generator(
    parameter_space_settings: Mapping[str, Any],
    potential_settings: Mapping[str, Mapping[str, Any]],
    prior: Prior | None = None,
    *,
    output_directory: str | Path | None = None,
) -> AbstractParameterGenerator:
    """Build the `AbstractParameterGenerator` named by a resolved configuration.

    Args:
        parameter_space_settings: A resolved configuration's
            `parameter_space_settings` section.
        potential_settings: A resolved configuration's `potential` section,
            as declared.
        prior: This run's composed `tnt.priors.Prior`, e.g. from
            `tnt.model_iterator.ModelIterator.from_configuration` -- only
            required by `"PriorSampler"` and `"PriorSupportPoints"`.
        output_directory: Run output root. Generator diagnostics are placed
            in its `parameter_generator` subdirectory. No folder is created
            until an output is written.

    Returns:
        The `AbstractParameterGenerator` matching
        `parameter_space_settings.generator_type`.
    """
    generator_type = parameter_space_settings["generator_type"]
    generator_cls = get_parameter_generator_class(generator_type)
    if generator_cls is None:
        allowed = ", ".join(sorted(parameter_generator_type_names()))
        raise ValueError(
            "Unknown parameter_space_settings.generator_type: "
            f"{generator_type!r}; expected one of: {allowed}."
        )
    generator_settings = parameter_space_settings["generator_settings"]
    max_new_mods_per_iter = parameter_space_settings["stopping_criteria"][
        "max_new_mods_per_iter"
    ]
    generator_output = (
        Path(output_directory) / "parameter_generator"
        if output_directory is not None
        else None
    )
    if generator_cls in (PriorSampler, PriorSupportPoints):
        if prior is None:
            raise ValueError(
                f"build_parameter_generator: generator_type {generator_type!r} "
                "requires a built Prior."
            )
        return generator_cls(
            potential_settings=potential_settings,
            generator_settings=generator_settings,
            max_new_mods_per_iter=max_new_mods_per_iter,
            output_directory=generator_output,
            prior=prior,
            seed=generator_settings["seed"],
            num_samples=generator_settings["num_samples"],
            num_warmup=generator_settings["num_warmup"],
            target_accept_prob=generator_settings["target_accept_prob"],
        )
    return generator_cls(
        potential_settings=potential_settings,
        generator_settings=generator_settings,
        max_new_mods_per_iter=max_new_mods_per_iter,
        output_directory=generator_output,
    )
