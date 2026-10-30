"""Common proposal contract and declared parameter coordinates."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import equinox as eqx
from unxt import Quantity

from tnt.all_models import AllModels
from tnt.parameter_generator.coordinates import (
    existing_parameter_rows,
    parameter_units,
    proposal_key,
)
from tnt.parameter_generator.outputs import diagnostic_path
from tnt.potential import raw_parameter_dimensions
from tnt.validation import _mapping, _number, _required

_LOGGER = logging.getLogger(__name__)
ParameterSet = dict[str, dict[str, Quantity]]
"""One proposed point in parameter space: component name -> {parameter name: value}.

Mass parameters (`ml`, `mge_mass_scale`, ...) and shape parameters are both
just entries here -- `AbstractParameterGenerator` doesn't distinguish
between them. `parameter_space_settings.potential_rescalings` is a
separate, `ModelIterator`-owned mechanism layered on top of whichever
`ParameterSet`s the generator proposes; it isn't part of this type. Each
`Quantity` keeps whatever unit its parameter was declared in -- nothing
here converts into a shared internal unit system (see `tnt.potential`'s
module docstring for why).
"""


def _declared_parameter_quantity(
    parameter: Mapping[str, Any], dimension: str | None, path: str
) -> Quantity:
    """Parse one declared potential parameter's `{value, unit}` into a `Quantity`.

    Returns a `Quantity` in its own **declared** unit -- no unit-system
    conversion here, matching `tnt.spatial_binnings`' `_declared_angle_quantity`.
    `dimension=None` (not a recognized parameter) and `dimension="dimensionless"`
    both mean no unit -- `unit` must be absent then, treated identically to
    configuration validation's handling of the same distinction. The declared
    unit's *physical* correctness is already guaranteed by
    `tnt.configuration.validation.validate_configuration_quantities`, which
    runs during configuration resolution, well before any
    `AbstractParameterGenerator` exists -- this only checks `unit`'s
    presence matches `dimension`, the same structural check that
    validation already enforces, not a re-validation of unit correctness.
    """
    value = _number(_required(parameter, "value", path), f"{path}.value")
    if dimension is None or dimension == "dimensionless":
        if "unit" in parameter:
            raise ValueError(
                f"{path}.unit is not supported because this parameter is "
                "dimensionless or does not yet have a declared dimension."
            )
        return Quantity(value, "")
    if "unit" not in parameter:
        raise ValueError(f"{path} is missing required field: unit.")
    return Quantity(value, parameter["unit"])


class AbstractParameterGenerator(eqx.Module):
    """Proposes the next round of parameters to evaluate, given `AllModels` so far.

    `_type` matches `parameter_space_settings.generator_type`.

    `potential_settings` (a resolved configuration's `potential` section,
    as declared -- every parameter's value, unit, and, for a search, its
    declared `prior`) is common to every generator, since even a fixed
    single point is drawn from it.
    `generator_settings` is
    `parameter_space_settings.generator_settings`; not every generator type
    needs any of it, so `_required_generator_settings` names the subset of
    its keys this `_type` actually reads. Configuration validation reads this
    static metadata through the registry without constructing generators.
    `max_new_mods_per_iter` (from `parameter_space_settings.stopping_criteria`,
    shared and generic -- not any one generator's own setting) caps how many
    candidates `generate_parameters` returns per call, uniformly across
    every generator regardless of how many its own `_propose_free_parameters`
    happens to produce.

    `output_directory`, when configured, is the generator-owned subdirectory.
    The factory derives it from the run output root; direct construction may
    leave it unset when no default output destinations are needed.
    """

    _type: ClassVar[str]
    _required_generator_settings: ClassVar[frozenset[str]] = frozenset()

    potential_settings: Mapping[str, Mapping[str, Any]]
    generator_settings: Mapping[str, Any]
    max_new_mods_per_iter: int
    output_directory: Path | None = eqx.field(default=None, static=True, kw_only=True)

    def generate_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        """Propose the next round of `ParameterSet`s, capped at `max_new_mods_per_iter`.

        Args:
            all_models: Every model evaluated so far (empty on the first
                iteration).
        """
        candidates, _ = self._finalize_proposals(
            self._propose_free_parameters(all_models), all_models
        )
        return candidates

    def _finalize_proposals(
        self, candidates: Sequence[ParameterSet], all_models: AllModels
    ) -> tuple[list[ParameterSet], dict[str, int]]:
        """Remove exact duplicates before capping, preserving proposal order.

        Identity includes every declared parameter, including fixed ones, after
        compatible unit conversion. Failed existing evaluations also count.
        No separation tolerance or automatic refill is introduced.
        """
        units = parameter_units(self.potential_settings)
        seen = {tuple(row) for row in existing_parameter_rows(all_models, units)}
        unique = []
        for candidate in candidates:
            key = proposal_key(candidate, units)
            if key not in seen:
                unique.append(candidate)
                seen.add(key)
        selected = unique[: self.max_new_mods_per_iter]
        diagnostics = {
            "proposed": len(candidates),
            "duplicates_removed": len(candidates) - len(unique),
            "capped": len(unique) - len(selected),
            "returned": len(selected),
        }
        _LOGGER.info(
            "%s proposals: %d proposed, %d duplicates removed, %d capped, %d returned.",
            self._type,
            diagnostics["proposed"],
            diagnostics["duplicates_removed"],
            diagnostics["capped"],
            diagnostics["returned"],
        )
        return selected, diagnostics

    def diagnostic_path(self, name: str, iteration: int | None = None) -> Path:
        """Choose a diagnostic destination within this generator's output folder."""
        if self.output_directory is None:
            raise ValueError(
                "Generator output_directory is not configured; provide a plot path "
                "or build the generator with the run output_directory."
            )
        return diagnostic_path(self.output_directory, name, iteration)

    def _diagnostic_plot_path(self, name: str, iteration: int) -> Path | None:
        """Resolve an optional plot destination for generators supporting plots."""
        enabled = self.generator_settings.get("plot_diagnostics", False)
        if not isinstance(enabled, bool):
            raise TypeError("generator_settings.plot_diagnostics must be a boolean")
        return self.diagnostic_path(name, iteration) if enabled else None

    def _propose_free_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        """This generator's own proposal logic, before the shared size cap.

        Args:
            all_models: Every model evaluated so far (empty on the first
                iteration).
        """
        raise NotImplementedError

    def _declared_component_values(
        self, component_name: str, component: Mapping[str, Any]
    ) -> dict[str, Quantity]:
        """This component's parameters, as declared, each as a `Quantity`.

        Every `AbstractParameterGenerator` subclass proposing fixed/declared
        (rather than sampled) values reads them through here, so "every such
        generator returns unit-ful `Quantity`s in their own declared unit"
        is a property of this base class, not something each subclass has
        to remember to do itself.
        """
        dimensions = raw_parameter_dimensions(
            component.get("type"), component.get("parameterization")
        )
        return {
            parameter_name: _declared_parameter_quantity(
                _mapping(
                    parameter,
                    f"potential.{component_name}.parameters.{parameter_name}",
                ),
                dimensions.get(parameter_name),
                f"potential.{component_name}.parameters.{parameter_name}",
            )
            for parameter_name, parameter in component.get("parameters", {}).items()
        }
