"""Proposing the next round of potential parameters to evaluate.

Mostly a signature-only scaffold -- `GridSearchParameterGenerator` still
raises `NotImplementedError` -- except `SinglePointParameterGenerator` and
`PriorSampler` and `PriorSupportPoints`, which are fully implemented.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import astropy.units as au
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
from unxt import Quantity

from tnt.all_models import AllModels
from tnt.potential import raw_parameter_dimensions
from tnt.priors import Prior, PriorDiagnostics
from tnt.registry import register_typed_class
from tnt.support_points import SupportPointDiagnostics, optimize_support_points
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
    its keys this `_type` actually reads -- `tnt.configuration.validation`
    mirrors this set (it can't import this module -- see its own note)
    to validate `generator_settings` against whichever `generator_type`
    is configured.
    `max_new_mods_per_iter` (from `parameter_space_settings.stopping_criteria`,
    shared and generic -- not any one generator's own setting) caps how many
    candidates `generate_parameters` returns per call, uniformly across
    every generator regardless of how many its own `_propose_free_parameters`
    happens to produce.
    """

    _type: ClassVar[str]
    _required_generator_settings: ClassVar[frozenset[str]] = frozenset()

    potential_settings: Mapping[str, Mapping[str, Any]]
    generator_settings: Mapping[str, Any]
    max_new_mods_per_iter: int

    def generate_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        """Propose the next round of `ParameterSet`s, capped at `max_new_mods_per_iter`.

        Args:
            all_models: Every model evaluated so far (empty on the first
                iteration).
        """
        return list(self._propose_free_parameters(all_models))[
            : self.max_new_mods_per_iter
        ]

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


_PARAMETER_GENERATOR_REGISTRY: dict[str, type[AbstractParameterGenerator]] = {}


def register_parameter_generator(
    cls: type[AbstractParameterGenerator],
) -> type[AbstractParameterGenerator]:
    """Register one concrete parameter generator for configuration dispatch."""
    return register_typed_class(
        _PARAMETER_GENERATOR_REGISTRY,
        cls,
        family="parameter generator",
    )


def get_parameter_generator_class(
    type_name: str,
) -> type[AbstractParameterGenerator] | None:
    """Return the registered generator class for ``type_name``, if any."""
    return _PARAMETER_GENERATOR_REGISTRY.get(type_name)


def parameter_generator_type_names() -> frozenset[str]:
    """Return every explicitly registered parameter-generator type name."""
    return frozenset(_PARAMETER_GENERATOR_REGISTRY)


def parameter_generator_required_settings() -> dict[str, frozenset[str]]:
    """Return required configuration-setting names for every generator type."""
    return {
        type_name: cls._required_generator_settings
        for type_name, cls in _PARAMETER_GENERATOR_REGISTRY.items()
    }


@register_parameter_generator
class GridSearchParameterGenerator(AbstractParameterGenerator):
    """Proposes parameters on a grid, per each parameter's declared `prior`."""

    _type: ClassVar[str] = "GridSearch"
    _required_generator_settings: ClassVar[frozenset[str]] = frozenset(
        {"delta_chi2_threshold"}
    )

    def _propose_free_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        raise NotImplementedError


@register_parameter_generator
class SinglePointParameterGenerator(AbstractParameterGenerator):
    """Proposes one fixed `ParameterSet`, taken from each parameter's declared value.

    Ignores `all_models` and always proposes the same point -- meant for
    evaluating one nominal potential rather than searching parameter space.
    `parameter_space_settings.stopping_criteria` (e.g. `n_new_iter: 1`) is
    what stops the search after it's evaluated. Needs no `generator_settings`.
    Never looks at `priors:` -- a single fixed point needs every parameter
    fully declared directly (`value`/`fixed`), not sampled.
    """

    _type: ClassVar[str] = "SinglePoint"

    def _propose_free_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        return [
            {
                component_name: self._declared_component_values(
                    component_name, component
                )
                for component_name, component in self.potential_settings.items()
            }
        ]


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
        # degenerate setup (see `tnt.priors_diagnostic_plot
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


@register_parameter_generator
class PriorSampler(AbstractParameterGenerator):
    """Proposes parameters by sampling from a `tnt.priors.Prior`.

    Reads only `all_models.n_iterations()`, to fold into its PRNG key so
    each call draws a fresh batch rather than repeating the same one --
    never the fitted chi2/scores `all_models` also holds, so this is not
    score-conditioned; a generator that refines its proposals from fitted
    scores is a distinct, future generator. A thin adapter beyond that: all
    model composition, factor detection, and the `Predictive`-vs-`MCMC`
    choice live on `Prior` itself, reusable by whatever generator needs
    them next -- this class only derives a PRNG key, calls `Prior.sample`,
    and converts the result into `ParameterSet`s.

    `num_samples` is this sampler's own setting -- how large a batch
    `Prior.sample` actually draws each call -- independent of
    `max_new_mods_per_iter` (the generic, cross-generator cap on how many
    candidates `generate_parameters` *returns*, from
    `AbstractParameterGenerator`). They used to be conflated (the same
    value served both jobs), which broke as soon as that value was small:
    `Prior.sample_with_diagnostics` always computes a split R-hat, and
    `numpyro.diagnostics.split_gelman_rubin` requires at least 4 draws to
    do that regardless of how few candidates the caller actually wants.
    `tnt.configuration.validation` requires `num_samples >= 100` (a batch
    that small isn't statistically meaningful for MCMC/`Predictive`
    sampling either way) and `num_samples >= max_new_mods_per_iter` (so
    this never silently returns fewer candidates than requested) --
    `generate_parameters`'s own `[:max_new_mods_per_iter]` trim still does
    the actual capping, unchanged.

    Each call logs its own batch's divergence count and worst *finite*
    ESS/split-R-hat across every parameter, at `INFO` level, tagged with the
    iteration it was drawn for (see `_format_batch_diagnostics_summary` for
    why "worst finite" and not a plain `min`/`max` -- a site with a
    genuinely non-finite statistic, e.g. an intentionally constant
    `Delta(...)` prior, is named explicitly rather than silently hidden
    behind another site's healthy number). This is the actual batch
    `generate_parameters` goes on to propose from, not
    `ModelIterator.from_configuration`'s separate, independently-keyed
    pilot figure (`tnt.priors_diagnostic_plot.plot_prior_diagnostics`),
    which `jax.random.fold_in` guarantees never matches any real
    iteration's own key, not even the first (`fold_in(key, 0) != key`).
    That pilot draw remains a useful pre-flight sanity check on the
    composed model as a whole; it just isn't a substitute for seeing what
    each real batch's own diagnostics look like.
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

    def _propose_free_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        # Folding in the cumulative iteration count (not conditioning on
        # fitted scores) is enough to stop a repeated call -- another
        # iteration, or a resumed run -- from redrawing an identical batch.
        iteration = all_models.n_iterations()
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
        return _parameter_sets_from_samples(samples, self.potential_settings)


@register_parameter_generator
class PriorSupportPoints(AbstractParameterGenerator):
    """Augment existing locations with a batch of prior support points.

    Sampling and physical validation remain eager orchestration. The numerical
    kernel receives scaled arrays, with no knowledge of priors or model fits.
    Only unconstrained coordinates are supported. The iteration cap determines
    the new-point count; existing AllModels locations remain fixed.
    """

    _type: ClassVar[str] = "PriorSupportPoints"
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
    ) -> tuple[list[str], np.ndarray, PriorDiagnostics, int]:
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
        return names, rows, diagnostics, discarded

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

    def generate_with_diagnostics(
        self,
        all_models: AllModels,
        *,
        record_history: bool = False,
    ) -> tuple[list[ParameterSet], dict[str, Any]]:
        """Resample the reference and optimize new points with history fixed."""
        count, batch_size = self._design_settings()
        names, rows, sampling, discarded = self._sample_cloud(count, batch_size)
        fixed = self._existing_locations(all_models, names)
        # Transform old and new locations through the same conditional model.
        rows, decode = self._coordinate_maps(rows, names, batch_size)
        if len(fixed):
            fixed, _ = self._coordinate_maps(fixed, names, batch_size)
        raw, optimization, diagnostics = self._optimize_cloud(
            rows, count, names, decode, fixed, record_history
        )
        if not self._feasible_rows(raw, names).all():
            raise ValueError(
                "Mapped support points failed final feasibility validation; "
                "distribution transforms do not encode every geometry or plugin "
                "constraint. No candidates returned."
            )
        selected = {name: raw[:, index] for index, name in enumerate(names)}
        candidates = _parameter_sets_from_samples(selected, self.potential_settings)
        if not names:
            candidates *= count
        diagnostics.update(
            optimization=optimization,
            sampling=sampling,
            discarded_infeasible_draws=discarded,
        )
        _LOGGER.info(
            "PriorSupportPoints: objective %.8g, status %s, iterations %d",
            optimization.objective,
            optimization.status,
            optimization.iterations,
        )
        _LOGGER.info(
            "PriorSupportPoints reference: %d feasible draws, %d discarded; %s.",
            len(rows) - count,
            discarded,
            _format_batch_diagnostics_summary(sampling),
        )
        return candidates, diagnostics

    def _propose_free_parameters(self, all_models: AllModels) -> Sequence[ParameterSet]:
        """Adapt the diagnostic interface to the common proposal contract."""
        return self.generate_with_diagnostics(all_models)[0]


def build_parameter_generator(
    parameter_space_settings: Mapping[str, Any],
    potential_settings: Mapping[str, Mapping[str, Any]],
    prior: Prior | None = None,
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
    )
