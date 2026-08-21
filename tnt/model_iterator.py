"""Orchestrates the generate -> evaluate -> record -> stop? model search loop.

Recurring pattern: the method doing the actual work stays ignorant of
context it doesn't need, and its caller stamps that context on afterward.
`_solve` solves one `OrbitLibrary`'s weights without knowing which
`Potential` it came from; `_evaluate` stamps that `Potential` on to build
each `Model`. `_evaluate` itself doesn't know which search round it's in;
`run` stamps `Model.iteration` on after collecting its results. This keeps
each piece's inputs matching what it actually uses.

Everything here -- including `from_configuration` and both
`minimum_delta_chi2` stopping modes -- is implemented, but only down to what
it delegates to: `build_potential`, `Potential.generate_orbit_library`,
`AbstractWeightSolver.solve`, `OrbitLibrary.rescaled`,
`build_weight_solver`, `build_orbit_sampler`, and `build_orbit_dithering`
are themselves still unimplemented, so `from_configuration`'s result can't
actually run `_evaluate` end-to-end yet.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Self

import equinox as eqx
import numpy as np
from unxt import AbstractUnitSystem

from tnt.all_models import AllModels
from tnt.configuration_compatibility import (
    _critical_configuration,
    ensure_resume_compatible,
)
from tnt.kinematics import AbstractKinematics, build_kinematics
from tnt.mge import LightMGE, MassMGE, build_mges
from tnt.model import Model
from tnt.orbit_library import (
    AbstractOrbitDithering,
    AbstractOrbitSampler,
    OrbitLibrary,
    build_orbit_dithering,
    build_orbit_sampler,
)
from tnt.parameter_generator import (
    AbstractParameterGenerator,
    ParameterSet,
    build_parameter_generator,
)
from tnt.populations import Populations, build_populations
from tnt.potential import Potential, build_potential
from tnt.run_config_log import (
    RunConfigLog,
    RunManifestReference,
)
from tnt.spatial_binnings import build_spatial_binnings
from tnt.units import normalize_potential_settings
from tnt.weight_solver import AbstractWeightSolver, OrbitWeights, build_weight_solver

_LOGGER = logging.getLogger(__name__)


@dataclass
class ModelIterator:
    """Owns the parameter search loop and its stopping decision.

    `potential_settings`/`mges` hold the fixed, per-run potential-component
    structure (types, MGE references, `include` flags) that
    `AbstractParameterGenerator` fills in with concrete parameter values
    each iteration -- see `tnt.potential.build_potential`.
    """

    potential_settings: Mapping[str, Mapping[str, Any]]
    mges: Mapping[str, LightMGE | MassMGE]
    kinematic_data: Mapping[str, AbstractKinematics]
    population_data: Mapping[str, Populations]
    weight_solver: AbstractWeightSolver
    parameter_generator: AbstractParameterGenerator
    orbit_library_settings: Mapping[str, Any]
    orbit_sampler: AbstractOrbitSampler
    orbit_dithering: AbstractOrbitDithering
    potential_rescalings: Mapping[str, Any]
    which_chi2: str
    stopping_criteria: Mapping[str, Any]
    execution_settings: Mapping[str, Any]
    run_id: int
    run_manifest: RunManifestReference
    critical_configuration: Mapping[str, Any]

    @classmethod
    def from_configuration(
        cls,
        config: Mapping[str, Any],
        unit_system: AbstractUnitSystem,
        run_manifest_path: Path,
    ) -> Self:
        """Build a `ModelIterator` from a fully resolved configuration.

        Deliberately takes already-resolved, plain-data inputs rather than
        a `tnt.configuration.Configuration` -- matching `tnt.mge.build_mges`
        and the other `build_*` functions this calls; see that docstring
        for why.

        Runtime construction is ordered by dependency: build named MGEs,
        build named spatial binnings, then call `build_kinematics` and
        `build_populations` with their shared registries and
        `io_settings.input_directory`, before constructing the iterator's
        remaining model-search services.

        Args:
            config: A resolved configuration, e.g. `Configuration.as_dict()`.
            unit_system: The unit system to convert MGE/kinematics/
                population quantities into, e.g.
                `Configuration.unit_systems.internal`.
            run_manifest_path: The immutable run manifest associated with this
                configuration, e.g. `Configuration.run_manifest_path`. Its run
                ID is recorded for every round in `run`'s `RunConfigLog`.
        """
        _require_supported_model_processing_order(config["execution_settings"])
        run_manifest = RunManifestReference.from_run_manifest(run_manifest_path)
        input_directory = config["io_settings"]["input_directory"]
        parameter_space_settings = config["parameter_space_settings"]

        mges = build_mges(config["MGEs"], input_directory, unit_system)
        spatial_binnings = build_spatial_binnings(
            config["spatial_binnings"],
            input_directory,
            unit_system,
            config["mge_settings"]["projected_mass_quad_order"],
        )
        kinematic_data = build_kinematics(
            config["kinematic_data"],
            input_directory,
            unit_system,
            spatial_binnings,
            mges,
        )
        population_data = build_populations(
            config["population_data"], input_directory, unit_system, spatial_binnings
        )
        potential_settings = normalize_potential_settings(
            config["potential"], unit_system
        )

        return cls(
            potential_settings=potential_settings,
            mges=mges,
            kinematic_data=kinematic_data,
            population_data=population_data,
            weight_solver=build_weight_solver(config["weight_solver_settings"]),
            parameter_generator=build_parameter_generator(
                parameter_space_settings, potential_settings
            ),
            orbit_library_settings=config["orbit_library_settings"],
            orbit_sampler=build_orbit_sampler(config["orbit_library_settings"]),
            orbit_dithering=build_orbit_dithering(config["orbit_library_settings"]),
            potential_rescalings=parameter_space_settings["potential_rescalings"],
            which_chi2=parameter_space_settings["which_chi2"],
            stopping_criteria=parameter_space_settings["stopping_criteria"],
            execution_settings=config["execution_settings"],
            run_id=run_manifest.run_id,
            run_manifest=run_manifest,
            critical_configuration=_critical_configuration(config),
        )

    def run(
        self,
        all_models: AllModels | None = None,
        run_config_log: RunConfigLog | None = None,
    ) -> tuple[AllModels, RunConfigLog]:
        """Iterate until a stopping criterion is met.

        Each round asks `parameter_generator` for the next `ParameterSet`s
        (given every model evaluated so far), evaluates each via
        `_evaluate` -- which also produces any `potential_rescalings`
        models -- and appends every resulting `Model` to the running
        `AllModels`. Stops when `parameter_generator` proposes nothing
        more, or once `stopping_criteria` is met: `n_new_iter` rounds have run
        during the current call, `target_model_count` cumulative models have
        been evaluated before a new round starts, or the best `which_chi2`
        chi2 has stopped improving by at least `minimum_delta_chi2` between
        successful rounds. The model count is a soft target: every model in a
        round that has already begun is evaluated, so the final count can
        exceed it.

        A fresh search stops after recording its first round if that round
        produces no successful model, because the parameter generator has no
        valid chi2 base from which to continue. A resumed `AllModels` that
        contains only failed models stops before proposing another round for
        the same reason. Once any successful model exists, a later round that
        produces only failures does not terminate the search or participate in
        the delta-chi2 check; the previous best remains the generator's base.

        `n_new_iter` is a per-call allowance: resuming a previously written
        `AllModels` can run up to that many additional iterations. Iteration
        labels remain cumulative, starting at `all_models.n_iterations()`.
        `target_model_count`, in contrast, is judged cumulatively via
        `len(all_models)`. `minimum_delta_chi2` compares consecutive rounds
        that produce successful models, seeded from `all_models`'s own best if
        it contains a successful model.

        Also records, in `run_config_log`, the ID of the TNT run that produced
        each round -- one row per round, not per `Model`. The immutable run
        manifest links that run ID to its resolved configuration and execution
        provenance.

        Args:
            all_models: Models evaluated in a previous run to resume from,
                or `None` to start fresh.
            run_config_log: The iteration-to-run log from previous runs to
                resume, or `None` to start fresh.

        Returns:
            The final `AllModels` and `RunConfigLog`, covering every
            model and round recorded so far.
        """
        _require_supported_model_processing_order(self.execution_settings)
        models = AllModels() if all_models is None else all_models
        run_config_log = (
            RunConfigLog() if run_config_log is None else run_config_log
        )
        if len(run_config_log) != models.n_iterations():
            raise ValueError(
                "AllModels and RunConfigLog must describe the same number "
                f"of iterations; received {models.n_iterations()} and "
                f"{len(run_config_log)}, respectively."
        )
        ensure_resume_compatible(
            self.critical_configuration,
            run_config_log.baseline_run_reference(
                self.run_manifest.repository,
                current_run_id=self.run_id,
            ),
            models,
            self.which_chi2,
        )
        n_new_iter = self.stopping_criteria["n_new_iter"]
        target_model_count = self.stopping_criteria["target_model_count"]
        initial_iteration_count = models.n_iterations()

        has_successful_model = models.has_successful_model()
        if len(models) and not has_successful_model:
            _LOGGER.warning(
                "Resumed AllModels contains no successful model; stopping before "
                "proposing another iteration."
            )
            _LOGGER.info(
                "Stopped after %d iteration(s), %d model(s) total.",
                models.n_iterations(),
                len(models),
            )
            return models, run_config_log

        previous_best_chi2: float | None = (
            models.best(self.which_chi2)[self.which_chi2]
            if has_successful_model
            else None
        )
        while (
            models.n_iterations() - initial_iteration_count < n_new_iter
            and len(models) < target_model_count
        ):
            proposed = self.parameter_generator.generate_parameters(models)
            if not proposed:
                _LOGGER.info("Parameter generator proposed nothing more; stopping.")
                break

            iteration = models.n_iterations()
            run_config_log = run_config_log.append(iteration, self.run_id)
            n_before = len(models)
            n_successful = 0
            for parameters in proposed:
                for model in self._evaluate(parameters):
                    model = eqx.tree_at(lambda m: m.iteration, model, iteration)
                    models = models.append(model)
                    n_successful += int(model.weights_done)
            _LOGGER.info(
                "Iteration %d: evaluated %d parameter set(s) into %d model(s).",
                iteration,
                len(proposed),
                len(models) - n_before,
            )

            if not n_successful:
                if previous_best_chi2 is None:
                    _LOGGER.warning(
                        "Initial iteration %d produced no successful model; "
                        "stopping because parameter generation has no chi2 base.",
                        iteration,
                    )
                    break
                _LOGGER.info(
                    "Iteration %d produced no successful model; retaining the "
                    "previous best value for %s and continuing.",
                    iteration,
                    self.which_chi2,
                )
                continue

            best_chi2 = models.best(self.which_chi2)[self.which_chi2]
            if previous_best_chi2 is not None and self._chi2_stopped_improving(
                previous_best_chi2, best_chi2
            ):
                _LOGGER.info(
                    "Best %s chi2 stopped improving (%s -> %s); stopping.",
                    self.which_chi2,
                    previous_best_chi2,
                    best_chi2,
                )
                break
            previous_best_chi2 = best_chi2

        _LOGGER.info(
            "Stopped after %d iteration(s), %d model(s) total.",
            models.n_iterations(),
            len(models),
        )
        return models, run_config_log

    def _chi2_stopped_improving(
        self, previous_best_chi2: float, best_chi2: float
    ) -> bool:
        """Whether the best chi2 improved by less than `minimum_delta_chi2`.

        Improvement is `previous_best_chi2 - best_chi2`, since a smaller
        chi2 is better. Absolute mode compares that difference directly;
        relative mode divides it by `previous_best_chi2`. When `enabled` is
        `False`, this stopping criterion is skipped, letting another criterion
        or the parameter generator stop the search instead.

        When the previous best is exactly zero, relative improvement is
        undefined but no lower nonnegative chi2 is possible. A positive
        threshold therefore stops, while a zero threshold does not.

        Args:
            previous_best_chi2: The best `which_chi2` chi2 before this
                round's models were added.
            best_chi2: The best `which_chi2` chi2 after this round's
                models were added.
        """
        threshold = self.stopping_criteria["minimum_delta_chi2"]
        if not threshold["enabled"]:
            return False

        minimum_improvement = threshold["value"]
        improvement = previous_best_chi2 - best_chi2
        if threshold["mode"] == "absolute":
            return improvement < minimum_improvement
        if threshold["mode"] == "relative":
            if previous_best_chi2 == 0:
                return minimum_improvement > 0
            return improvement / previous_best_chi2 < minimum_improvement
        raise ValueError(
            "minimum_delta_chi2.mode must be 'absolute' or 'relative', "
            f"got {threshold['mode']!r}."
        )

    def _evaluate(self, parameters: ParameterSet) -> list[Model]:
        """Evaluate one proposed `ParameterSet`, and its cheap mass rescalings.

        Builds one `Potential`/`OrbitLibrary` for `parameters` --
        `Potential.generate_orbit_library` takes `self.orbit_sampler`/
        `self.orbit_dithering` alongside `self.orbit_library_settings` -- then,
        if `potential_rescalings.enabled`, also produces
        `range_count` additional `Model`s at nearby mass scales via
        `Potential.rescale`/`OrbitLibrary.rescaled`, reusing the same
        orbit-library integration rather than re-integrating once per mass
        scale.

        Each returned `Model.iteration` is a placeholder -- `run` stamps the
        real one on afterward (see module docstring).

        Sets `Model.orblib_done`/`weights_done` (and `weights`/`chi2`) to
        reflect what actually happened, per `Model`'s own docstring, rather
        than assuming success. A failed orbit integration or weight solve is
        caught as a bare `Exception` -- there's no narrower exception type
        established anywhere else in the codebase yet for either failure, so
        this is a placeholder pending one. Deliberately makes only one attempt
        at each; configuration requires
        `weight_solver_settings.reattempt_failures: false` until retry behavior
        is implemented.

        Logs each failure as a `WARNING`, identified by `parameters` (and,
        for a rescaled variant, its mass scale) -- `_evaluate` is what logs
        a failed `_solve`, not `_solve` itself, since only `_evaluate` knows
        which proposed point a given `OrbitLibrary` belongs to (see module
        docstring).
        """
        potential = build_potential(
            _settings_with_parameters(self.potential_settings, parameters),
            self.mges,
        )

        try:
            orbit_library = potential.generate_orbit_library(
                self.orbit_library_settings,
                self.orbit_sampler,
                self.orbit_dithering,
            )
        except Exception as error:  # noqa: BLE001 -- placeholder, see _evaluate's docstring
            _LOGGER.warning("Orbit integration failed for %s: %s", parameters, error)
            return [_unsolved_model(potential)]

        potentials_and_libraries = [(potential, orbit_library, 1.0)]
        if self.potential_rescalings["enabled"]:
            potentials_and_libraries += [
                (
                    potential.rescale(mass_scale),
                    orbit_library.rescaled(mass_scale),
                    mass_scale,
                )
                for mass_scale in self.mass_scales
            ]

        models = []
        for (
            rescaled_potential,
            rescaled_library,
            mass_scale,
        ) in potentials_and_libraries:
            solved = self._solve(rescaled_library)
            if solved.error is not None:
                _LOGGER.warning(
                    "Weight solve failed for %s (mass_scale=%s): %s",
                    parameters,
                    mass_scale,
                    solved.error,
                )
            models.append(_model(rescaled_potential, solved))
        return models

    def _solve(self, orbit_library: OrbitLibrary) -> _SolveResult:
        """Solve one `OrbitLibrary`'s weights against `self.kinematic_data`.

        See `_evaluate`'s docstring for why failures are caught as a bare
        `Exception`, with no retry, and why a failure is reported back via
        `_SolveResult.error` rather than logged here directly.
        """
        try:
            result = self.weight_solver.solve(
                orbit_library, list(self.kinematic_data.values())
            )
        except Exception as error:  # noqa: BLE001 -- placeholder, see _evaluate's docstring
            return _SolveResult(
                weights_done=False, weights=None, chi2=None, error=str(error)
            )

        return _SolveResult(
            weights_done=True, weights=result.weights, chi2=result.chi2, error=None
        )

    @functools.cached_property
    def mass_scales(self) -> tuple[float, ...]:
        """The rescaled mass scales `_evaluate` additionally solves for.

        `range_count` values spanning `potential_rescalings.mass_scale_range`,
        linearly or logarithmically spaced per `spacing` -- excluding `1.0`
        (the unscaled point), which `_evaluate` always solves separately as
        the base model regardless of `potential_rescalings`. Computed once
        per `ModelIterator`, since `potential_rescalings` is fixed for the
        whole run.
        """
        mass_scale_range = self.potential_rescalings["mass_scale_range"]
        range_count = self.potential_rescalings["range_count"]
        spacing = (
            np.geomspace
            if self.potential_rescalings["spacing"] == "logarithmic"
            else np.linspace
        )
        scales = spacing(
            mass_scale_range["minimum"], mass_scale_range["maximum"], range_count
        )
        return tuple(float(scale) for scale in scales if not np.isclose(scale, 1.0))


def _settings_with_parameters(
    potential_settings: Mapping[str, Mapping[str, Any]], parameters: ParameterSet
) -> dict[str, dict[str, Any]]:
    """Overlay a proposed `ParameterSet`'s values onto `potential_settings`.

    `build_potential` expects the same schema as a resolved configuration's
    internal-runtime `potential` section (`fixed`/`value`/... per parameter);
    declared units have already been converted and removed. This
    keeps every declared field other than `value`, which each parameter
    takes from `parameters` instead of the fixed config.
    """
    return {
        component_name: {
            **component,
            "parameters": {
                parameter_name: (
                    {**parameter, "value": parameters[component_name][parameter_name]}
                    if component_name in parameters
                    and parameter_name in parameters[component_name]
                    else parameter
                )
                for parameter_name, parameter in component.get("parameters", {}).items()
            },
        }
        for component_name, component in potential_settings.items()
    }


def _require_supported_model_processing_order(
    execution_settings: Mapping[str, Any],
) -> None:
    """Reject the configured execution order that TNT cannot run yet."""
    model_processing_order = execution_settings["model_processing_order"]
    if model_processing_order == "stage_by_stage":
        raise NotImplementedError(
            "execution_settings.model_processing_order='stage_by_stage' is not "
            "implemented; use 'model_by_model'."
        )
    if model_processing_order != "model_by_model":
        raise ValueError(
            "execution_settings.model_processing_order must be 'model_by_model' "
            f"or 'stage_by_stage', got {model_processing_order!r}."
        )


class _SolveResult(NamedTuple):
    """The outcome of `ModelIterator._solve`, before it's stamped onto a `Potential`.

    `error` is the caught exception's message when `weights_done` is
    `False`, otherwise `None` -- carried back rather than logged by
    `_solve` itself, since only `_evaluate` knows which `Potential` it
    belongs to (see module docstring).
    """

    weights_done: bool
    weights: OrbitWeights | None
    chi2: dict[str, float] | None
    error: str | None


def _model(potential: Potential, solved: _SolveResult) -> Model:
    """A `Model` for `potential`, given its `OrbitLibrary`'s already-solved weights."""
    return Model(
        potential=potential,
        orblib_done=True,
        weights_done=solved.weights_done,
        weights=solved.weights,
        chi2=solved.chi2,
        iteration=0,
    )


def _unsolved_model(potential: Potential) -> Model:
    """A `Model` for `potential`, whose orbit integration itself didn't succeed."""
    return Model(
        potential=potential,
        orblib_done=False,
        weights_done=False,
        weights=None,
        chi2=None,
        iteration=0,
    )
