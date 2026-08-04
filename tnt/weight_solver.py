"""Solving orbit weights and chi2 metrics against kinematic data.

Signature-only scaffold: every method raises `NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp

from tnt.kinematics import AbstractKinematics
from tnt.orbit_library import OrbitLibrary


class OrbitWeights(eqx.Module):
    """Solved weight of every orbit in an `OrbitLibrary`."""

    weights: jnp.ndarray


class WeightSolverResult(eqx.Module):
    """The outcome of solving one `OrbitLibrary` against `AbstractKinematics`.

    `chi2` holds every metric computed alongside the weights, e.g. `"chi2"`,
    `"kinchi2"`, `"kinmapchi2"` (`parameter_space_settings.which_chi2`
    selects which one drives the parameter search).
    """

    weights: OrbitWeights
    chi2: dict[str, float]


class AbstractWeightSolver(eqx.Module):
    """Solves orbit weights and chi2 metrics for one `OrbitLibrary`.

    `_type` matches `weight_solver_settings.type`.
    """

    _type: ClassVar[str]

    def solve(
        self,
        orbit_library: OrbitLibrary,
        kinematic_data: Sequence[AbstractKinematics],
    ) -> WeightSolverResult:
        """Solve for orbit weights and chi2 metrics against every data set."""
        raise NotImplementedError


class NNLSWeightSolver(AbstractWeightSolver):
    """Non-negative least squares weight solver.

    `solver` names a JAX-native NNLS implementation -- cvxopt/scipy are no
    longer supported, since they aren't traceable/jittable. Which JAX
    option(s) `solver` picks between is still undecided (see
    `weight_solver_settings.nnls_solver` in `configuration_validation.py`);
    once chosen, `solve` becoming jittable removes what was otherwise the
    main obstacle to jitting `ModelIterator._evaluate`.
    """

    _type: ClassVar[str] = "NNLS"
    solver: str
    regularisation: float

    def solve(
        self,
        orbit_library: OrbitLibrary,
        kinematic_data: Sequence[AbstractKinematics],
    ) -> WeightSolverResult:
        raise NotImplementedError


def build_weight_solver(
    weight_solver_settings: Mapping[str, Any],
) -> AbstractWeightSolver:
    """Build the `AbstractWeightSolver` named by a resolved configuration.

    Args:
        weight_solver_settings: A resolved configuration's
            `weight_solver_settings` section.

    Returns:
        The `AbstractWeightSolver` matching `weight_solver_settings.type`.
    """
    raise NotImplementedError
