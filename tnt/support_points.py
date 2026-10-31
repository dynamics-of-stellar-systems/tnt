"""Blocked sp.ccp support points (Mak & Joseph 2018, Section 3, Eq. 22).

The JAX kernel consumes dimensionless coordinates only. Validation, logging,
prior sampling, units, and conversion of diagnostics belong to eager callers.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array


@dataclass(frozen=True)
class SupportPointHistory:
    """Objective and minimum pairwise distance, including initialization."""

    objective: tuple[float, ...]
    minimum_pairwise_distance: tuple[float, ...]


@dataclass(frozen=True)
class SupportPointDiagnostics:
    """Eager objectives, native solver outcome, displacement and spacing."""

    objective: float
    initial_objective: float
    converged: bool
    status: str
    iterations: int
    final_maximum_displacement: float
    minimum_pairwise_distance: float
    mean_nearest_spacing: float
    history: SupportPointHistory | None = None


def _minimum_pairwise_distance(points: Array) -> Array:
    """Minimum over distinct pairs; undefined for fewer than two points."""
    n = len(points)
    if n < 2:
        return jnp.full((), jnp.nan, dtype=points.dtype)
    distances = jnp.linalg.norm(points[:, None] - points[None], axis=-1)
    return jnp.min(jnp.where(jnp.eye(n, dtype=bool), jnp.inf, distances))


def _blocks(reference: Array, chunk_size: int) -> tuple[Array, Array]:
    """Pad a reference into fixed-size blocks and an explicit validity mask."""
    n, d = reference.shape
    size = (n + chunk_size - 1) // chunk_size * chunk_size
    padded = jnp.pad(reference, ((0, size - n), (0, 0)))
    blocks = padded.reshape(size // chunk_size, chunk_size, d)
    mask = (jnp.arange(size) < n).reshape(size // chunk_size, chunk_size)
    return blocks, mask


def _objective(points: Array, blocks: Array, mask: Array, n: int) -> Array:
    """Compute the exact unsmoothed objective with bounded reference workspace."""

    def accumulate(total: Array, block: tuple[Array, Array]) -> tuple[Array, None]:
        x, valid = block
        distance = jnp.linalg.norm(points[:, None] - x[None], axis=-1)
        return total + jnp.where(valid[None], distance, 0).sum(), None

    total, _ = jax.lax.scan(accumulate, jnp.zeros((), points.dtype), (blocks, mask))
    pair = jnp.linalg.norm(points[:, None] - points[None], axis=-1)
    return 2 * total / (len(points) * n) - pair.mean()


def _proposal(
    points: Array, fixed: Array, blocks: Array, mask: Array, n: int, epsilon: float
) -> Array:
    """Evaluate Eq. 22 with inverse-distance denominator protection."""

    def accumulate(
        carry: tuple[Array, Array], block: tuple[Array, Array]
    ) -> tuple[tuple[Array, Array], None]:
        numerator, denominator = carry
        x, valid = block
        distance = jnp.linalg.norm(points[:, None] - x[None], axis=-1)
        weights = jnp.where(valid[None], 1 / jnp.maximum(distance, epsilon), 0)
        return (numerator + weights @ x, denominator + weights.sum(1)), None

    (numerator, denominator), _ = jax.lax.scan(
        accumulate,
        (jnp.zeros_like(points), jnp.zeros(len(points), dtype=points.dtype)),
        (blocks, mask),
    )
    combined = jnp.concatenate((points, fixed), axis=0)
    delta = points[:, None] - combined[None]
    distance = jnp.linalg.norm(delta, axis=-1)
    repulsion = (delta / jnp.maximum(distance[..., None], epsilon)).sum(1)
    return (numerator + n / len(combined) * repulsion) / denominator[:, None]


@partial(jax.jit, static_argnames=("chunk_size",))
def energy_objective_jax(
    points: Array, reference: Array, chunk_size: int = 1024
) -> Array:
    """Return energy distance with the reference-only constant omitted."""
    dtype = jnp.result_type(points, reference, 0.0)
    points, reference = (
        jnp.asarray(points, dtype=dtype),
        jnp.asarray(reference, dtype=dtype),
    )
    blocks, mask = _blocks(reference, chunk_size)
    return _objective(points, blocks, mask, len(reference))


def energy_objective(
    points: np.ndarray, reference: np.ndarray, chunk_size: int = 1024
) -> float:
    """Return an eager Python scalar; use energy_objective_jax while tracing."""
    return float(energy_objective_jax(points, reference, chunk_size=chunk_size))


def _maximum_point_displacement(delta: Array) -> Array:
    """Measure absolute fixed-point error using per-point Euclidean distance."""
    # FixedPointIteration initializes its scalar error at the process default
    # precision. Match that state dtype even for explicitly float32 positions.
    return jnp.max(jnp.linalg.norm(delta, axis=1)).astype(jnp.asarray(0.0).dtype)


@partial(jax.jit, static_argnames=("max_iterations", "chunk_size", "trace_callback"))
def optimize_support_points_jax(
    reference: Array,
    initial: Array,
    *,
    fixed_points: Array | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    chunk_size: int = 1024,
    distance_epsilon: float = 1e-8,
    trace_callback: Callable[[Array, Array], None] | None = None,
) -> optx.Solution:
    """Solve the sp.ccp fixed point with Optimistix's built-in iteration.

    Args:
        reference: Validated finite reference cloud of shape (N, d).
        initial: Finite initial design of shape (m, d).
        fixed_points: Existing locations of shape (k, d); these never move.
        max_iterations: Positive static iteration budget.
        tolerance: Absolute maximum per-point displacement tolerance.
        chunk_size: Positive static reference block size.
        distance_epsilon: Positive inverse-distance safeguard, not separation.
        trace_callback: Optional ordered host callback recording objective and
            minimum pairwise distance after each step.

    Returns:
        Native Optimistix solution: value contains the final design, aux the
        last full displacement, and result/stats the termination diagnostics.
        Residual physical constraints must be checked after mapping back.
    """
    dtype = jnp.result_type(reference, initial, 0.0)
    reference, initial = (
        jnp.asarray(reference, dtype=dtype),
        jnp.asarray(initial, dtype=dtype),
    )
    fixed = (
        jnp.empty((0, initial.shape[1]), dtype=dtype)
        if fixed_points is None
        else jnp.asarray(fixed_points, dtype=dtype)
    )
    blocks, mask = _blocks(reference, chunk_size)

    def fixed_point(points: Array, args: None) -> tuple[Array, Array]:
        del args
        proposal = _proposal(
            points, fixed, blocks, mask, len(reference), distance_epsilon
        )
        if trace_callback is not None:
            combined = jnp.concatenate((proposal, fixed), axis=0)
            jax.debug.callback(
                trace_callback,
                _objective(combined, blocks, mask, len(reference)),
                _minimum_pairwise_distance(combined),
                ordered=True,
            )
        return proposal, _maximum_point_displacement(proposal - points).astype(
            points.dtype
        )

    solver = optx.FixedPointIteration(
        rtol=0.0,
        atol=tolerance,
        norm=_maximum_point_displacement,
    )
    return optx.fixed_point(
        fixed_point,
        solver,
        initial,
        has_aux=True,
        max_steps=max_iterations,
        throw=False,
    )


def _validate_inputs(
    reference: np.ndarray,
    initial: np.ndarray,
    max_iterations: int,
    chunk_size: int,
    tolerance: float,
    epsilon: float,
) -> None:
    """Check eager numerical inputs before entering the compiled kernel."""
    if (
        reference.ndim != 2
        or initial.ndim != 2
        or reference.shape[1] != initial.shape[1]
        or len(reference) == 0
        or len(initial) == 0
        or not np.isfinite(reference).all()
        or not np.isfinite(initial).all()
    ):
        raise ValueError("reference and initial must be finite nonempty (n, d) arrays")
    for name, value in (
        ("max_iterations", max_iterations),
        ("chunk_size", chunk_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not (
        np.isfinite(tolerance)
        and tolerance > 0
        and np.isfinite(epsilon)
        and epsilon > 0
    ):
        raise ValueError("tolerance and distance_epsilon must be finite and positive")


def optimize_support_points(
    reference: np.ndarray,
    initial: np.ndarray,
    *,
    fixed_points: np.ndarray | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    chunk_size: int = 1024,
    distance_epsilon: float = 1e-8,
    record_history: bool = False,
) -> tuple[np.ndarray, SupportPointDiagnostics]:
    """Validate arrays and run unconstrained Optimistix fixed-point iteration.

    Convergence uses absolute maximum per-point displacement strictly below
    tolerance (Optimistix's convention), with zero relative tolerance. Objective
    decrease is not enforced; numerical coincidences use distance_epsilon.
    """
    x, z = np.asarray(reference, dtype=float), np.asarray(initial, dtype=float)
    _validate_inputs(x, z, max_iterations, chunk_size, tolerance, distance_epsilon)
    fixed = (
        np.empty((0, z.shape[1]))
        if fixed_points is None
        else np.asarray(fixed_points, dtype=float)
    )
    if fixed.ndim != 2 or fixed.shape[1] != z.shape[1] or not np.isfinite(fixed).all():
        raise ValueError("fixed_points must be a finite (k, d) array matching initial")
    events: list[tuple[float, float]] = []

    def collect(objective: Array, spacing: Array) -> None:
        events.append((float(objective), float(spacing)))

    result = optimize_support_points_jax(
        jnp.asarray(x),
        jnp.asarray(z),
        fixed_points=jnp.asarray(fixed),
        max_iterations=max_iterations,
        tolerance=tolerance,
        chunk_size=chunk_size,
        distance_epsilon=distance_epsilon,
        trace_callback=collect if record_history else None,
    )
    points = np.asarray(result.value)
    if record_history:
        jax.effects_barrier()
        if len(events) != int(result.stats["num_steps"]):
            raise RuntimeError(
                "Optimization trace does not match the solver step count"
            )
    if not np.isfinite(points).all():
        status = "nonfinite"
    elif bool(result.result == optx.RESULTS.successful):
        status = "converged"
    elif bool(result.result == optx.RESULTS.nonlinear_max_steps_reached):
        status = "iteration_limit"
    else:
        status = "solver_failure"
    combined = np.concatenate((points, fixed), axis=0)
    initial_combined = np.concatenate((z, fixed), axis=0)
    pair = np.linalg.norm(combined[:, None] - combined[None], axis=-1)
    np.fill_diagonal(pair, np.inf)
    nearest = pair.min(1)
    initial_objective = energy_objective(initial_combined, x, chunk_size)
    history = (
        SupportPointHistory(
            (initial_objective, *(event[0] for event in events)),
            (
                float(_minimum_pairwise_distance(jnp.asarray(initial_combined))),
                *(event[1] for event in events),
            ),
        )
        if record_history
        else None
    )
    diagnostics = SupportPointDiagnostics(
        objective=energy_objective(combined, x, chunk_size),
        initial_objective=initial_objective,
        converged=status == "converged",
        status=status,
        iterations=int(result.stats["num_steps"]),
        final_maximum_displacement=float(result.aux),
        minimum_pairwise_distance=float(nearest.min()) if len(combined) > 1 else np.nan,
        mean_nearest_spacing=float(nearest.mean()) if len(combined) > 1 else np.nan,
        history=history,
    )
    return points, diagnostics
