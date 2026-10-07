"""`XZGridFromBoundaryOrbitSampler`: DYNAMITE-matching `boundin`/`boundmid`
search for the `(x, z)`-plane start space.

Implements van den Bosch et al. (2008, MNRAS 385, 647) sec. 4.3 -- the
scheme DYNAMITE, and its Schwarzschild orbit-superposition antecedents, use
to build a tube-orbit library with a completeness guarantee: every genuine
tube-orbit family at a given energy gets sampled, not just whichever radii
happen to fall in a fixed grid. Ported directly from DYNAMITE's own Fortran
reference (`orbitstart_f.f90`), matching its early-stopping search and
per-step bracket formulas.

At each energy shell, a trial orbit's *type* (box, short-axis tube,
long-axis tube, or chaotic) changes as its start radius moves along the
`(x, z)` plane's `theta` sweep from the x-axis to the z-axis. For a
*regular* shell, this traces out van den Bosch et al. (2008)'s four-region
picture cleanly: box orbits near the centre, then a short-axis-tube
annulus, then (at some shells) a long-axis-tube annulus further out. This
sampler locates the two edges of that short-axis-tube annulus directly --
`boundin` (`_inner_tube_boundary`): a continuation search across `theta`,
starting at the x-axis (where the box <-> short-axis-tube edge is found
cleanly) and working toward the z-axis; `boundmid` (`_outer_tube_boundary`):
the short- <-> long-axis-tube edge, or the equipotential itself where no
such edge exists at that `(energy, theta)` -- and samples the `(x, z)`
start space's radial sub-grid between them.

Not every shell has a clean four-region structure. A shell comes back
`irregular` when `_inner_tube_boundary`'s own continuation search hits its
bracket edge, or a long-axis tube is never found along the sweep
(DYNAMITE's own `notubes` condition, `orbitstart_f.f90:434`). Irregularity
also isn't purely local to one shell: DYNAMITE's own global step
(`orbitstart_f.f90:124-138`) marks every shell *inward* of the outermost
irregular one irregular too, without running the boundary search for them
at all. `generate_ics` mirrors this by walking shells outermost-to-innermost
and, from the first locally irregular shell on, delegating directly to
`xz_grid_from_origin._single_shell_ics` -- the same uniform-grid-to-the-
centre procedure that sampler uses for every shell, reused here rather than
reimplemented, instead of a separately-maintained inline fallback.

Produces only the `+v_y` population, same convention as
`xz_grid_from_origin.py` -- the counter-rotating mirror is a later,
post-integration concern.
"""

from __future__ import annotations

from typing import ClassVar

import diffrax
import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
import jax
import jax.numpy as jnp
import optimistix as optx
from unxt import Quantity

from tnt.orbit_library.base import AbstractOrbitSampler
from tnt.orbit_library.common import (
    _R_CEILING_FACTOR,
    _R_FLOOR_FACTOR,
    _circular_period,
    _equipotential_radius,
    _phase_space_derivs,
    _potential_at,
    _xz_direction,
    _xz_orbit_ic,
)
from tnt.orbit_library.xz_grid_from_origin import _single_shell_ics

# Matches `findtubeorbitwidth`'s own crossing-count cap (`intsteps = 400`,
# `orbitstart_f.f90:645`) exactly.
_TUBE_SEARCH_N_CROSSINGS = 100
# `findtubeorbitwidth` integrates for up to `500 * intsteps * tcirc` =
# 200,000 periods (`orbitstart_f.f90:645,668`) as a safety ceiling for
# pathological (non-converging) trial points -- reduced here (`5_000.0`,
# 20x the ~250-period convergence typically observed) to keep the typical,
# converging case's cost the one that matters, matching this module's own
# early-stopping design goal; see `_orbit_width_over_crossings`.
_TUBE_SEARCH_TIME_PERIODS = 5_000.0
# `find_type` (`orbitstart_f.f90:856`) integrates for `100 * tcirc`;
# `_CLASSIFY_N_SAMPLES` matches its own `intsteps = 5000` exactly.
_CLASSIFY_PERIODS = 100.0
_CLASSIFY_N_SAMPLES = 5000

# `diffrax`'s own per-step adaptive accuracy, and `optimistix.Bisection`'s
# for refining a crossing once found. DYNAMITE's own `findtubeorbitwidth`/
# `find_type` use a fixed `integrator_accuracy = 1d-4`
# (`orbitstart_f.f90:26`, `RTOL(:) = TOL`, `ATOL(:) = 1e-8_dp`) -- matched
# here directly.
_INTEGRATOR_RTOL = 1e-4
_INTEGRATOR_ATOL = 1e-8
# DYNAMITE's own crossing-refinement bisection (`SOLOUTOB`,
# `orbitstart_f.f90:797-812`) converges on `abs(residual) < R * 1e-4`
# capped at 40 iterations.
_CROSSING_RTOL = 1e-4
_CROSSING_ATOL = 1e-6
_CROSSING_MAX_STEPS = 40
# `_minimize_tube_width`'s own fixed golden-section iteration count.
_TUBE_SEARCH_N_ITER = 30
_GOLDEN_RATIO = (5.0**0.5 - 1.0) / 2.0

# Orbit family codes, matching `find_type`'s classification
# (`orbitstart_f.f90:826-927`), zero-indexed: which angular-momentum
# component keeps one sign throughout a trial integration (an `X`, `Y`, or
# `Z` tube), all three change sign (a box orbit), or none of the above
# (chaotic/unclassified).
_ORBIT_TYPE_X_TUBE = 0
_ORBIT_TYPE_Y_TUBE = 1
_ORBIT_TYPE_Z_TUBE = 2
_ORBIT_TYPE_BOX = 3
_ORBIT_TYPE_CHAOTIC = 4

# `_inner_tube_boundary` and `_outer_tube_boundary` each bracket their own
# continuation search around the *previous* theta step's solution -- ported
# as two distinct, dedicated bracket constructions (matching
# `find_innerboundary`, `orbitstart_f.f90:328-379`, and `find_outerboundary`,
# `:471-518`, respectively) rather than one shared rule.
_GUESS_MARGIN_LOWER_FRACTION = 0.11
_GUESS_MARGIN_UPPER_FRACTION = 0.89
_INNER_SECOND_STEP_GUESS_LOWER_FRACTION = 0.02
_INNER_SECOND_STEP_GUESS_UPPER_FRACTION = 0.99
_INNER_SECOND_STEP_HALF_WIDTH_FRACTION = 0.10
_INNER_SECOND_STEP_FLOOR_FRACTION = 0.01
_INNER_STEP_GUESS_LOWER_FRACTION = 0.11
_INNER_STEP_GUESS_UPPER_FRACTION = 0.99
_INNER_STEP_HALF_WIDTH_FRACTION = 0.08
_INNER_STEP_FLOOR_FRACTION = 0.10
# `_outer_tube_boundary`'s equipotential-avoidance margin
# (`orbitstart_f.f90:482`, `rbu = boundout(j,k) - 1.0e-6`) -- DYNAMITE's own
# constant is an *absolute* offset in its km-scale internal units, meaningless
# ported literally into a potential whose length unit can be `O(1)`; applied
# here as a small relative fraction of `r_outer` instead, preserving the
# intent (never land exactly on the equipotential) rather than the literal
# constant.
_OUTER_BOUNDARY_EQUIPOTENTIAL_MARGIN_FRACTION = 1e-6
# A continuation step's solution counts as having hit its own bracket edge
# (and so marks its energy shell `irregular`) when it lands within this
# fraction of itself from either bracket bound -- mirrors
# `find_innerboundary`'s `1.0e-2` tolerance.
_BOUNDARY_EDGE_TOLERANCE = 1e-2


def _shell_integration_budgets(
    potential: gp.AbstractPotential, r_ref: jnp.ndarray, t0: Quantity
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """`(t_ceiling, classify_integration_time)` from one reference radius's
    circular period -- computed once per boundary search (not once per
    trial point) and reused for every trial radius that search evaluates.

    Both budgets share the same `_circular_period` evaluation (one
    `jax.grad` through the potential, not two) since they only differ by a
    constant multiplier.
    """
    period = _circular_period(potential, r_ref, t0)
    return _TUBE_SEARCH_TIME_PERIODS * period, _CLASSIFY_PERIODS * period


_TUBE_SEARCH_SOLVER_STEP = diffrax.Dopri8()
_TUBE_SEARCH_STEPSIZE_CONTROLLER = diffrax.PIDController(
    rtol=_INTEGRATOR_RTOL, atol=_INTEGRATOR_ATOL
)
# `flip=False` here too, for the same vmap reason as `common._EQUIPOTENTIAL_
# SOLVER` -- but the true sign order isn't known analytically (a crossing
# can approach the tracked plane from either side), so `_orbit_width_over_
# crossings` sorts `lower`/`upper` itself by the sign of the tracked
# coordinate at the accepted step's start before calling this.
_CROSSING_SOLVER = optx.Bisection(rtol=_CROSSING_RTOL, atol=_CROSSING_ATOL, flip=False)


def _orbit_width_over_crossings(
    potential: gp.AbstractPotential,
    ic: jnp.ndarray,
    plane: str,
    t_ceiling: jnp.ndarray,
    t0: Quantity,
) -> jnp.ndarray:
    """`max - min` of the plane-appropriate radius over every crossing of
    `plane` found in a trial orbit -- van den Bosch et al. (2008)'s
    `findtubeorbitwidth` (`orbitstart_f.f90:640-`).

    `plane="y"` tracks `y = 0` crossings and measures `sqrt(x**2 + z**2)`
    there -- used by `_inner_tube_boundary`'s `boundin` search, across the
    whole `theta` sweep: a family-agnostic thin-orbit test. `plane="x"`
    tracks `x = 0` crossings and measures `sqrt(y**2 + z**2)` -- used by
    `_outer_tube_boundary`'s `boundmid` search, specifically tracking a
    long-axis tube's own thin locus.

    A `jax.lax.while_loop` driving `diffrax`'s own low-level, single-step
    API directly (`solver.step` + `stepsize_controller.adapt_step_size`)
    rather than a `diffrax.diffeqsolve` call per iteration -- this mirrors
    `findtubeorbitwidth`/`SOLOUTOB`'s own structure directly: `SOLOUTOB` is
    itself a callback invoked once per accepted `DOP853` step
    (`orbitstart_f.f90:764-823`), exactly what the loop body below does too,
    stopping the instant `_TUBE_SEARCH_N_CROSSINGS` crossings are found or
    `t_ceiling` is reached. Every crossing found counts equally toward a
    running `min`/`max`, not just the first. Returns `inf` if no crossing at
    all is found within the time budget, so an enclosing minimization skips
    over it.

    The very first accepted step is never counted as a crossing: `ic` sits
    exactly on the tracked plane by construction (`_xz_orbit_ic` launches
    from it), so the tracked coordinate is exactly `0.0` at `t = 0` -- a
    trivial "crossing" from the launch itself, not a genuine return.
    """
    plane_idx = 1 if plane == "y" else 0
    other_idx = 0 if plane == "y" else 1

    derivs = _phase_space_derivs(potential, t0)
    term = diffrax.ODETerm(derivs)
    solver = _TUBE_SEARCH_SOLVER_STEP
    controller = _TUBE_SEARCH_STEPSIZE_CONTROLLER
    error_order = solver.error_order(term)

    t_start = jnp.asarray(0.0)
    tnext0, controller_state0 = controller.init(
        term, t_start, t_ceiling, ic, None, None, solver.func, error_order
    )
    solver_state0 = solver.init(term, t_start, tnext0, ic, None)

    def cond(state: tuple) -> jnp.ndarray:
        t = state[0]
        n_crossings = state[6]
        return (n_crossings < _TUBE_SEARCH_N_CROSSINGS) & (t < t_ceiling)

    def body(state: tuple) -> tuple:
        (
            t,
            y,
            solver_state,
            controller_state,
            made_jump,
            tnext,
            n_crossings,
            radius_min,
            radius_max,
            first,
        ) = state

        y1, y_error, dense_info, new_solver_state, _solver_result = solver.step(
            term, t, tnext, y, None, solver_state, made_jump
        )
        y_error = jax.tree_util.tree_map(
            lambda leaf: jnp.where(jnp.isnan(leaf), jnp.inf, leaf), y_error
        )
        (
            keep_step,
            tprev,
            tnext_new,
            made_jump_new,
            controller_state_new,
            _ctrl_result,
        ) = controller.adapt_step_size(
            t, tnext, y, y1, None, y_error, error_order, controller_state
        )
        tprev = jnp.minimum(tprev, t_ceiling)

        keep = lambda accepted, rejected: jnp.where(keep_step, accepted, rejected)
        y_kept = jax.tree_util.tree_map(keep, y1, y)
        solver_state_kept = jax.tree_util.tree_map(keep, new_solver_state, solver_state)

        c0, c1 = y[plane_idx], y_kept[plane_idx]
        crossed = keep_step & (jnp.sign(c0) != jnp.sign(c1)) & (~first)

        def refine(_: None) -> jnp.ndarray:
            interpolation = solver.interpolation_cls(t0=t, t1=tprev, **dense_info)
            lower = jnp.where(c0 < 0.0, t, tprev)
            upper = jnp.where(c0 < 0.0, tprev, t)

            def crossing_coord_at(tt: jnp.ndarray, _args: None) -> jnp.ndarray:
                return interpolation.evaluate(tt)[plane_idx]

            solution = optx.root_find(
                crossing_coord_at,
                _CROSSING_SOLVER,
                y0=0.5 * (lower + upper),
                options={"lower": lower, "upper": upper},
                max_steps=_CROSSING_MAX_STEPS,
                throw=False,
            )
            y_cross = interpolation.evaluate(solution.value)
            a, b = y_cross[other_idx], y_cross[2]
            return jnp.sqrt(a**2 + b**2)

        radius = jax.lax.cond(crossed, refine, lambda _: 0.0, operand=None)
        new_min = jnp.where(crossed, jnp.minimum(radius_min, radius), radius_min)
        new_max = jnp.where(crossed, jnp.maximum(radius_max, radius), radius_max)
        new_n = n_crossings + jnp.where(crossed, 1, 0)
        new_first = jnp.where(keep_step, jnp.asarray(False), first)

        return (
            tprev,
            y_kept,
            solver_state_kept,
            controller_state_new,
            made_jump_new,
            tnext_new,
            new_n,
            new_min,
            new_max,
            new_first,
        )

    init = (
        t_start,
        ic,
        solver_state0,
        controller_state0,
        jnp.asarray(False),
        tnext0,
        jnp.asarray(0),
        jnp.asarray(jnp.inf),
        jnp.asarray(-jnp.inf),
        jnp.asarray(True),
    )
    final_state = jax.lax.while_loop(cond, body, init)
    n_final, radius_min, radius_max = final_state[6], final_state[7], final_state[8]
    width = radius_max - radius_min
    return jnp.where(n_final > 0, width, jnp.inf)


def _classify_orbit_type(
    potential: gp.AbstractPotential,
    ic: jnp.ndarray,
    integration_time: jnp.ndarray,
    t0: Quantity,
) -> jnp.ndarray:
    """Which orbit family `ic` belongs to -- a direct, vectorized port of
    `find_type` (`orbitstart_f.f90:826-927`).

    Integrates at `_CLASSIFY_N_SAMPLES` fixed time steps (no crossing
    detection needed here -- DYNAMITE always integrates this for a fixed
    duration regardless of outcome) and checks, for each angular-momentum
    component, whether its sign is conserved throughout (`max * min > 0`):
    exactly one conserved-sign component means a tube circulating around
    that axis (`_ORBIT_TYPE_X_TUBE`/`_Y_TUBE`/`_Z_TUBE`); all three
    changing sign means a box orbit (`_ORBIT_TYPE_BOX`); anything else is
    `_ORBIT_TYPE_CHAOTIC`.
    """
    time_unit = potential.units["time"]
    length_unit = potential.units["length"]
    velocity_unit = length_unit / time_unit

    w0 = gc.PhaseSpaceCoordinate(
        q=Quantity(ic[:3], length_unit), p=Quantity(ic[3:], velocity_unit), t=t0
    )
    t0_val = t0.ustrip(time_unit)
    t_vals = t0_val + integration_time * jnp.linspace(0.0, 1.0, _CLASSIFY_N_SAMPLES)
    orbit = gd.compute_orbit(potential, w0, Quantity(t_vals, time_unit), dense=False)

    x, y, z = (
        orbit.q.x.ustrip(length_unit),
        orbit.q.y.ustrip(length_unit),
        orbit.q.z.ustrip(length_unit),
    )
    vx, vy, vz = (
        orbit.p.x.ustrip(velocity_unit),
        orbit.p.y.ustrip(velocity_unit),
        orbit.p.z.ustrip(velocity_unit),
    )
    lx, ly, lz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    lxc, lyc, lzc = (
        jnp.max(lx) * jnp.min(lx),
        jnp.max(ly) * jnp.min(ly),
        jnp.max(lz) * jnp.min(lz),
    )

    is_x_tube = (lxc > 0.0) & (lyc < 0.0) & (lzc < 0.0)
    is_y_tube = (lxc < 0.0) & (lyc > 0.0) & (lzc < 0.0)
    is_z_tube = (lxc < 0.0) & (lyc < 0.0) & (lzc > 0.0)
    is_box = (lxc < 0.0) & (lyc < 0.0) & (lzc < 0.0)

    return jnp.select(
        [is_x_tube, is_y_tube, is_z_tube, is_box],
        [_ORBIT_TYPE_X_TUBE, _ORBIT_TYPE_Y_TUBE, _ORBIT_TYPE_Z_TUBE, _ORBIT_TYPE_BOX],
        default=_ORBIT_TYPE_CHAOTIC,
    )


def _classify_at(
    potential: gp.AbstractPotential,
    energy: jnp.ndarray,
    theta: jnp.ndarray,
    r: jnp.ndarray,
    integration_time: jnp.ndarray,
    t0: Quantity,
) -> jnp.ndarray:
    """`_classify_orbit_type` for the `(x, z)`-start-space orbit at
    `(energy, theta, r)`.
    """
    ic = _xz_orbit_ic(potential, energy, theta, r, t0)
    return _classify_orbit_type(potential, ic, integration_time, t0)


def _minimize_tube_width(
    potential: gp.AbstractPotential,
    energy: jnp.ndarray,
    theta: jnp.ndarray,
    plane: str,
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    t_ceiling: jnp.ndarray,
    t0: Quantity,
) -> jnp.ndarray:
    """`argmin_r _orbit_width_over_crossings` at fixed `(energy, theta)`,
    bracketed in `[lo, hi]` -- the search step shared by
    `_inner_tube_boundary` and `_outer_tube_boundary`.

    A hand-rolled, fixed-`_TUBE_SEARCH_N_ITER`-iteration golden-section
    search via `jax.lax.scan`, not `optimistix.minimise`/`GoldenSearch`:
    composing `optimistix`'s own generic solver loop with an objective that
    itself contains a `jax.lax.while_loop` compiles and runs far slower.
    Each step picks one of its two trial points via `jax.lax.cond` -- not
    `jnp.where`, which would evaluate both (each a full, expensive
    trial-orbit integration) every iteration regardless of which one is
    kept.

    The width objective isn't guaranteed unimodal in `r` -- the result is
    clipped back into `[lo, hi]` regardless, so a poorly-behaved objective
    degrades to "a valid-but-possibly-suboptimal value" rather than
    poisoning the continuation searches this feeds.
    """

    def objective(r: jnp.ndarray) -> jnp.ndarray:
        ic = _xz_orbit_ic(potential, energy, theta, r, t0)
        return _orbit_width_over_crossings(potential, ic, plane, t_ceiling, t0)

    x1 = hi - _GOLDEN_RATIO * (hi - lo)
    x2 = lo + _GOLDEN_RATIO * (hi - lo)
    f1, f2 = objective(x1), objective(x2)

    def step(carry: tuple, _xs: None) -> tuple[tuple, None]:
        lo, hi, x1, x2, f1, f2 = carry
        go_right = f1 > f2

        def take_right(_: None) -> tuple:
            new_lo, new_hi = x1, hi
            new_x1 = x2
            new_x2 = new_lo + _GOLDEN_RATIO * (new_hi - new_lo)
            return new_lo, new_hi, new_x1, new_x2, f2, objective(new_x2)

        def take_left(_: None) -> tuple:
            new_lo, new_hi = lo, x2
            new_x2 = x1
            new_x1 = new_hi - _GOLDEN_RATIO * (new_hi - new_lo)
            return new_lo, new_hi, new_x1, new_x2, objective(new_x1), f1

        new_carry = jax.lax.cond(go_right, take_right, take_left, operand=None)
        return new_carry, None

    (_, _, x1_f, x2_f, f1_f, f2_f), _ = jax.lax.scan(
        step, (lo, hi, x1, x2, f1, f2), None, length=_TUBE_SEARCH_N_ITER
    )
    best = jnp.where(f1_f < f2_f, x1_f, x2_f)
    return jnp.clip(best, lo, hi)


def _inner_tube_boundary(
    potential: gp.AbstractPotential,
    energy: jnp.ndarray,
    theta_grid: jnp.ndarray,
    r_floor: jnp.ndarray,
    r_outer_grid: jnp.ndarray,
    t_ceiling: jnp.ndarray,
    classify_integration_time: jnp.ndarray,
    t0: Quantity,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """`boundin` at one energy shell, for every `theta` in `theta_grid` --
    van den Bosch et al. (2008)'s `find_innerboundary`
    (`orbitstart_f.f90:306-384`).

    A continuation search across `theta`, run backwards from the x-axis
    (`theta_grid[-1]`, where the box <-> short-axis-tube edge is found
    cleanly) toward the z-axis. `theta_grid` must be ascending; the search
    runs in reverse internally, and the returned grid is re-ordered back to
    match `theta_grid`.

    Ported as three distinct stages, matching `find_innerboundary` exactly:
    the x-axis step itself (a fixed bracket, `_GUESS_MARGIN_*`), the second
    theta step in (its own guess-clamp/half-width pair,
    `_INNER_SECOND_STEP_*`), and every remaining step (a third pair,
    `_INNER_STEP_*`, with the half-width scaled by the x-axis equipotential
    radius rather than the current step's own) -- the first two unrolled
    explicitly, the rest via `jax.lax.scan`.

    Returns:
        `(boundin_grid, irregular)`: `boundin_grid` has the same shape as
        `theta_grid`. `irregular` is `True` when any step's solution landed
        on its own bracket edge, or when a long-axis tube was never
        classified anywhere along the sweep.
    """
    theta_desc = theta_grid[::-1]
    r_outer_desc = r_outer_grid[::-1]
    r_outer_x_axis = r_outer_desc[0]

    def on_edge(value: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
        return (jnp.abs(value - lo) < _BOUNDARY_EDGE_TOLERANCE * value) | (
            jnp.abs(value - hi) < _BOUNDARY_EDGE_TOLERANCE * value
        )

    def is_x_tube(theta: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
        orbit_type = _classify_at(
            potential, energy, theta, r, classify_integration_time, t0
        )
        return orbit_type == _ORBIT_TYPE_X_TUBE

    # Stage 1 -- the x-axis step (`orbitstart_f.f90:328-338`).
    lo0 = jnp.maximum(r_floor, _GUESS_MARGIN_LOWER_FRACTION * r_outer_x_axis)
    hi0 = _GUESS_MARGIN_UPPER_FRACTION * r_outer_x_axis
    boundin0 = _minimize_tube_width(
        potential, energy, theta_desc[0], "y", lo0, hi0, t_ceiling, t0
    )
    edge0 = on_edge(boundin0, lo0, hi0)
    found_x0 = is_x_tube(theta_desc[0], boundin0)

    # Stage 2 -- the second theta step in (`orbitstart_f.f90:344-349`).
    r_outer_1 = r_outer_desc[1]
    guess1 = jnp.clip(
        boundin0,
        _INNER_SECOND_STEP_GUESS_LOWER_FRACTION * r_outer_1,
        _INNER_SECOND_STEP_GUESS_UPPER_FRACTION * r_outer_1,
    )
    lo1 = jnp.maximum(
        guess1 - _INNER_SECOND_STEP_HALF_WIDTH_FRACTION * r_outer_1,
        _INNER_SECOND_STEP_FLOOR_FRACTION * r_outer_1,
    )
    hi1 = jnp.minimum(
        guess1 + _INNER_SECOND_STEP_HALF_WIDTH_FRACTION * r_outer_1, r_outer_1
    )
    boundin1 = _minimize_tube_width(
        potential, energy, theta_desc[1], "y", lo1, hi1, t_ceiling, t0
    )
    edge1 = on_edge(boundin1, lo1, hi1)
    found_x1 = is_x_tube(theta_desc[1], boundin1)

    # Stage 3 -- every remaining theta step (`orbitstart_f.f90:354-379`).
    def step(carry: tuple, xs: tuple) -> tuple[tuple, jnp.ndarray]:
        prev_boundin, found_x_acc, edge_acc = carry
        theta_i, r_outer_i = xs
        guess = jnp.clip(
            prev_boundin,
            _INNER_STEP_GUESS_LOWER_FRACTION * r_outer_i,
            _INNER_STEP_GUESS_UPPER_FRACTION * r_outer_i,
        )
        lo = jnp.maximum(
            guess - _INNER_STEP_HALF_WIDTH_FRACTION * r_outer_x_axis,
            _INNER_STEP_FLOOR_FRACTION * r_outer_i,
        )
        hi = jnp.minimum(
            guess + _INNER_STEP_HALF_WIDTH_FRACTION * r_outer_x_axis, r_outer_i
        )
        boundin_i = _minimize_tube_width(
            potential, energy, theta_i, "y", lo, hi, t_ceiling, t0
        )
        edge_i = on_edge(boundin_i, lo, hi)
        found_x_i = is_x_tube(theta_i, boundin_i)
        new_carry = (boundin_i, found_x_acc | found_x_i, edge_acc | edge_i)
        return new_carry, boundin_i

    init_carry = (boundin1, found_x0 | found_x1, edge0 | edge1)
    (_, found_x_final, edge_final), boundin_rest_desc = jax.lax.scan(
        step, init_carry, (theta_desc[2:], r_outer_desc[2:])
    )

    boundin_grid = jnp.concatenate(
        [boundin0[None], boundin1[None], boundin_rest_desc]
    )[::-1]
    return boundin_grid, edge_final | (~found_x_final)


def _outer_tube_boundary(
    potential: gp.AbstractPotential,
    energy: jnp.ndarray,
    theta_grid: jnp.ndarray,
    boundin_grid: jnp.ndarray,
    r_outer_grid: jnp.ndarray,
    r_floor: jnp.ndarray,
    t_ceiling: jnp.ndarray,
    classify_integration_time: jnp.ndarray,
    nI2: int,
    t0: Quantity,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """`boundmid` at one energy shell, for every `theta` in `theta_grid` --
    van den Bosch et al. (2008)'s `find_outerboundary`
    (`orbitstart_f.f90:386-520`).

    Three outcomes from a radial scan at `theta_grid[-1]` (the x-axis),
    walked sequentially outward from `boundin` toward the equipotential and
    classifying at each step, exiting the instant an `x_tube` or `box` is
    found (at `k >= 2`):

    * the scan runs the whole way to the last step *and that last step is
      specifically classified `z_tube`* -- `boundmid` stays `= r_outer` for
      every `theta` (the `notubes` case, `orbitstart_f.f90:434-439` -- this
      also forces this energy shell *not* irregular, overriding
      `_inner_tube_boundary`'s own flag: the caller combines the two,
      `irregular_inner & ~notubes`);
    * it finds a long-axis tube directly -- every other `theta` gets a
      searched `boundmid`;
    * it finds a box orbit (or ends on anything other than `z_tube`) -- a
      `theta` sweep toward the z-axis, at that same relative radius,
      locates the index where long-axis tubes first appear; only `theta`
      below that index gets a searched `boundmid`.

    Wherever a search is called for, it's a second continuation search
    (tracking `x = 0` crossings -- `plane="x"`), with its own bracket
    construction ported from `find_outerboundary` (`orbitstart_f.f90:471-
    518`): the inner bound floor is the box<->tube conversion radius or the
    largest `boundin` value anywhere on the grid, the outer bound is
    additionally capped by the *previous* theta step's own found
    `boundmid`, the guess rescales the previous step's `boundmid` by the
    ratio of equipotential radii, and the inner bound is tapered toward
    that guess by a fraction depending on `rel_rbi` and `nI2`.

    Returns:
        `(boundmid_grid, notubes)`: `boundmid_grid` has the same shape as
        `theta_grid`; `notubes` is the scalar bool described above.
    """
    n_theta = theta_grid.shape[0]
    theta_top, boundin_top, r_outer_top = (
        theta_grid[-1],
        boundin_grid[-1],
        r_outer_grid[-1],
    )

    # `orbitstart_f.f90:419-431`: `3 * nI2` steps, walked sequentially and
    # exited the instant an `x_tube`/`box` is found at `k >= 2` -- the
    # carry freezes (`broken`) from that point on. `rel_rbi` only updates
    # on `z_tube` points seen *before* any break.
    n_scan = 3 * nI2
    k = jnp.arange(1, n_scan + 1)
    r_scan = boundin_top + (r_outer_top - boundin_top) * k / (n_scan + 1)

    def scan_step(carry: tuple, xs: tuple) -> tuple[tuple, None]:
        broken, last_type, last_r, rel_rbi_acc = carry
        k_i, r_i = xs
        type_i = _classify_at(
            potential, energy, theta_top, r_i, classify_integration_time, t0
        )
        is_break = (type_i == _ORBIT_TYPE_X_TUBE) | (type_i == _ORBIT_TYPE_BOX)
        break_now = is_break & (k_i >= 2) & (~broken)
        new_rel_rbi = jnp.where(
            (type_i == _ORBIT_TYPE_Z_TUBE) & (~broken), r_i / r_outer_top, rel_rbi_acc
        )
        new_last_type = jnp.where(broken, last_type, type_i)
        new_last_r = jnp.where(broken, last_r, r_i)
        return (broken | break_now, new_last_type, new_last_r, new_rel_rbi), None

    init_scan = (
        jnp.asarray(False),
        jnp.asarray(_ORBIT_TYPE_CHAOTIC),
        boundin_top,
        boundin_top / r_outer_top,
    )
    (broken, last_type, last_r, rel_rbi), _ = jax.lax.scan(
        scan_step, init_scan, (k, r_scan)
    )

    # `orbitstart_f.f90:434`: `notubes` fires only when the scan ran to
    # completion *and* the last point it saw is specifically `z_tube`.
    notubes = (~broken) & (last_type == _ORBIT_TYPE_Z_TUBE)
    found_x_directly = broken & (last_type == _ORBIT_TYPE_X_TUBE)
    bp = last_r / r_outer_top

    idx_desc = jnp.arange(n_theta - 2, -1, -1)
    types_theta_scan = jax.vmap(
        lambda th, r: _classify_at(
            potential, energy, th, r, classify_integration_time, t0
        )
    )(theta_grid[idx_desc], r_outer_grid[idx_desc] * bp)
    is_x_theta = types_theta_scan == _ORBIT_TYPE_X_TUBE
    found_onset = jnp.any(is_x_theta)
    onset_idx_from_scan = jnp.where(
        found_onset, idx_desc[jnp.argmax(is_x_theta)], n_theta - 1
    )

    effective_onset_idx = jnp.where(
        notubes, n_theta, jnp.where(found_x_directly, n_theta - 1, onset_idx_from_scan)
    )

    boundin_max = jnp.max(boundin_grid)
    taper_fraction = (1.0 - rel_rbi) / (3.0 * nI2)

    theta_desc2 = theta_grid[:-1][::-1]
    r_outer_desc2 = r_outer_grid[:-1][::-1]
    r_outer_at_onset = r_outer_grid[jnp.clip(effective_onset_idx, 0, n_theta - 1)]

    def step(carry: tuple, xs: tuple) -> tuple[tuple, jnp.ndarray]:
        prev_boundmid, prev_r_outer = carry
        theta_i, r_outer_i, idx_i = xs

        # `orbitstart_f.f90:471`: the seed for the first real search step --
        # the onset index itself is never really searched (its own result
        # is masked out below), so the step immediately below it needs its
        # `rbu`/`guess` built around the onset's own `r_outer`.
        at_seed = idx_i == (effective_onset_idx - 1)
        prev_boundmid = jnp.where(at_seed, r_outer_at_onset, prev_boundmid)
        prev_r_outer = jnp.where(at_seed, r_outer_at_onset, prev_r_outer)

        rbi_floor = jnp.maximum(rel_rbi * r_outer_i, boundin_max)
        equipotential_margin = _OUTER_BOUNDARY_EQUIPOTENTIAL_MARGIN_FRACTION * r_outer_i
        rbu = jnp.minimum(prev_boundmid, r_outer_i - equipotential_margin)
        guess = jnp.clip(prev_boundmid / prev_r_outer * r_outer_i, rbi_floor, rbu)
        rbi = jnp.maximum(guess * (1.0 - taper_fraction), rbi_floor)

        lo = jnp.where(rbi >= rbu, r_floor, rbi)
        hi = jnp.where(rbi >= rbu, r_outer_i, rbu)

        boundmid_i = _minimize_tube_width(
            potential, energy, theta_i, "x", lo, hi, t_ceiling, t0
        )
        searched = idx_i < effective_onset_idx
        result_i = jnp.where(searched, boundmid_i, r_outer_i)
        new_carry = (
            jnp.where(searched, boundmid_i, prev_boundmid),
            jnp.where(searched, r_outer_i, prev_r_outer),
        )
        return new_carry, result_i

    init_carry = (r_outer_top, r_outer_top)
    _, boundmid_rest_desc = jax.lax.scan(
        step, init_carry, (theta_desc2, r_outer_desc2, idx_desc)
    )
    boundmid_searched = jnp.concatenate([boundmid_rest_desc[::-1], r_outer_top[None]])

    boundmid_grid = jnp.where(notubes, r_outer_grid, boundmid_searched)
    return boundmid_grid, notubes


class XZGridFromBoundaryOrbitSampler(AbstractOrbitSampler):
    """`(x, z)`-plane start space over `(E, theta)`, tube radii searched.

    Same `rmin`/`rmax`/`nE`/`nI1`/`nI2` fields and the same `nE * nI1 * nI2`
    bundle count as `xz_grid_from_origin.XZGridFromOriginOrbitSampler` --
    only *which* radii get sampled differs: this sampler locates `boundin`/
    `boundmid` (van den Bosch et al. 2008 sec. 4.3) and samples between them
    for a regular shell, delegating to
    `xz_grid_from_origin._single_shell_ics` for an irregular one (and every
    shell inward of the first irregular one encountered, walking
    outermost-to-innermost).
    """

    _type: ClassVar[str] = "XZGridFromBoundary"
    # Same free velocity-flip mirror as `XZGridFromOriginOrbitSampler` --
    # see that class's own field for the derivation and caveat.
    _add_reverse_copies: ClassVar[bool] = True
    rmin: Quantity
    rmax: Quantity
    nE: int
    nI1: int
    nI2: int

    def n_bundles(self) -> int:
        return self.nE * self.nI1 * self.nI2

    def generate_ics(self, potential: gp.AbstractPotential) -> jnp.ndarray:
        t0 = Quantity(0.0, potential.units["time"])
        length_unit = potential.units["length"]
        rmin_val = self.rmin.ustrip(length_unit)
        rmax_val = self.rmax.ustrip(length_unit)
        r_floor = rmin_val * _R_FLOOR_FACTOR
        r_ceiling = rmax_val * _R_CEILING_FACTOR

        r_grid = 10.0 ** jnp.linspace(jnp.log10(rmin_val), jnp.log10(rmax_val), self.nE)
        x_axis_points = jnp.stack(
            [r_grid, jnp.zeros_like(r_grid), jnp.zeros_like(r_grid)], axis=-1
        )
        energies = jax.vmap(lambda pos: _potential_at(potential, pos, t0))(
            x_axis_points
        )

        theta_grid = (jnp.arange(self.nI1) + 0.5) * (jnp.pi / 2) / self.nI1

        # One energy shell's boundary search is independent of every
        # other's, but `jax.vmap`-ing this whole per-shell search is
        # catastrophic to compile (a `jax.lax.while_loop` itself calling
        # `diffrax.diffeqsolve`-equivalent stepping, nested inside two
        # levels of `jax.lax.scan`). A plain Python loop over shells
        # instead, calling one `jax.jit`-compiled `per_shell` (compiled
        # once, reused for every subsequent shell -- they all share the
        # same `theta_grid`/`nI2` shape), avoids that blowup.
        @jax.jit
        def per_shell(
            energy: jnp.ndarray, r_x_axis: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            r_outer_grid = jax.vmap(
                lambda th: _equipotential_radius(
                    potential, _xz_direction(th), energy, r_floor, r_ceiling, t0
                )
            )(theta_grid)

            # Two-stage circular-time refinement (`orbitstart_f.f90:78-87`,
            # `111-117`): `_inner_tube_boundary` uses a rough estimate at
            # half this shell's own x-axis radius, while
            # `_outer_tube_boundary` uses a refined one at the x-axis
            # `boundin` `_inner_tube_boundary` itself just found.
            rough_r = 0.5 * r_x_axis
            inner_t_ceiling, inner_classify_time = _shell_integration_budgets(
                potential, rough_r, t0
            )
            boundin_grid, irregular_inner = _inner_tube_boundary(
                potential,
                energy,
                theta_grid,
                r_floor,
                r_outer_grid,
                inner_t_ceiling,
                inner_classify_time,
                t0,
            )

            refined_r = boundin_grid[-1]
            outer_t_ceiling, outer_classify_time = _shell_integration_budgets(
                potential, refined_r, t0
            )
            boundmid_grid, notubes = _outer_tube_boundary(
                potential,
                energy,
                theta_grid,
                boundin_grid,
                r_outer_grid,
                r_floor,
                outer_t_ceiling,
                outer_classify_time,
                self.nI2,
                t0,
            )

            return boundin_grid, boundmid_grid, irregular_inner & ~notubes

        frac = (jnp.arange(1, self.nI2 + 1) - 0.9) / (self.nI2 - 0.8)

        shell_rows: list[jnp.ndarray] = [None] * self.nE  # type: ignore[list-item]

        # Outermost -> innermost, stopping the boundary search the instant a
        # shell comes back locally irregular. That shell, and every one
        # inward of it, delegates directly to `xz_grid_from_origin`'s own
        # uniform-grid-to-the-centre procedure -- the same thing a regular
        # shell's boundin/boundmid search would have fallen back to anyway
        # (see this module's own docstring), reused rather than
        # reimplemented locally.
        propagating = False
        for i in range(self.nE - 1, -1, -1):
            energy, r_x_axis = energies[i], r_grid[i]
            if propagating:
                shell_rows[i] = _single_shell_ics(
                    potential, energy, theta_grid, r_floor, r_ceiling, t0, self.nI2
                )
                continue
            boundin_grid, boundmid_grid, irregular = per_shell(energy, r_x_axis)
            if bool(irregular):
                shell_rows[i] = _single_shell_ics(
                    potential, energy, theta_grid, r_floor, r_ceiling, t0, self.nI2
                )
                propagating = True
                continue
            r_shell = (
                boundin_grid[:, None]
                + frac[None, :] * (boundmid_grid - boundin_grid)[:, None]
            )
            shell_rows[i] = jax.vmap(
                lambda th, r, energy=energy: _xz_orbit_ic(potential, energy, th, r, t0)
            )(
                jnp.broadcast_to(theta_grid[:, None], r_shell.shape).reshape(-1),
                r_shell.reshape(-1),
            )

        return jnp.concatenate(shell_rows, axis=0)
