"""Dopri8 integration of a single static-potential orbit.

The reference period is supplied explicitly. It is DYNAMITE's energy-shell
clock, not a period measured from the orbit being integrated. Neither file
I/O nor observational projection belongs in this numerical kernel.
"""

from __future__ import annotations

from collections.abc import Mapping

import diffrax
import equinox as eqx
import galax.potential as gp
import jax
import jax.numpy as jnp
from galax.potential.params import ConstantParameter
from unxt import Quantity

from tnt.orbit_library.common import _phase_space_derivs
from tnt.orbit_library.data import OrbitIntegrationDiagnostics, OrbitTrajectory
from tnt.validation import _integer, _positive_number


class OrbitIntegrationError(RuntimeError):
    """An orbit could not be integrated into a usable trajectory."""


class OrbitIntegrationSettings(eqx.Module):
    """Static compilation settings for one Dopri8 solve.

    ``accuracy`` is the relative and absolute tolerance on normalized
    positions/velocities, not on dimensioned coordinates. The normalization
    uses a launch length scale and its reference-period velocity scale.
    ``max_steps`` counts accepted and rejected adaptive steps. These settings
    deliberately expose no unvalidated alternative solver or automatic retry.
    """

    orbital_periods: float = eqx.field(static=True, default=200.0)
    n_stored_timesteps: int = eqx.field(static=True, default=50_000)
    accuracy: float = eqx.field(static=True, default=1e-5)
    max_steps: int = eqx.field(static=True, default=1_000_000)

    def __check_init__(self) -> None:
        _positive_number(self.orbital_periods, "orbital_periods")
        _positive_number(self.accuracy, "accuracy")
        for name in ("n_stored_timesteps", "max_steps"):
            if _integer(getattr(self, name), name) < 1:
                raise ValueError(f"{name} must be positive.")


def require_static_potential(potential: gp.AbstractPotential) -> None:
    """Reject unsupported or time-dependent force models before freezing time.

    Supported inputs are TNT's curated native Galax classes and the two
    Gaussian classes used by TNT MGEs, optionally composed. Nonconstant
    parameter objects, rotating Monari bars, transformed potentials, and
    arbitrary custom subclasses require an explicit future extension.
    """
    # Import at call time: the potential package itself imports orbit data.
    from tnt.potential.registry import parameter_schema_is_known

    if not isinstance(potential, gp.AbstractPotential):
        raise TypeError("Integration requires a Galax potential.")
    if isinstance(potential, Mapping):
        if type(potential) is not gp.CompositePotential:
            raise NotImplementedError(
                "Only native CompositePotential compositions are validated."
            )
        if not potential:
            raise ValueError("Cannot integrate an empty composite potential.")
        for component in potential.values():
            require_static_potential(component)
        return
    name = type(potential).__name__
    known = parameter_schema_is_known(name, None) or name in {
        "AxisymmetricGaussianPotential", "TriaxialGaussianPotential"
    }
    if not known or getattr(gp, name, None) is not type(potential):
        raise NotImplementedError(f"Static integration is not validated for {name}.")
    if isinstance(potential, gp.MonariEtAl2016BarPotential):
        raise NotImplementedError("Rotating bar integration is not implemented.")
    for parameter_name in potential.parameters:
        if not isinstance(getattr(potential, parameter_name), ConstantParameter):
            raise NotImplementedError(
                "Time-dependent potential parameters are unsupported."
            )


_DEFAULT_SETTINGS = OrbitIntegrationSettings()


@eqx.filter_jit
def integrate_orbit(
    potential: gp.AbstractPotential,
    position: Quantity,
    velocity: Quantity,
    reference_period: Quantity,
    key: jax.Array,
    settings: OrbitIntegrationSettings = _DEFAULT_SETTINGS,
) -> OrbitTrajectory:
    """Integrate one orbit and return uniform-time samples plus diagnostics.

    The solve spans ``settings.orbital_periods * reference_period``. For
    ``N`` samples, DYNAMITE uses spacing ``duration / (N + 4)`` and first
    sample ``spacing * (1 + uniform_random_offset)``. The exact solve endpoint
    is evaluated separately for diagnostics, and is not a histogram sample.

    The caller owns the JAX random key. A failed solver returns diagnostics
    with ``successful=False``; its samples must not be used. Invalid inputs
    raise explicitly, including under JIT. Production orchestration raises
    ``OrbitIntegrationError`` on unsuccessful diagnostics. The numerical
    kernel supports differentiation with fixed sample count and settings;
    this does not make hard histogram binning differentiable.
    """
    require_static_potential(potential)
    length_unit, time_unit = potential.units["length"], potential.units["time"]
    speed_unit = length_unit / time_unit
    q0, v0 = position.ustrip(length_unit), velocity.ustrip(speed_unit)
    period = reference_period.ustrip(time_unit)
    if q0.shape != (3,) or v0.shape != (3,) or period.shape != ():
        raise ValueError(
            "Expected position/velocity (3,) and a scalar reference period."
        )
    initial = jnp.concatenate([q0, v0]).astype(jnp.result_type(q0, v0, 1.0))
    initial = eqx.error_if(
        initial, ~jnp.all(jnp.isfinite(initial)), "Initial coordinates must be finite."
    )
    period = eqx.error_if(
        period, ~jnp.isfinite(period) | (period <= 0),
        "The reference period must be finite and positive."
    )
    return _solve(potential, initial, period, key, settings)


def _solve(
    potential: gp.AbstractPotential,
    initial: jax.Array,
    period: jax.Array,
    key: jax.Array,
    settings: OrbitIntegrationSettings,
) -> OrbitTrajectory:
    length_unit, time_unit = potential.units["length"], potential.units["time"]
    speed_unit = length_unit / time_unit
    length_scale = jnp.maximum(
        jnp.linalg.norm(initial[:3]),
        jnp.linalg.norm(initial[3:]) * period / (2 * jnp.pi),
    )
    length_scale = eqx.error_if(
        length_scale, ~jnp.isfinite(length_scale) | (length_scale <= 0),
        "A nonzero finite position or velocity scale is required."
    )
    speed_scale = 2 * jnp.pi * length_scale / period
    scales = jnp.concatenate([jnp.full(3, length_scale), jnp.full(3, speed_scale)])
    rhs = _phase_space_derivs(potential, Quantity(0.0, time_unit))

    def normalized_rhs(t: jax.Array, y: jax.Array, args: None) -> jax.Array:
        return period * rhs(t * period, y * scales, args) / scales

    step = settings.orbital_periods / (settings.n_stored_timesteps + 4)
    offset = jax.random.uniform(key, dtype=initial.dtype)
    sample_times = step * (1 + offset + jnp.arange(settings.n_stored_timesteps))
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(normalized_rhs), diffrax.Dopri8(),
        t0=0.0, t1=settings.orbital_periods, dt0=None, y0=initial / scales,
        saveat=diffrax.SaveAt(ts=sample_times, t1=True),
        stepsize_controller=diffrax.PIDController(
            rtol=settings.accuracy, atol=settings.accuracy
        ),
        max_steps=settings.max_steps, throw=False,
    )
    states = solution.ys * scales
    diagnostics = _diagnostics(potential, initial, states, speed_scale, solution)
    return OrbitTrajectory(
        times=Quantity(sample_times * period, time_unit),
        positions=Quantity(states[:-1, :3], length_unit),
        velocities=Quantity(states[:-1, 3:], speed_unit),
        reference_period=Quantity(period, time_unit), diagnostics=diagnostics,
    )


def _diagnostics(
    potential: gp.AbstractPotential,
    initial: jax.Array,
    states: jax.Array,
    speed_scale: jax.Array,
    solution: diffrax.Solution,
) -> OrbitIntegrationDiagnostics:
    length_unit, time_unit = potential.units["length"], potential.units["time"]
    all_states = jnp.concatenate([initial[None, :], states])
    phi = potential.potential(
        Quantity(all_states[:, :3], length_unit), Quantity(0.0, time_unit)
    ).ustrip(length_unit**2 / time_unit**2)
    energy = phi + 0.5 * jnp.sum(all_states[:, 3:] ** 2, axis=-1)
    errors = jnp.abs(energy[1:] - energy[0]) / speed_scale**2
    finite = jnp.all(jnp.isfinite(states)) & jnp.all(jnp.isfinite(errors))
    return OrbitIntegrationDiagnostics(
        successful=(solution.result == diffrax.RESULTS.successful) & finite,
        max_energy_error=jnp.max(errors), final_energy_error=errors[-1],
        num_steps=jnp.asarray(solution.stats["num_steps"]),
        num_accepted_steps=jnp.asarray(solution.stats["num_accepted_steps"]),
        num_rejected_steps=jnp.asarray(solution.stats["num_rejected_steps"]),
    )
