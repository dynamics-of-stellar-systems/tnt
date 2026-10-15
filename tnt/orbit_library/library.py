"""Bounded-memory library assembly and optional, separate trajectory files.

Reducers own the scientific projections and return one orbit's aggregated
contribution. The host loop retains no library-wide raw trajectory array.
Each requested mass scale is reduced before that orbit's samples are freed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import galax.potential as gp
import jax
import jax.numpy as jnp
import numpy as np

from tnt.orbit_library.data import (
    OrbitInitialConditions,
    OrbitLibrary,
    OrbitTrajectory,
    TrajectoryReference,
)
from tnt.orbit_library.integration import (
    OrbitIntegrationError,
    OrbitIntegrationSettings,
    integrate_orbit,
    require_static_potential,
)
from tnt.validation import _positive_number

type OrbitReducer = Callable[[OrbitTrajectory], jax.Array]
_DEFAULT_SETTINGS = OrbitIntegrationSettings()


def build_orbit_library(
    potential: gp.AbstractPotential,
    initial_conditions: OrbitInitialConditions,
    reducers: Mapping[str, OrbitReducer],
    key: jax.Array,
    settings: OrbitIntegrationSettings = _DEFAULT_SETTINGS,
    *,
    mass_scales: Sequence[float] = (1.0,),
    trajectory_directory: Path | None = None,
    save_bundle_ids: Sequence[int] = (),
) -> dict[float, OrbitLibrary]:
    """Integrate/reduce one member per bundle, returning libraries by mass scale.

    Every reducer receives one temporary trajectory and returns an array
    without a bundle axis. Its array is stacked by bundle in the result.
    Reducers must calculate time-normalized fractions/moments using all
    samples, including time outside an aperture. They retain ownership of
    spatial/velocity bin definitions and units. They must not return raw
    trajectories disguised as contributions. At least one reducer is required.

    ``mass_scales`` multiplies every component's mass together. Positions
    stay fixed, velocities scale by sqrt(s), and times by 1/sqrt(s).
    Reducers are evaluated separately at each scale: bin counts themselves
    are never rescaled. No counter-rotating copies are constructed yet.

    Selected base-mass trajectories are saved as NPZ files outside the
    library. Their metadata records the potential, units, initial state,
    reference period, key, solver/settings, and bundle/member IDs. Existing
    files are never overwritten. A library's ``potential_mass_scale`` states
    how these base-mass files relate to its contributions.

    This eager orchestration is not a differentiable model-search loop.
    Its numerical integration/reduction kernels can be JIT compiled.
    """
    require_static_potential(potential)
    ids, scales, selected = _validate_request(
        initial_conditions,
        reducers,
        mass_scales,
        save_bundle_ids,
        trajectory_directory,
    )
    reduced: dict[float, dict[str, list[jax.Array]]] = {
        scale: {name: [] for name in reducers} for scale in scales
    }
    diagnostics = []
    references = []
    potential_metadata = _potential_metadata(potential) if selected else {}
    for index, bundle_id in enumerate(ids):
        orbit_key = jax.random.fold_in(key, int(bundle_id))
        trajectory = integrate_orbit(
            potential,
            initial_conditions.positions[index, 0],
            initial_conditions.velocities[index, 0],
            initial_conditions.reference_periods[index, 0],
            orbit_key,
            settings,
        )
        if not bool(trajectory.diagnostics.successful):
            raise OrbitIntegrationError(
                f"Orbit integration failed for bundle {bundle_id}."
            )
        for scale in scales:
            values = _reduce(_rescale_trajectory(trajectory, scale), reducers)
            for name, value in values.items():
                if not bool(jnp.all(jnp.isfinite(value))):
                    raise OrbitIntegrationError(
                        f"Nonfinite contribution {name!r} for bundle {bundle_id}."
                    )
                reduced[scale][name].append(value)
        diagnostics.append(trajectory.diagnostics)
        if int(bundle_id) in selected:
            references.append(
                _write_trajectory(
                    trajectory_directory,
                    int(bundle_id),
                    trajectory,
                    initial_conditions,
                    index,
                    orbit_key,
                    settings,
                    potential_metadata,
                )
            )
        del trajectory
    all_diagnostics = jax.tree.map(lambda *xs: jnp.stack(xs)[:, None], *diagnostics)
    return {
        scale: OrbitLibrary(
            contributions={
                name: jnp.stack(rows) for name, rows in contributions.items()
            },
            bundle_ids=initial_conditions.bundle_ids,
            reference_periods=initial_conditions.reference_periods / np.sqrt(scale),
            diagnostics=all_diagnostics,
            trajectories=tuple(references),
            potential_mass_scale=scale,
        )
        for scale, contributions in reduced.items()
    }


def _validate_request(
    starts: OrbitInitialConditions,
    reducers: Mapping[str, OrbitReducer],
    mass_scales: Sequence[float],
    save_bundle_ids: Sequence[int],
    directory: Path | None,
) -> tuple[np.ndarray, tuple[float, ...], set[int]]:
    ids = np.asarray(starts.bundle_ids)
    if len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids > 2**32 - 1):
        raise ValueError("Bundle IDs must be unique unsigned 32-bit integers.")
    for quantity in (starts.positions, starts.velocities, starts.reference_periods):
        if not np.all(np.isfinite(np.asarray(quantity.value))):
            raise ValueError("Initial conditions and periods must be finite.")
    if np.any(np.asarray(starts.reference_periods.value) <= 0):
        raise ValueError("Reference periods must be positive.")
    if not reducers or any(
        not isinstance(name, str) or not name.strip() or not callable(reducer)
        for name, reducer in reducers.items()
    ):
        raise ValueError("Supply at least one named callable orbit reducer.")
    scales = tuple(_positive_number(s, "mass_scales") for s in mass_scales)
    if not scales or len(set(scales)) != len(scales):
        raise ValueError("Supply nonempty, distinct mass scales.")
    selected = set(save_bundle_ids)
    if not selected.issubset(set(ids.tolist())):
        raise ValueError(
            "Requested trajectory bundle IDs are not in the initial conditions."
        )
    if selected and directory is None:
        raise ValueError("A trajectory_directory is required to save trajectories.")
    if selected:
        directory = Path(directory).expanduser().resolve()
        for bundle_id in selected:
            if (directory / f"bundle-{bundle_id}-member-0.npz").exists():
                raise FileExistsError(
                    f"Trajectory for bundle {bundle_id} already exists."
                )
    return ids, scales, selected


@eqx.filter_jit
def _rescale_trajectory(
    trajectory: OrbitTrajectory, mass_scale: float
) -> OrbitTrajectory:
    speed_scale = np.sqrt(mass_scale)
    return OrbitTrajectory(
        times=trajectory.times / speed_scale,
        positions=trajectory.positions,
        velocities=trajectory.velocities * speed_scale,
        reference_period=trajectory.reference_period / speed_scale,
        diagnostics=trajectory.diagnostics,
    )


@eqx.filter_jit
def _reduce(
    trajectory: OrbitTrajectory, reducers: Mapping[str, OrbitReducer]
) -> dict[str, jax.Array]:
    return {
        name: jnp.asarray(reducer(trajectory)) for name, reducer in reducers.items()
    }


def _potential_metadata(potential: gp.AbstractPotential) -> dict[str, Any]:
    if isinstance(potential, Mapping):
        return {
            "type": type(potential).__name__,
            "components": {
                name: _potential_metadata(p) for name, p in potential.items()
            },
        }

    def quantity_data(quantity):
        return {
            "value": np.asarray(quantity.value).tolist(),
            "unit": str(quantity.unit),
        }

    return {
        "type": type(potential).__name__,
        "parameters": {
            name: quantity_data(getattr(potential, name).value)
            for name in potential.parameters
        },
        "constants": {
            name: quantity_data(q) for name, q in potential.constants.items()
        },
        "units": {
            name: str(potential.units[name])
            for name in ("length", "time", "mass", "angle")
        },
    }


def _write_trajectory(
    directory: Path,
    bundle_id: int,
    trajectory: OrbitTrajectory,
    starts: OrbitInitialConditions,
    index: int,
    key: jax.Array,
    settings: OrbitIntegrationSettings,
    potential: dict[str, Any],
) -> TrajectoryReference:
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"bundle-{bundle_id}-member-0.npz"
    metadata = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "member_index": 0,
        "frame": "intrinsic_cartesian",
        "potential_mass_scale": 1.0,
        "potential": potential,
        "potential_id": hashlib.sha256(
            json.dumps(potential, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "solver": "diffrax.Dopri8",
        "settings": asdict(settings),
        "random_key": np.asarray(jax.random.key_data(key)).tolist(),
        "position_unit": str(trajectory.positions.unit),
        "velocity_unit": str(trajectory.velocities.unit),
        "time_unit": str(trajectory.times.unit),
        "reference_period": float(trajectory.reference_period.value),
        "diagnostics": {
            name: np.asarray(value).item()
            for name, value in asdict(trajectory.diagnostics).items()
        },
    }
    # Exclusive creation protects previously saved trajectories, even if
    # another writer races the preflight existence check.
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            times=np.asarray(trajectory.times.value),
            positions=np.asarray(trajectory.positions.value),
            velocities=np.asarray(trajectory.velocities.value),
            initial_position=np.asarray(
                starts.positions[index, 0].ustrip(trajectory.positions.unit)
            ),
            initial_velocity=np.asarray(
                starts.velocities[index, 0].ustrip(trajectory.velocities.unit)
            ),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    return TrajectoryReference(bundle_id=bundle_id, member_index=0, path=str(path))
