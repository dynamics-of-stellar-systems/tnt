"""Retained orbit contributions and separate, transient trajectory data.

The leading library axis is always the fitted bundle. Individual trajectories
are temporary inputs to contribution calculations, not fields of a library.
Only one member per bundle is supported in the first integration milestone.
"""

from __future__ import annotations

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from unxt import Quantity


class OrbitIntegrationDiagnostics(eqx.Module):
    """Scalar diagnostics for one orbit, or arrays indexed by bundle/member.

    Energy errors are divided by a characteristic kinetic-energy scale,
    independent of the potential's arbitrary additive energy zero.
    ``successful`` requires solver success and finite trajectory/diagnostics;
    it does not impose an uncalibrated physical energy-error threshold.
    """

    successful: jax.Array
    max_energy_error: jax.Array
    final_energy_error: jax.Array
    num_steps: jax.Array
    num_accepted_steps: jax.Array
    num_rejected_steps: jax.Array


class OrbitTrajectory(eqx.Module):
    """Uniform-time samples for one orbit, kept outside ``OrbitLibrary``.

    Positions and velocities have shape ``(n_samples, 3)``; times have
    shape ``(n_samples,)``. These arrays can be reduced, optionally written,
    and released before the next orbit is integrated. Stored times exclude
    the launch point, following DYNAMITE's offset sampling convention.
    """

    times: Quantity
    positions: Quantity
    velocities: Quantity
    reference_period: Quantity
    diagnostics: OrbitIntegrationDiagnostics

    def __check_init__(self) -> None:
        self.times.ustrip("Myr")
        self.positions.ustrip("kpc")
        self.velocities.ustrip("kpc/Myr")
        self.reference_period.ustrip("Myr")
        n_samples = self.times.shape[0] if self.times.ndim == 1 else -1
        if n_samples < 1 or self.positions.shape != (n_samples, 3):
            raise ValueError(
                "Trajectory times/positions must have shapes (n,) and (n, 3)."
            )
        if self.velocities.shape != self.positions.shape:
            raise ValueError("Trajectory velocities must have the positions' shape.")
        if self.reference_period.shape != ():
            raise ValueError("A trajectory must have one scalar reference period.")


class OrbitInitialConditions(eqx.Module):
    """Launch coordinates and explicit DYNAMITE reference clocks.

    Positions/velocities have shape ``(n_bundles, 1, 3)`` and
    reference periods ``(n_bundles, 1)``. The size-one member axis is
    intentional: cubic dithering beyond one remains unsupported. Bundle
    IDs are stable zero-based identifiers, including after selecting a
    subset. A clock can be shared by stationary and tube populations;
    it must not be inferred from the launch radius of each individual orbit.
    """

    positions: Quantity
    velocities: Quantity
    reference_periods: Quantity
    bundle_ids: jax.Array

    def __check_init__(self) -> None:
        self.positions.ustrip("kpc")
        self.velocities.ustrip("kpc/Myr")
        self.reference_periods.ustrip("Myr")
        if self.positions.ndim != 3 or self.positions.shape[1:] != (1, 3):
            raise ValueError("Initial positions must have shape (n_bundles, 1, 3).")
        n_bundles = self.positions.shape[0]
        if n_bundles < 1 or self.velocities.shape != self.positions.shape:
            raise ValueError("Initial velocities must match nonempty positions.")
        if self.reference_periods.shape != (n_bundles, 1):
            raise ValueError("Reference periods must have shape (n_bundles, 1).")
        if self.bundle_ids.shape != (n_bundles,):
            raise ValueError("Bundle IDs must have shape (n_bundles,).")
        if not jnp.issubdtype(self.bundle_ids.dtype, jnp.integer):
            raise ValueError("Bundle IDs must be integers.")


class OrbitLibrary(eqx.Module):
    """Retained, independently weightable contributions for each bundle.

    Each named contribution array has a leading ``n_bundles`` axis. The
    remaining axes and units belong to its producer (e.g. a kinematics
    object's spatial/velocity histogram or an intrinsic mass grid).
    Producers must preserve fractions outside observational apertures;
    an aperture's histogram is not independently normalized to unity.

    Diagnostics and periods have shape ``(n_bundles, 1)``. Raw trajectory
    arrays are deliberately absent. Optional trajectory files use the same
    bundle IDs and member index zero, and live in separate storage.
    """

    contributions: dict[str, jax.Array]
    bundle_ids: jax.Array
    reference_periods: Quantity
    diagnostics: OrbitIntegrationDiagnostics

    def __check_init__(self) -> None:
        self.reference_periods.ustrip("Myr")
        if self.bundle_ids.ndim != 1 or self.bundle_ids.size < 1:
            raise ValueError("A library requires a nonempty vector of bundle IDs.")
        if not jnp.issubdtype(self.bundle_ids.dtype, jnp.integer):
            raise ValueError("Bundle IDs must be integers.")
        shape = (self.bundle_ids.size, 1)
        if self.reference_periods.shape != shape:
            raise ValueError(
                "Library reference periods must have shape (n_bundles, 1)."
            )
        for diagnostic in jax.tree.leaves(self.diagnostics):
            if diagnostic.shape != shape:
                raise ValueError("Library diagnostics must have shape (n_bundles, 1).")
        for name, values in self.contributions.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Contribution names must be nonempty strings.")
            if values.ndim < 1 or values.shape[0] != shape[0]:
                raise ValueError(
                    f"Contribution {name!r} must have a leading bundle axis."
                )

    def rescaled(self, mass_scale: float) -> Self:
        """Aggregate data cannot be generically rescaled after binning.

        Produce contributions at each requested mass scale while temporary
        trajectories are available. A future histogram-specific rescaling
        contract must state its rebinning approximation explicitly.
        """
        raise NotImplementedError(
            "Recompute contributions at each mass scale before discarding trajectories."
        )
