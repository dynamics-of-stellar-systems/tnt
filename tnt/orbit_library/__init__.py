"""Orbit libraries integrated in a `tnt.potential.Potential`.

Split across submodules by concern: `base` (the abstract
`AbstractOrbitSampler`/`AbstractOrbitDithering` contracts, the dithering
scheme(s), and the `build_orbit_sampler`/`build_orbit_dithering`/`generate_ics`
entry points), `data` (retained contributions and transient trajectories),
`integration` (single-orbit Dopri8 solves), `library` (bounded-memory
aggregation and optional trajectory files),
`common` (equipotential
search and launch-point numerics shared across samplers), and one sibling
module per concrete `AbstractOrbitSampler` (`stationary_grid`,
`xz_grid_from_origin`, `xz_grid_from_boundary`).
"""

from __future__ import annotations

from tnt.orbit_library.base import (
    AbstractOrbitDithering,
    AbstractOrbitSampler,
    CubicOrbitDithering,
    build_orbit_dithering,
    build_orbit_sampler,
    generate_ics,
)
from tnt.orbit_library.data import (
    OrbitInitialConditions,
    OrbitIntegrationDiagnostics,
    OrbitLibrary,
    OrbitTrajectory,
    TrajectoryReference,
)
from tnt.orbit_library.integration import (
    OrbitIntegrationError,
    OrbitIntegrationSettings,
    integrate_orbit,
)
from tnt.orbit_library.library import build_orbit_library
from tnt.orbit_library.stationary_grid import StationaryGridOrbitSampler
from tnt.orbit_library.xz_grid_from_boundary import XZGridFromBoundaryOrbitSampler
from tnt.orbit_library.xz_grid_from_origin import XZGridFromOriginOrbitSampler

__all__ = [
    "AbstractOrbitDithering",
    "AbstractOrbitSampler",
    "CubicOrbitDithering",
    "OrbitInitialConditions",
    "OrbitIntegrationDiagnostics",
    "OrbitIntegrationError",
    "OrbitIntegrationSettings",
    "OrbitLibrary",
    "OrbitTrajectory",
    "StationaryGridOrbitSampler",
    "TrajectoryReference",
    "XZGridFromBoundaryOrbitSampler",
    "XZGridFromOriginOrbitSampler",
    "build_orbit_dithering",
    "build_orbit_library",
    "build_orbit_sampler",
    "generate_ics",
    "integrate_orbit",
]
