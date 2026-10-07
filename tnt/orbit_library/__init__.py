"""Orbit libraries integrated in a `tnt.potential.Potential`.

Split across submodules by concern: `base` (the abstract
`AbstractOrbitSampler`/`AbstractOrbitDithering` contracts, the dithering
scheme(s), `OrbitLibrary` itself, and the `build_orbit_sampler`/
`build_orbit_dithering`/`generate_ics` entry points), `common` (equipotential
search and launch-point numerics shared across samplers), and one sibling
module per concrete `AbstractOrbitSampler` (`grid`, `random`,
`stationary_grid`, `xz_grid_from_origin`, `xz_grid_from_boundary`).
"""

from __future__ import annotations

from tnt.orbit_library.base import (
    AbstractOrbitDithering,
    AbstractOrbitSampler,
    CubicOrbitDithering,
    OrbitLibrary,
    build_orbit_dithering,
    build_orbit_sampler,
    generate_ics,
)
from tnt.orbit_library.grid import GridOrbitSampler
from tnt.orbit_library.random import RandomOrbitSampler
from tnt.orbit_library.stationary_grid import StationaryGridOrbitSampler
from tnt.orbit_library.xz_grid_from_boundary import XZGridFromBoundaryOrbitSampler
from tnt.orbit_library.xz_grid_from_origin import XZGridFromOriginOrbitSampler

__all__ = [
    "AbstractOrbitDithering",
    "AbstractOrbitSampler",
    "CubicOrbitDithering",
    "GridOrbitSampler",
    "OrbitLibrary",
    "RandomOrbitSampler",
    "StationaryGridOrbitSampler",
    "XZGridFromBoundaryOrbitSampler",
    "XZGridFromOriginOrbitSampler",
    "build_orbit_dithering",
    "build_orbit_sampler",
    "generate_ics",
]
