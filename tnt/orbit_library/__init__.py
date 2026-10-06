"""Orbit libraries integrated in a `tnt.potential.Potential`.

Signature-only scaffold: every method raises `NotImplementedError`.

Split across submodules by concern: `base` (the abstract
`AbstractOrbitSampler`/`AbstractOrbitDithering` contracts, the dithering
scheme(s), `OrbitLibrary` itself, and the `build_orbit_sampler`/
`build_orbit_dithering` entry points), and one sibling module per concrete
`AbstractOrbitSampler` (`grid`, `random`).
"""

from __future__ import annotations

from tnt.orbit_library.base import (
    AbstractOrbitDithering,
    AbstractOrbitSampler,
    CubicOrbitDithering,
    OrbitLibrary,
    build_orbit_dithering,
    build_orbit_sampler,
)
from tnt.orbit_library.grid import GridOrbitSampler
from tnt.orbit_library.random import RandomOrbitSampler

__all__ = [
    "AbstractOrbitDithering",
    "AbstractOrbitSampler",
    "CubicOrbitDithering",
    "GridOrbitSampler",
    "OrbitLibrary",
    "RandomOrbitSampler",
    "build_orbit_dithering",
    "build_orbit_sampler",
]
