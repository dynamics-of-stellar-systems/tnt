"""`GridOrbitSampler`: a box + tube + counter-rotating-tube grid over (E, I2, I3)."""

from __future__ import annotations

from typing import ClassVar

import galax.potential as gp
import jax.numpy as jnp

from tnt.orbit_library.base import AbstractOrbitSampler


class GridOrbitSampler(AbstractOrbitSampler):
    """Box + tube + counter-rotating-tube grid over (E, I2, I3).

    `n_bundles` is `3 * nE * nI2 * nI3`: a box-orbit library and a
    tube-orbit library, each of size `nE * nI2 * nI3`, plus a
    counter-rotating copy of the tube library.
    """

    _type: ClassVar[str] = "Grid"
    # `generate_ics` already builds its own counter-rotating tube copy
    # internally (see `n_bundles`'s own `3 *` factor) -- an external mirror
    # on top would double-count it.
    _add_reverse_copies: ClassVar[bool] = False
    logrmin: float
    logrmax: float
    nE: int
    nI2: int
    nI3: int

    def n_bundles(self) -> int:
        raise NotImplementedError

    def generate_ics(self, potential: gp.AbstractPotential) -> jnp.ndarray:
        raise NotImplementedError
