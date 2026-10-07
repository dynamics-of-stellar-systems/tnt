"""`RandomOrbitSampler`: orbit bundles from randomly sampled initial conditions."""

from __future__ import annotations

from typing import ClassVar

import galax.potential as gp
import jax.numpy as jnp

from tnt.orbit_library.base import AbstractOrbitSampler


class RandomOrbitSampler(AbstractOrbitSampler):
    """Orbit bundles from randomly sampled initial conditions.

    Fields beyond `logrmin`/`logrmax` are still undecided -- only `type`,
    `logrmin`, and `logrmax` are validated so far
    (`tnt.configuration.validation._validate_orbit_sampler`).
    """

    _type: ClassVar[str] = "Random"
    # Provisional: fields and orbit-family composition are still undecided
    # (see this class's own docstring), so there's nothing to mirror yet.
    _add_reverse_copies: ClassVar[bool] = False
    logrmin: float
    logrmax: float

    def n_bundles(self) -> int:
        raise NotImplementedError

    def generate_ics(self, potential: gp.AbstractPotential) -> jnp.ndarray:
        raise NotImplementedError
