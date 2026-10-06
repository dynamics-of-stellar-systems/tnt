"""`RandomOrbitSampler`: orbit bundles from randomly sampled initial conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp

from tnt.orbit_library.base import AbstractOrbitDithering, AbstractOrbitSampler

if TYPE_CHECKING:
    from tnt.potential import Potential


class RandomOrbitSampler(AbstractOrbitSampler):
    """Orbit bundles from randomly sampled initial conditions.

    Fields beyond `logrmin`/`logrmax` are still undecided -- only `type`,
    `logrmin`, and `logrmax` are validated so far
    (`tnt.configuration.validation._validate_orbit_sampler`).
    """

    _type: ClassVar[str] = "Random"
    logrmin: float
    logrmax: float

    def n_bundles(self) -> int:
        raise NotImplementedError

    def generate_ics(
        self,
        potential: Potential,
        dithering: AbstractOrbitDithering,
    ) -> jnp.ndarray:
        raise NotImplementedError
