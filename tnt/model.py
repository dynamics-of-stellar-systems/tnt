"""One evaluated point in the model search: a potential and its solved weights.

Signature-only scaffold.
"""

from __future__ import annotations

import equinox as eqx

from tnt.potential import Potential
from tnt.weight_solver import OrbitWeights


class Model(eqx.Module):
    """A `Potential` and its solved `OrbitWeights`/chi2 metrics, if solved.

    Deliberately doesn't hold the `OrbitLibrary` it was solved against --
    too large to keep once weights are solved, and not needed again unless
    recomputing. Has no separate mass-scale field either: a model produced
    by `parameter_space_settings.potential_rescalings` shows its rescaled
    mass through `potential`'s own component parameters (`ml`/
    `mge_mass_scale`), the same as any other model.

    `potential` is always set -- it's the proposed point being evaluated,
    known before evaluation even starts. `weights`/`chi2` are `None` until
    evaluation actually succeeds: `ModelIterator._evaluate` is responsible
    for setting `orblib_done`/`weights_done` (and `weights`/`chi2`) to
    reflect what actually happened -- `orblib_done` is `False` if orbit
    integration itself failed, and `weights_done` is `False` if weight
    solving failed on its single attempt. `weights_done` implies
    `orblib_done`.
    """

    potential: Potential
    orblib_done: bool
    weights_done: bool
    weights: OrbitWeights | None
    chi2: dict[str, float] | None
    iteration: int
    """The 0-based `ModelIterator.run` search round that produced this model.

    Every model from one round -- including any `potential_rescalings`
    models -- shares the same `iteration`. Lets `AllModels.n_iterations`
    recover how many rounds a resumed run already completed, without a
    separately persisted counter.
    """
