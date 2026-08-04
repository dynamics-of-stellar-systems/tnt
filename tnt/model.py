"""One evaluated point in the model search: a potential and its solved weights.

Signature-only scaffold.
"""

from __future__ import annotations

import equinox as eqx

from tnt.potential import Potential
from tnt.weight_solver import OrbitWeights


class Model(eqx.Module):
    """A `Potential` and its solved `OrbitWeights`/chi2 metrics.

    Deliberately doesn't hold the `OrbitLibrary` it was solved against --
    too large to keep once weights are solved, and not needed again unless
    recomputing. Has no separate mass-scale field either: a model produced
    by `parameter_space_settings.potential_rescalings` shows its rescaled
    mass through `potential`'s own component parameters (`ml`/
    `mge_mass_scale`), the same as any other model.
    """

    potential: Potential
    weights: OrbitWeights
    chi2: dict[str, float]
