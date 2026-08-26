"""One evaluated point in the model search: a potential and its solved weights.

Signature-only scaffold.
"""

from __future__ import annotations

import equinox as eqx
from unxt import Quantity

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

    `potential` is `None` only if building it from the proposed point failed
    outright (e.g. `tnt.mge.MGEDeprojectionError` for an invalid MGE viewing
    geometry) -- otherwise it's always set, since it's the proposed point
    being evaluated, known before orbit integration starts.
    `ModelIterator._evaluate` is responsible for setting `valid_potential`/
    `orblib_done`/`weights_done` (and `weights`/`chi2`) to reflect what
    actually happened: `valid_potential` is `False` if `potential` itself
    couldn't be built, `orblib_done` is `False` if orbit integration itself
    failed (implies `valid_potential` is `True` -- orbit integration was
    only attempted because building the potential succeeded), and
    `weights_done` is `False` if weight solving failed on its single
    attempt (implies `orblib_done`).
    """

    potential: Potential | None
    valid_potential: bool
    raw_parameters: dict[str, dict[str, Quantity]]
    """`potential`'s components, in their configuration's own parameterization.

    See `tnt.potential.raw_potential_parameters` -- computed once alongside
    `potential` itself (including for each `potential_rescalings` variant,
    since a mass rescale changes these values), since recomputing it needs
    context (`potential_settings`, `cosmological_parameters`) that `Model`
    doesn't otherwise carry. `AllModels` reads this directly to build its
    per-parameter table columns. When `potential is None`, this is instead
    the raw proposed `tnt.parameter_generator.ParameterSet` `_evaluate`
    received -- not parameterization-inverted, since there's no successfully
    built `Potential` to invert against.
    """
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
