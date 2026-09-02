"""Illustration for the prior concept (see the `prior-concept` PR).

Two draws from the same run's `tnt.priors.Prior`:

1.  per-parameter priors only -- `stars.ml ~ Uniform(1, 9)`,
    `dh.m ~ LogUniform(1e10, 1e13)` -- drawn with `numpyro.infer.Predictive`;
2.  the same, plus one prior *plugin*: a `Normal(4e10, 5e9)` `numpyro.factor`
    on the mass enclosed within 10 kpc, `M(<10kpc)`, a derived quantity of
    the whole potential that the plugin gets from
    `context.build_potential()`. Drawn with `numpyro.infer.NUTS`.

Writes `prior_concept_illustration.json` (the raw samples) next to this file.
Run from the repo root:  uv run python aidocs/prior_concept_illustration.py
"""

from __future__ import annotations

import json
from pathlib import Path

import galax.potential as gp
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import unxt as u
from unxt import Quantity

from tnt.mge import LightMGE
from tnt.potential import Potential
from tnt.priors import Prior, PriorContext

_UNITS = u.unitsystem("kpc", "Myr", "Msun", "rad")
_ANGLES = {
    "theta": 0.4280191007651427,
    "phi": 1.0363519257903688,
    "psi": 2.033205815,
}
_R_10_KPC = Quantity(jnp.array([10.0, 0.0, 0.0]), "kpc")
_T0 = Quantity(0.0, "Myr")
_TARGET, _WIDTH = 4.0e10, 5.0e9

_MGE = LightMGE(
    I=Quantity(jnp.array([1.0e3, 2.0e2, 3.0e1]), "Lsun / pc2"),
    sigma=Quantity(jnp.array([0.3, 1.2, 4.0]), "kpc"),
    q=Quantity(jnp.array([0.85, 0.78, 0.90]), ""),
    PA_twist=Quantity(jnp.zeros(3), "rad"),
)

_POTENTIAL = {
    "dh": {
        "type": "NFWPotential",
        "parameters": {
            "m": {
                "unit": "Msun",
                "fixed": False,
                "value": 1.0e12,
                "prior": {"distribution": "LogUniform", "args": [1.0e10, 1.0e13]},
            },
            "r_s": {"unit": "kpc", "fixed": True, "value": 15.0},
        },
    },
    "stars": {
        "type": "TriaxialLightMGEPotential",
        "mge": "stars_mge",
        "parameters": {
            "ml": {
                "unit": "Msun / Lsun",
                "fixed": False,
                "value": 5.0,
                "prior": {"distribution": "Uniform", "args": [1.0, 9.0]},
            },
            **{
                n: {"unit": "rad", "fixed": True, "value": v}
                for n, v in _ANGLES.items()
            },
        },
    },
}


def _m_within_10kpc_plugin(context: PriorContext) -> None:
    import numpyro

    potential = context.build_potential().to_galax(context.unit_system)
    enclosed = gp.spherical_mass_enclosed(potential, _R_10_KPC, _T0).ustrip("Msun")
    numpyro.factor(
        "m_within_10kpc",
        dist.Normal(_TARGET, _WIDTH).log_prob(jnp.asarray(enclosed)),
    )


def _enclosed(resolved, ml: jnp.ndarray, m_halo: jnp.ndarray) -> jnp.ndarray:
    parameters = {
        "dh": {"m": Quantity(m_halo, "Msun"), "r_s": Quantity(15.0, "kpc")},
        "stars": {
            "ml": Quantity(ml, "Msun / Lsun"),
            **{n: Quantity(v, "rad") for n, v in _ANGLES.items()},
        },
    }
    built = Potential.build(resolved, parameters, {}, validate=False).to_galax(_UNITS)
    return gp.spherical_mass_enclosed(built, _R_10_KPC, _T0).ustrip("Msun")


def main() -> None:
    mges = {"stars_mge": _MGE}
    resolved = Potential.resolve(_POTENTIAL, mges)

    per_parameter = Prior(_POTENTIAL, {}, mges)
    a = per_parameter.sample(jax.random.PRNGKey(1), num_samples=3000)

    with_plugin = Prior(
        _POTENTIAL,
        {"m_within_10kpc": _m_within_10kpc_plugin},
        mges,
        resolved_potential=resolved,
        cosmological_parameters={"H": Quantity(70.0, "km / (s Mpc)")},
        unit_system=_UNITS,
    )
    b = with_plugin.sample(jax.random.PRNGKey(0), num_samples=2000, num_warmup=800)

    m_a = jax.vmap(lambda x, y: _enclosed(resolved, x, y))(a["stars.ml"], a["dh.m"])
    m_b = jax.vmap(lambda x, y: _enclosed(resolved, x, y))(b["stars.ml"], b["dh.m"])

    def col(x):
        return [float(f"{v:.4g}") for v in x.tolist()]

    out = {
        "target": _TARGET,
        "width": _WIDTH,
        "per_parameter": {
            "ml": col(a["stars.ml"]),
            "m_halo": col(a["dh.m"]),
            "m_within_10kpc": col(m_a),
        },
        "with_plugin": {
            "ml": col(b["stars.ml"]),
            "m_halo": col(b["dh.m"]),
            "m_within_10kpc": col(m_b),
        },
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")

    def line(label: str, m: jnp.ndarray) -> str:
        return (
            f"{label}: mean {float(jnp.mean(m)):.2e}  std {float(jnp.std(m)):.2e}  "
            f"range [{float(jnp.min(m)):.2e}, {float(jnp.max(m)):.2e}]"
        )

    print(f"wrote {path}")
    print(line("M(<10 kpc)  per-parameter prior ", m_a))
    print(line("M(<10 kpc)  + mass-enclosed prior", m_b))
    print(f"            factor target        : {_TARGET:.2e} +/- {_WIDTH:.0e}")


if __name__ == "__main__":
    main()
