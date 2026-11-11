"""Build the prior-generator guide figures with TNT's diagnostic renderers.

Run manually with:
    uv run python docs/_ext/prior_generator_examples.py --output-directory /tmp/guide
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import galax.potential as gp
import jax.numpy as jnp
import numpyro.distributions as dist
import unxt as u

from tnt.all_models import AllModels
from tnt.mge import LightMGE
from tnt.parameter_generator import build_parameter_generator
from tnt.potential import Potential
from tnt.priors import Prior, PriorContext

POINT_COUNT = 24
SAMPLE_COUNT = 2400
SEED = 42


def enclosed_mass_prior(context: PriorContext) -> None:
    """Prefer total mass within 10 kpc of (4 +/- 0.5) * 10**10 solar masses."""
    potential = context.build_potential().to_galax(context.unit_system)

    def log_probability():
        mass = gp.spherical_mass_enclosed(
            potential,
            u.Quantity([10.0, 0.0, 0.0], "kpc"),
            u.Quantity(0.0, "Myr"),
        ).ustrip("Msun")
        return dist.Normal(4.0e10, 5.0e9).log_prob(jnp.asarray(mass))

    context.factor_if_valid("mass_within_10kpc", log_probability)


def example_prior() -> tuple[dict, Prior]:
    """A spherical two-component galaxy with only dm.m and stars.ml free.

    Spherical geometry makes the enclosed mass unambiguous. The MGE light
    profile, halo scale radius and viewing angle are fixed for both generators.
    """
    settings = {
        "dm": {
            "type": "NFWPotential",
            "parameters": {
                "m": {
                    "unit": "Msun",
                    "fixed": False,
                    "prior": {"distribution": "LogUniform", "args": [3.0e10, 3.0e11]},
                },
                "r_s": {"unit": "kpc", "fixed": True, "value": 15.0},
            },
        },
        "stars": {
            "type": "OblateLightMGEPotential",
            "mge": "light",
            "parameters": {
                "ml": {
                    "unit": "Msun / Lsun",
                    "fixed": False,
                    "prior": {"distribution": "Uniform", "args": [1.0, 9.0]},
                },
                "inclination": {"unit": "deg", "fixed": True, "value": 90.0},
            },
        },
    }
    mges = {
        "light": LightMGE(
            I=u.Quantity(jnp.array([1000.0, 200.0, 30.0]), "Lsun / pc2"),
            sigma=u.Quantity(jnp.array([0.3, 1.2, 4.0]), "kpc"),
            q=u.Quantity(jnp.ones(3), ""),
            PA_twist=u.Quantity(jnp.zeros(3), "rad"),
            major_axis_pa=u.Quantity(0.0, "deg"),
        )
    }
    prior = Prior(
        settings,
        {"enclosed_mass": enclosed_mass_prior},
        mges,
        resolved_potential=Potential.resolve(settings, mges),
        cosmological_parameters={"H": u.Quantity(70.0, "km / (s Mpc)")},
        unit_system=u.unitsystem("kpc", "Myr", "Msun", "rad"),
    )
    return settings, prior


def generate_figures(output_directory: Path) -> None:
    """Generate actual proposals and copy their complete diagnostic figures.

    Each build uses a fresh temporary run directory so saved proposal cursors
    cannot advance between documentation builds. Nothing integrates orbits or
    solves model weights; this example only chooses evaluation locations.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    settings, prior = example_prior()
    with tempfile.TemporaryDirectory(prefix="tnt-prior-guide-") as temporary:
        for kind, figure_name, diagnostic_name in (
            ("PriorSampler", "prior_sampler.png", "prior_diagnostics"),
            (
                "PriorSupportPoints",
                "prior_support_points.png",
                "support_points_diagnostics",
            ),
        ):
            root = Path(temporary) / kind
            generator = build_parameter_generator(
                {
                    "generator_type": kind,
                    "generator_settings": {
                        "seed": SEED,
                        "num_samples": SAMPLE_COUNT,
                        "num_warmup": 600,
                        "target_accept_prob": 0.9,
                        "max_iterations": 1500,
                        "tolerance": 1.0e-4,
                        "plot_diagnostics": True,
                    },
                    "stopping_criteria": {"max_new_mods_per_iter": POINT_COUNT},
                },
                settings,
                prior,
                output_directory=root,
            )
            batch = generator.generate_parameters(AllModels())
            if len(batch.parameters) != POINT_COUNT:
                raise RuntimeError(f"{kind} returned {len(batch.parameters)} points")
            optimization = batch.diagnostics.get("optimization")
            if optimization is not None:
                print(
                    f"Support optimization: {optimization.status}; "
                    f"{optimization.iterations} steps; "
                    f"displacement {optimization.final_maximum_displacement}"
                )
            sampling = batch.diagnostics["sampling"]
            print(
                f"{kind}: {POINT_COUNT} proposals; "
                f"{int(sampling.diverging.sum())} divergent draws; "
                f"ESS {sampling.ess}; split R-hat {sampling.split_rhat}"
            )
            shutil.copyfile(
                generator.diagnostic_path(diagnostic_name, iteration=0),
                output_directory / figure_name,
            )


def _build_guide(app) -> None:
    from sphinx.util import logging

    logging.getLogger(__name__).info("Generating prior-generator comparison figures")
    generate_figures(Path(app.srcdir) / "_generated" / "prior_generators")


def setup(app):
    app.connect("builder-inited", _build_guide)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    generate_figures(parser.parse_args().output_directory)
