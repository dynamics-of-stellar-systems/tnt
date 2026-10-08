# Priors

A `prior` on a potential parameter (`{distribution: "<numpyro.distributions
class>", args: [...]}`, sibling to `value`/`unit`/`fixed`) declares its
search-space distribution and compiles to one `numpyro.sample` site, keyed by
`"<component>.<parameter>"`. Every non-fixed parameter must declare one;
configuration preparation rejects a `fixed: false` parameter with no `prior`.
`PriorSampler` composes every non-fixed
parameter across the whole `potential` section, plus every
configured `parameter_space_settings.priors` plugin, into one numpyro model
per run (`tnt.priors.Prior`) and draws candidates from it.

A prior **plugin** is a user's own Python function -- not a class, not
shipped by TNT itself (there are no built-in priors; even the canonical
mass-fraction example below is documentation, not a package feature) --
referenced from config as `plugin: "<path>:<function_name>"`, with `<path>`
resolved relative to `io_settings.input_directory`, the same convention MGE,
kinematics, and population files already use:

```yaml
parameter_space_settings:
  priors:
    dh_mass_fraction:
      plugin: "priors/mass_fraction.py:mass_fraction"
```

A plugin function may only call `numpyro.factor` -- never `numpyro.sample` or
`numpyro.deterministic`. This is what lets ordinary per-parameter priors and
plugins compose safely with no coordination between them: a plugin can add a
soft log-density preference over values already established elsewhere, but
it can never independently assign or overwrite a parameter's value, so it can
never collide with that parameter's own declared `prior`.

**The signature is fixed:** a plugin is callable with one positional argument,
a `tnt.priors.PriorContext`, and returns nothing. (A trailing parameter with a
default, or `*args`, is tolerated but never populated; any other arity is
rejected when the plugin is loaded.) `PriorContext` carries the run's
state a plugin is allowed to read -- currently `context.candidate` (the
parameter values assembled so far this draw, `{component: {parameter: value}}`,
bare unit-stripped numbers) and `context.mges` (the run's named MGEs). New
fields are added there over time; because a plugin only ever names this one
argument, that stays backward-compatible.

For a preference over a *derived* quantity of the whole potential -- an
enclosed mass, a circular velocity -- call `context.build_potential()`. It
assembles this draw's `tnt.potential.Potential` from `context.candidate`
via `Potential.build_with_validity`, itself JAX-traceable;
`context.build_potential().to_galax(context.unit_system)` then gives a
`galax` potential to evaluate. The run's resolved potential structure and
cosmology are captured when the `Prior` is built -- a plugin never handles
them. The first call in a draw registers that draw's validity as a
`numpyro.factor` site -- an invalid deprojection geometry gets exactly
`-inf` log-probability, not a `nan` relying on gradient propagation to get
rejected -- and caches the result, so a later call in the same draw, from
the same plugin or another, is free.

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from tnt.priors import PriorContext


def mass_fraction(context: PriorContext) -> None:
    stars = context.mges["stars"]
    candidate = context.candidate
    # `candidate`'s values are bare (unit-stripped) numbers, so `total_light`
    # is stripped to match -- here to Lsun, paired with `ml`'s Msun / Lsun.
    # Stripping the per-component term before summing (rather than summing
    # the `Quantity` and stripping after) also sidesteps a `jax.numpy.sum`
    # incompatibility with `unxt.Quantity`'s multi-alias physical types
    # (Lsun's `PhysicalType` has two names, `{'power', 'radiant flux'}`).
    total_light = jnp.sum(
        (2 * jnp.pi * stars.I * stars.sigma**2 * stars.q).ustrip("Lsun")
    )
    f = candidate["dh"]["M_200"] / (total_light * candidate["stars"]["ml"])
    # A smooth distribution here (Normal, TruncatedNormal, ...), not a hard
    # Uniform: MCMC/NUTS needs real gradient signal to respect a factor.
    # Uniform.log_prob is flat inside its support and discontinuous at the
    # edges, so NUTS gets no useful gradient and the constraint is not
    # actually enforced in practice -- verified empirically, not just
    # theoretically, during this feature's implementation.
    numpyro.factor("dh_mass_fraction", dist.Normal(0.2, 0.1).log_prob(f))
```

`M_200` here is `dh`'s own ordinary `prior` (e.g. `LogUniform`), not declared
or derived by the plugin -- NFW's existing `concentration_m200`
parameterization (see [Potential](potential.md)) is unmodified; the plugin
only adds a preference on the ratio, never assigns `M_200` directly.

A plugin that needs the assembled potential -- e.g. a preference on the mass
enclosed within 10 kpc:

```python
import galax.potential as gp

from unxt import Quantity


def m_within_10kpc(context: PriorContext) -> None:
    potential = context.build_potential().to_galax(context.unit_system)
    enclosed = gp.spherical_mass_enclosed(
        potential, Quantity([10.0, 0.0, 0.0], "kpc"), Quantity(0.0, "Myr")
    ).ustrip("Msun")
    numpyro.factor(
        "m_within_10kpc",
        dist.Normal(4.0e10, 5.0e9).log_prob(jnp.asarray(enclosed)),
    )
```

A `-inf` validity factor keeps a plugin like this one honest even when the
rest of the prior routinely proposes a physically invalid shape -- e.g. an
independent `Uniform(0, 1)` prior on each of an MGE's `pqu`-parameterized
`p`, `q`, `u` satisfies TNT's `0 < q <= p <= 1`, `p < u <= 1` convention only
a small fraction of the time. NUTS still concentrates almost entirely inside
that narrow valid region rather than needing it fixed at an already-valid
point (see `test_plugin_factor_on_enclosed_mass_rejects_invalid_pqu_shape_draws`
in `tests/unit_tests/test_priors.py`).

`Prior.sample` (called by `PriorSampler`) traces the composed model once and
picks how to draw from it automatically: `numpyro.infer.Predictive`
(unconditioned prior-predictive sampling) if the model has no factor sites,
otherwise `numpyro.infer.MCMC`/`NUTS` over the prior+factor joint. Both paths
never touch chi2 or orbit integration -- `PriorSampler` doesn't read
`all_models`. Genuine posterior sampling (conditioning on a `Model`'s actual
chi2) needs a bridge turning that into a `numpyro.factor`, which doesn't
exist yet; that's real, separate future work reusing the same composed-model
machinery, not something `PriorSampler` does today.
