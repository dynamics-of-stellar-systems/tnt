# Priors

A `prior` on a potential parameter (`{distribution: "<numpyro.distributions
class>", args: [...]}`, sibling to `value`/`unit`/`fixed`) declares its
search-space distribution and compiles to one `numpyro.sample` site, keyed by
`"<component>.<parameter>"`. Every non-fixed parameter must declare one --
configuration preparation rejects a `fixed: false` parameter with no
`prior` -- **except** a parameter covered by a *default prior* (see below),
which is sampled regardless of whether it declares one.
`PriorSampler` composes every non-fixed
parameter across the whole `potential` section, plus every
configured `parameter_space_settings.priors` plugin, into one numpyro model
per run (`tnt.priors.Prior`) and draws candidates from it.

## Default priors

Some parameters have a joint validity constraint that genuinely can't be
expressed as an independent per-parameter `prior`, because it depends on
the component's own MGE data, not just on sibling parameters declared in
the same config. For these, TNT always samples a built-in *default prior*
instead -- there's no opt-in setting, and the covered parameters don't need
(but may still have) a declared `prior` of their own. Currently registered:

- `TriaxialLightMGEPotential` / `TriaxialMassMGEPotential` with
  `parameterization: "pqu"` -- `p`, `q`, and `u` are all covered. With none
  of the three fixed: `p ~ Uniform(0, 1)`; `q ~ Uniform(0, min(p, q'))`,
  `q'` being this MGE's own anchor axis ratio (`min` of its components'
  observed `q`) -- not just `q <= p`, because `q`'s *intrinsic* axis ratio
  can never exceed its own anchor Gaussian's *observed* one (projection
  can only make a shape look rounder than it truly is, never flatter), and
  without that bound a sampled `(p, q)` can leave no valid `u` at all, for
  any viewing angle; `u ~ Uniform(...)` within the MGE-dependent domain
  `potential.md` describes, margin-adjusted for numerical precision near
  its endpoints. Fixing any one of `p`/`q`/`u` tightens the other(s)'
  bounds the same way -- e.g. fixing `q` bounds a sampled `p` below by
  that same `q`, not just the reverse -- rather than leaving them to
  sample as if nothing were fixed (see `_pqu_default_prior_sample`'s own
  docstring for the exact bound in every fixed/free combination). The
  mass parameter (`ml` or `mge_mass_scale`) is *not* covered -- it still
  needs its own declared `prior` if non-fixed.
- `OblateLightMGEPotential` / `OblateMassMGEPotential` with
  `parameterization: "q_min"` -- `q_min ~ Uniform(0, q')`, the oblate
  counterpart of the above (`q'` is the same anchor quantity). Again, the
  mass parameter is not covered.

A declared `prior` on a covered parameter is not rejected, and it does not
replace the default-prior sample: it's folded in as an *additional*
`numpyro.factor` scoring the already-sampled value, the same way a prior
plugin's factor would. A default prior itself is always built from plain
`numpyro.sample` statements and never adds a factor of its own, so a
component covered only by default priors -- no plugins, no extra declared
priors on covered parameters -- still stays on the cheap `Predictive`
sampling path described below.

Programmatically, `tnt.potential.registry.get_default_prior(type_name,
parameterization)` returns the registered `DefaultPriorSpec` (or `None`),
and `tnt.potential.triaxial_mge`/`tnt.potential.oblate_mge` hold the actual
registered sampling functions, each with the full derivation in its own
docstring.

## Prior plugins

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
them. The first call in a draw -- whether to `build_potential()`,
`build_potential_with_validity()`, or `factor_if_valid()` -- registers that
draw's validity as a `numpyro.factor` site: an invalid deprojection
geometry gets exactly `-inf` log-probability, not a `nan` relying on
gradient propagation to get rejected. The result is cached, so a later
call in the same draw, from the same plugin or another, is free.

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
enclosed within 10 kpc -- should register its factor through
`context.factor_if_valid(name, compute)` rather than calling
`numpyro.factor` directly:

```python
import galax.potential as gp

from unxt import Quantity


def m_within_10kpc(context: PriorContext) -> None:
    potential = context.build_potential().to_galax(context.unit_system)

    def enclosed_mass_log_prob():
        enclosed = gp.spherical_mass_enclosed(
            potential, Quantity([10.0, 0.0, 0.0], "kpc"), Quantity(0.0, "Myr")
        ).ustrip("Msun")
        return dist.Normal(4.0e10, 5.0e9).log_prob(jnp.asarray(enclosed))

    context.factor_if_valid("m_within_10kpc", enclosed_mass_log_prob)
```

`factor_if_valid` evaluates `compute` only when this draw's potential is
valid, via `jax.lax.cond` (which evaluates, and differentiates, only the
taken branch) rather than `jnp.where` (which would still evaluate and
differentiate both). This matters because the `-inf` validity factor by
itself does not make the *whole* model safe: on an invalid draw, a derived
computation like this one can divide by a zero-placeholder parameter and
produce a `NaN`, and `-inf + NaN` is `NaN`, not `-inf` -- silently undoing
the validity factor's rejection rather than reinforcing it. Calling
`numpyro.factor` directly here, unconditionally, would reintroduce exactly
the failure mode the `-inf` factor exists to avoid.

A `-inf` validity factor keeps a plugin like this one honest even on the
rare draw the rest of the prior can't rule out up front -- `pqu`'s default
prior (see above) already keeps the vast majority of `(p, q, u)` draws
valid by construction, but a residual sliver (the MGE-dependent domain
narrowing to nothing at the active floating-point precision) still needs
catching downstream. NUTS still concentrates almost entirely inside the
valid region rather than needing it fixed at an already-valid point (see
`test_default_prior_keeps_pqu_draws_valid_without_a_declared_prior` in
`tests/unit_tests/test_priors.py`).

`Prior.sample` (called by `PriorSampler`) traces the composed model once and
picks how to draw from it automatically: `numpyro.infer.Predictive`
(unconditioned prior-predictive sampling) if the model has no factor sites,
otherwise `numpyro.infer.MCMC`/`NUTS` over the prior+factor joint. Both paths
never touch chi2 or orbit integration -- `PriorSampler` doesn't read
`all_models`. Genuine posterior sampling (conditioning on a `Model`'s actual
chi2) needs a bridge turning that into a `numpyro.factor`, which doesn't
exist yet; that's real, separate future work reusing the same composed-model
machinery, not something `PriorSampler` does today.

## Enabling `PriorSampler`

`parameter_space_settings.generator_type: "PriorSampler"` selects it in
place of the packaged default (`GridSearch`). Its own `generator_settings`
-- `seed`, `num_samples`, `num_warmup`, `target_accept_prob` -- already have
packaged defaults (`tnt/defaults/default_config.yaml`), the same way
`GridSearch`'s own `delta_chi2_threshold` does, so switching `generator_type`
alone is enough to run with them; declare any of the four yourself only to
override:

```yaml
parameter_space_settings:
  generator_type: "PriorSampler"
  which_chi2: "kinchi2"
  generator_settings:
    seed: 0
    num_samples: 1000
    num_warmup: 1000
    target_accept_prob: 0.8
  stopping_criteria:
    max_new_mods_per_iter: 50
    n_new_iter: 10
    target_model_count: 500
    minimum_delta_chi2: {enabled: true, mode: "absolute", value: 1.0}
  potential_rescalings:
    enabled: false
    range_count: 10
    mass_scale_range: {minimum: 0.1, maximum: 10.0}
    spacing: "logarithmic"
  priors:
    dh_mass_fraction:
      plugin: "priors/mass_fraction.py:mass_fraction"
```

`num_samples` is how large a batch `Prior.sample` actually draws each call
-- independent of `stopping_criteria.max_new_mods_per_iter` (the generic,
cross-generator cap on how many candidates `generate_parameters` *returns*).
They used to be conflated; keeping them separate is what lets `num_samples`
carry its own, statistically-motivated floor (configuration preparation
requires `num_samples > 100` and `num_warmup > 100` -- a batch or warmup
run that small isn't a meaningful MCMC/`Predictive` draw) independent of
`max_new_mods_per_iter`, which a user is free to set much lower (e.g. to
throttle how many new models one iteration evaluates). Preparation also
requires `num_samples >= max_new_mods_per_iter`, so `PriorSampler` never
silently returns fewer candidates than requested. `target_accept_prob`
(numpyro's own default, `0.8`) only affects `NUTS`'s step-size adaptation,
used when the composed model has factor sites -- a plugin, or a declared
`prior` on a default-prior-covered parameter; a pure-sample model ignores
`num_warmup`/`target_accept_prob` entirely and draws through `Predictive`
instead. `priors` (the plugin registry) is optional -- `priors: {}` is a
`PriorSampler` run with no plugins, sampling the declared/default priors
alone. `potential_rescalings` and the rest of `stopping_criteria` behave
exactly as they do for any other `generator_type`; `PriorSampler` doesn't
change their meaning.

## Diagnostic plot and per-batch logging, not a pass/fail gate

A factor-bearing model's MCMC draw having a high divergence rate or low
effective sample size doesn't automatically mean something is broken -- how
rough a chain looks depends partly on how wide the prior needs to be for
the actual science (a young, single-age stellar population needs a narrow
mass-to-light prior; a mixed old-and-young population needs a much wider
one). But it doesn't automatically mean nothing is broken either: the same
numbers can just as well flag real sampling trouble -- a poorly-adapted
step size, a pathological posterior shape -- even when the prior itself is
exactly as wide as the science requires. Divergence/ESS/R-hat alone can't
distinguish "this prior is wide" from "this chain has a real problem," and
no fixed threshold on them can either, so TNT doesn't try to automate that
call.

Two separate things report these diagnostics, at two different times:

- A one-off **pilot** figure. Whenever a resolved configuration uses
  `generator_type: "PriorSampler"`, `ModelIterator.from_configuration`
  draws once -- at the same `generator_settings.num_samples`/`num_warmup`/
  `target_accept_prob` the real run will use, but with its own,
  independent key (`PRNGKey(seed)`, not `PriorSampler`'s own
  `fold_in(PRNGKey(seed), n_iterations())`) -- and saves a diagnostic
  figure (pairwise scatter colored by divergence, each parameter's
  marginal histogram, an ESS/split-R-hat table, and per-parameter trace
  plots) to `io_settings.output_directory/prior_diagnostics.png`. A useful
  pre-flight sanity check on the composed model as a whole, but -- because
  of that independent key -- never literally the same draw as any real
  iteration's own batch, not even the first.
- **Per-batch logging.** Every real batch `PriorSampler` actually proposes
  from logs its own divergence count and worst ESS/split-R-hat across
  every parameter, at `INFO` level, tagged with the iteration it was drawn
  for.

Neither raises or blocks on what it finds; review the figure and the
logged summaries yourself and decide whether `PriorSampler`'s output looks
usable before relying on it.

`Prior.sample_with_diagnostics(key, num_samples, num_warmup,
target_accept_prob)` is the underlying call behind both -- the same return
as `sample`, plus a `PriorDiagnostics` that's always populated. A pure
`Predictive` draw is exact and independent by construction, so its
ESS/split-R-hat there are a sanity check on that independence, not a
convergence diagnostic; `tnt.priors_diagnostic_plot.plot_prior_diagnostics`
builds the pilot figure from this, and `PriorSampler` logs from it
directly, either way.
