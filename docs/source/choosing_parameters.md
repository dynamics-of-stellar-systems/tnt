# Choosing model evaluation points

Orbit models are expensive. TNT can choose a small set of parameter
combinations before you run them. `PriorSampler` draws locations from your
prior; `PriorSupportPoints` arranges points to represent it more deliberately.
Both include any prior plugins you supply.

## One prior, two ways to explore it

Take a spherical toy galaxy with stars and a dark halo. Only **`dm.m`**, the
halo mass parameter, and **`stars.ml`**, the stellar mass-to-light ratio, are
free. The light profile, halo scale radius and all other parameters are fixed.

Here `dm.m` has a log-uniform prior from 3 × 10¹⁰ to 3 × 10¹¹ solar masses,
and `stars.ml` a uniform prior from 1 to 9 solar mass-to-light units.
A plugin favours a **total mass inside 10 kpc of
(4 ± 0.5) × 10¹⁰ solar masses**, using a Gaussian preference.
More stellar mass leaves less room for dark matter: this gives the sloping band.

These diagnostic plots are generated during the documentation build.
Blue contours show the combined prior; orange diamonds mark the **24 proposed
model locations**. The other panels show marginal distributions and sampling
checks. Click either figure to enlarge it.

## PriorSampler: draw from the distribution

```{figure} _generated/prior_generators/prior_sampler.png
:width: 100%
:alt: PriorSampler diagnostic showing 24 random proposals over the correlated dark-halo-mass and stellar-mass-to-light prior.

Twenty-four sampled locations. Some lie close together; other parts of the
band receive fewer points.
```

Use `PriorSampler` when you want draws from your specified prior. With a small
budget, coverage depends on the draw. Later batches use fresh draws from the pool.

## PriorSupportPoints: spread a small evaluation budget

```{figure} _generated/prior_generators/prior_support_points.png
:width: 100%
:alt: PriorSupportPoints diagnostic showing 24 optimized proposals representing the same correlated prior, with optimization histories below.

Twenty-four support points for the same prior. The lower panels show their
placement optimization, not an orbit-model fit.
```

`PriorSupportPoints` moves points to represent a larger reference sample.
It is useful when models are costly and you want deliberate coverage from
few evaluations. Later iterations add points while keeping existing locations
fixed.

Support points are **not random prior draws**: a small design can miss some
width of a narrow band, and minimum spacing is not guaranteed. Neither generator
estimates a posterior at this stage.

See [Priors](priors.md) and [Support points](support_points.md) for settings.
The {download}`complete example <../_ext/prior_generator_examples.py>` builds
both figures with the same model and enclosed-mass plugin.
