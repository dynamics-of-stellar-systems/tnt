# Support points from the prior

`PriorSupportPoints` augments evaluation locations using the existing run
prior. It draws a cheap reference cloud, transforms it to NumPyro unconstrained
coordinates, optimizes support points, and maps the design back to the original
parameters and declared units. Reference draws and optimization do not run
expensive models, likelihoods, posterior estimation, or adaptive acquisition.

```yaml
parameter_space_settings:
  generator_type: PriorSupportPoints
  generator_settings:
    num_samples: 4000
    seed: 42
    num_warmup: 800
    target_accept_prob: 0.9
    max_iterations: 1500
    tolerance: 1.0e-6
    chunk_size: 1024
    distance_epsilon: 1.0e-8
  stopping_criteria:
    max_new_mods_per_iter: 24
    n_new_iter: 1
```

This override fragment still requires the usual potential and other run
settings. `max_new_mods_per_iter` requests that many new points for optimization
on every call. Shared exact-duplicate filtering can reduce the returned count.
Existing downstream potential rescaling still applies; disable it to evaluate exactly this design.
The packaged defaults request 10 new points per call, 1,000 samples and warmup steps,
500 optimizer steps, tolerance `1e-6`, block size 1,024, and epsilon `1e-8`.
Algorithm controls belong to `generator_settings`, not global `numerics_settings`.

## Coordinates and feasibility

The generator inherits directly from `AbstractParameterGenerator` and receives
the already-built `Prior`. It neither subclasses nor constructs `PriorSampler`.
`Prior.unconstrain` and `Prior.constrain` replay conditional distribution
supports: `q` can depend on `p`, and `u` on both. Transforms extracted from one
fixed trace must not be reused across different positions.

Each transformed scalar coordinate is centered and scaled using the reference
mean and population standard deviation. Distances are Euclidean in these
standardized unconstrained coordinates. This is not whitening or a general
prior-CDF transport. The support transform is distinct from the prior's density
shape. Changing coordinates changes the distance metric and therefore the
finite design; distribution-selected plot axes do not change this metric.
No Jacobian weighting is needed for the transformed empirical reference cloud.

Fixed parameters are restored directly. Constant sampled coordinates, including
Delta point masses, stay constant and contribute no distance. Discrete or
vector-valued sample sites are rejected. Exact boundary draws mapping to
nonfinite coordinates raise rather than being clipped. Heavy-tailed reference
scales may be unstable; energy distance requires a finite first moment in the
chosen coordinates.

The sampler draws `num_samples + max_new_mods_per_iter` values on every call. Infeasible or
nonfinite draws are filtered, then the last `m` feasible draws initialize the
design and the rest form its reference cloud. This avoids exact reference
coincidences at initialization. With MCMC the draws may remain correlated.
Sampling diagnostics describe the original batch; the effective reference size
and discarded count are reported. Fewer than `m+1` feasible draws raises.

No reference samples or scaling are cached in memory or persisted. With
unchanged seed and settings, each call regenerates the same cloud and scaling,
but still executes sampling and any required MCMC warmup again. Prior-cache
design and storage remain deferred.

## Successive iterations

All existing `AllModels` rows supply fixed support locations, regardless of
their iteration or fit success. Sample-site columns are converted to declared
units and validated for finite values, exact uniqueness, and prior/geometry
feasibility. Missing columns, duplicates, or invalid locations raise errors.
No approximate matching tolerance is introduced.

Existing locations use the same prior transforms and current reference scaling
as the new points. With `k` existing and `m` new locations, optimize the energy
distance of the combined `k+m` point set while moving only the new coordinates.
Repulsion includes new/new and new/existing pairs, normalized by the total
point count. Initial/final objectives and spacing diagnostics describe the full
combined design. Only new points are returned; old locations are never moved
or silently deduplicated. Diagnostics include `existing_support_points` and
`new_support_points` (the number optimized). The sole public method
`generate_parameters` returns a `ParameterBatch`, with finalized `parameters`
and `diagnostics`. The ABC applies duplicate filtering in complete declared
coordinates before returning. `batch.diagnostics["proposals"]` reports proposed,
duplicate-removed, capped and returned counts. No replacements are generated. For an all-fixed or point-mass prior,
the first call returns one unique location, and later calls return none once
it exists. With no sampled sites, generation constructs that single parameter
set directly; it does not manufacture repeated proposals to match the requested
optimization count. Optimizer statistics and histories describe the optimized design
before filtering; overlays show only returned points. Existing table rows are
never silently deduplicated. Sequential batches need not reproduce a
joint optimization of the same total number of points.

Support transforms encode distribution bounds, but arbitrary hard plugins and
additional physical or numerical geometry restrictions can remain. Intermediate
positions are not physically constrained. The complete final mapped-back design
is checked against conditional supports, finite sample/factor log probabilities,
and `Potential.build_with_validity` when potential context is available. Any
invalid final point causes an error with no candidates returned. There is no
projection, random replacement, or silent filtering of infeasible final points.

## Objective and Optimistix solver

For a first iteration the scaled-coordinate objective is

\[
J(z)=\frac{2}{mN}\sum_{i=1}^m\sum_{j=1}^N\|z_i-x_j\|_2
 -\frac{1}{m^2}\sum_{i=1}^m\sum_{l=1}^m\|z_i-z_l\|_2.
\]

The reference-only energy-distance constant is omitted. Objectives cannot be
compared across different reference clouds or metrics without accounting for
that constant. Separation is encouraged, not imposed: no minimum distance or
variance preservation is guaranteed. Constant priors can give coincident points.

The blocked sp.ccp map is Eq. (22) of
[Mak & Joseph (2018), Section 3](https://arxiv.org/html/1609.01811v7#S3).
It is iterated using **`optimistix.fixed_point` with built-in
`optimistix.FixedPointIteration`**, not a custom solver or loop. The solver uses
`rtol=0`, `atol=tolerance`, and the norm
`max_i ||new_z_i - old_z_i||_2`. Optimistix declares convergence when this absolute
maximum displacement is **strictly below** tolerance.

The exact majorization map theoretically decreases the objective under the
paper's assumptions. `distance_epsilon` protects inverse distances at
coincidences without adding a separation constraint. Floating-point arithmetic
and this safeguard mean descent is not numerically enforced. Inspect initial
and final objectives; global optimality is not promised.

The numerical API accepts optional `fixed_points` in scaled coordinates.
Its output contains only new points; existing coordinates never enter the
solver's changing state. With no existing points, it reproduces ordinary
support-point generation.

`optimize_support_points_jax` returns a native
`optimistix.Solution`: `value` is the final design, `aux` the final maximum
displacement, and `result`/`stats` report the solver outcome and steps.
It supports JAX `jit` and `vmap`. The eager `optimize_support_points` wrapper
validates inputs and returns Python diagnostics: initial/final objective,
convergence flag/status, iterations, `final_maximum_displacement`,
`minimum_pairwise_distance`, and mean nearest-neighbour spacing. Both spacing
summaries are NaN when fewer than two points exist. Status is `converged`,
`iteration_limit`, `nonfinite`, or `solver_failure`; final physical validation
is separate.
Generation collects per-step history when plotting is enabled or history is
explicitly requested. `energy_objective_jax` is
traceable; `energy_objective` preserves its Python-float return for analysis.

Reference workspace is `O(m * chunk_size * d)`, with `O(Nd)` stored samples and
`O(m^2 d)` support-pair workspace. No reference `N x N` matrix is constructed.

## Diagnostic plots

With `generator_settings.plot_diagnostics: true` (the packaged default), each
support-point generation call saves its actual sampled batch, returned/fixed
point overlays and optimization histories under
`<output_directory>/parameter_generator/`. Filenames and titles identify the
cumulative model-search iteration, for example
`support_points_diagnostics_iteration_0000.png`. The history axes count optimizer
steps within that round. Neither plotting nor rendering performs another sample
or optimization. The background retains all original sampling draws (including
initialization draws and any subsequently excluded infeasible draws), in their
original ordering with their actual divergence flags and sampling statistics.

`generate_parameters(all_models)` returns a `ParameterBatch` from the same call.
Use `batch.parameters` for candidates and `batch.diagnostics` for diagnostic data. When plotting is enabled, or `record_history=True` is requested,
`info` also contains the actual `sample_batch`, original-coordinate fixed points,
and parameter units needed to render later:

```python
from tnt.parameter_generator.diagnostics.support_points import (
    plot_prior_support_points_diagnostics,
)

# Set plot_diagnostics: false to collect data without automatically writing.
batch = generator.generate_parameters(all_models, record_history=True)
plot_prior_support_points_diagnostics(
    batch.parameters, batch.diagnostics, "support_diagnostics.png"
)
```

The explicit destination parent must exist. The renderer only draws previously
generated data. It adds small orange diamonds for returned points, identical open
slate circles for existing fixed points, and objective and minimum pairwise-distance panels with solver status. Optimizer statistics describe the design before
shared duplicate filtering; overlays show only returned candidates.

All earlier model-search iterations are pooled into one fixed-point group;
only the current new batch has a different style. Marginals overlay separate
fixed (dashed slate) and new (solid orange) histogram outlines on the prior
histogram. All three use common bins and are independently normalized to unit
total mass, so sample counts do not determine their relative heights. Paired
panels retain the prior 68/95/99.7% contours and muted divergence markers. See
[Prior diagnostic display conventions](priors.md#diagnostic-display-conventions)
for axis selection and contour estimation. These visual changes do not alter
the scaled unconstrained optimization metric.

The optimization panels are stacked below the corner grid, extend from the
corner grid's left edge to the trace plots' right edge, and share an
optimizer-iteration axis. Their heading reports the readable
solver status and step count; the figure title identifies the generator and
model-search iteration.

The lower optimization panel shows the minimum Euclidean distance over all
distinct pairs in the combined fixed-plus-new design, in normalized
unconstrained coordinates. It includes initialization and every solver step;
for fewer than two points it is undefined (NaN). Fixed-fixed pairs are included,
so an existing close pair can set a ceiling on this statistic.
The upper panel shows the combined energy objective on a linear y-axis.
A flattening objective alone does not certify the displacement stopping criterion.

Objective and minimum pairwise-distance histories include initialization
(optimizer step zero). Ordered JAX callbacks record actual solver steps,
retaining two scalars per step rather than point trajectories. The maximum
per-point displacement still determines convergence; only its final value is
retained as `final_maximum_displacement`. Recording adds objective
evaluations and synchronization. With plotting disabled and no explicit history
request, no callbacks or plot-data retention are added. Returned plot data is
owned by the caller, not cached across generation calls. Independent of plotting,
`info` includes sampling/optimization diagnostics, site ordering, reference
centers/scales, constant dimensions, effective reference size and discarded count.

## Interpreting a finite design

Support points minimize distributional discrepancy in the chosen coordinates;
they do not preserve every marginal variance or maximize spacing. A small
design can concentrate near a narrow prior degeneracy while underrepresenting
its transverse width. Increasing the number of points can recover more width,
but there is no universal variance ordering relative to random prior draws.

For validation, compare a design with an equally sized random prior sample
using an independent reference cloud and the same coordinate transform and
scaling. The reference-only term cancels in this comparison. Report both
energy discrepancy and spacing, and inspect important physical projections;
improvement in discrepancy need not improve minimum physical spacing.
The focused tests exercise uniform and correlated distributions, real
conditional priors, plugin feasibility and sequential augmentation through the
model iterator. A deterministic narrow Gaussian example checks that a small
design can lose transverse width while a larger design recovers it.

Prior representation does not guarantee posterior accuracy or provide a
posterior error bound. Subsequent inference and adaptive evaluation are
separate work.
