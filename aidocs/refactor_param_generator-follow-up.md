# Parameter-generator refactor and remaining follow-up

Tracked in [GitHub issue #88](https://github.com/dynamics-of-stellar-systems/tnt/issues/88).

Completed in PR #90 (the scope agreed after the support-point PR):

1. Implement duplicate filtering in AbstractParameterGenerator: remove repeated
   proposals within a batch and locations already present in AllModels before
   applying the proposal cap. Use exact equality after compatible unit
   conversion; do not introduce an approximate separation threshold.
2. Reorganize parameter-generator outputs under
   `<output_directory>/parameter_generator/`, including prior diagnostics.
   Use cumulative iteration filenames for support-point diagnostics; keep
   renderers accept explicit paths and already-generated data.
3. Introduce a `tnt.parameter_generator` package, separating the abstract
   interface, concrete generators, construction/registration, and supporting
   output/diagnostic concerns according to repository conventions.
4. Keep ordinary and diagnostic proposal interfaces consistent after filtering.
   Plot returned points while labelling optimization statistics as describing
   the optimized design before filtering. Both prior-based generators own
   optional actual-batch plots through plot_diagnostics; SinglePoint needs none.
   Label model-search iterations in filenames and titles; keep hist/scatter
   backgrounds unchanged.

Still deferred from this refactor:

- Prior sample caching is a separate functional change. A reusable sample-store
  service should be shared by consumers of `Prior`, while `Prior` remains
  independent of filesystem/output management. Store direct constrained draws
  in original declared coordinates and units, without scaling. Define a
  prior-specific identity covering declarations, plugin contents and relevant
  physical context, invalidation, resume behavior, pool selection and exhaustion
  before implementing reuse. Existing model-resume compatibility deliberately
  excludes prior/search declarations and cannot establish cache validity.
  Consider `cache_prior_samples` defaulting to true, and account for
  future pilot requirements. No in-memory or persistent prior cache is
  introduced here.
- Plot appearance was deferred from the original refactor and is now implemented
  on the diagnostic-plot branch: probability-mass contours, distribution-based
  log axes, purple divergence markers, fixed/new support marginal histograms,
  table and boxed legend in the corner-grid whitespace, horizontal traces on
  the right, and stacked optimization panels with readable solver status.
- Dtype-aware validation of configured tolerances belongs to separate numerical
  policy work. This plotting change does not introduce precision-dependent
  tolerance defaults, validation thresholds or a float32/float64 comparison.

Preserve sequential support-point augmentation: optimize
`max_new_mods_per_iter` new points with existing `AllModels` locations fixed,
then apply shared duplicate filtering without refill. Existing locations must
still be unique; filtering proposals does not repair old duplicate table rows.

The parameter_grid_relative_tolerance configuration field currently has no
runtime comparison consumer. Its presence does not define duplicate identity;
retain it pending clarification of its intended role.

A broader pre-flight execution pilot, including representative orbit integration
and weight-solving peak-memory/concurrency measurements for resource allocation,
is separate follow-up work. Current plots describe actual generation batches;
no separate pre-flight sampling or execution profiling is added.
