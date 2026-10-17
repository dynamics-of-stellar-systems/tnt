# Orbit integration follow-ups

Updated: 2026-10-17.

## Agreed scope and present state

- Use Diffrax `Dopri8`; numerical equivalence to DYNAMITE's DOP853 is a
  convergence question, not an assumption of identical adaptive steps.
- Default duration is 200 DYNAMITE reference periods. The refined
  tube-boundary clock is shared by populations on the same energy shell;
  it is not each orbit's measured recurrence period.
- Support `n_dither=1` for this first implementation. Larger bundles are
  explicitly rejected until their sampling and averaging are implemented.
- Retain per-bundle contributions and diagnostics. Save selected trajectories
  separately, referenced by stable bundle/member IDs. Compute contributions
  for each requested common mass scale while temporary samples are available.
- Programmatic integration and reduction were implemented on
  `codex/orbit-integration`, through commit `a685064`. Analytic tests and a
  triaxial-potential reduction test pass. A subsequent controlled integrator
  comparison now covers 70 paired runs and 120 force probes. The user chose
  `1e-9` as the configurable YAML and Python integration default on 2026-10-16,
  following the benchmark's initial explicit working recommendation of `1e-8`.
  Universal production accuracy and distribution equivalence are not certified.
- The user authorized the comparison, long-time statistics, and presentation
  site, with incremental commits, before continuing the scientific features.
  Comparison code and results are on `codex/orbit-integration-comparison` in
  the separate comparison repository. See its `orbit-integration/REPORT.md`.

## Completed benchmark and remaining calibration

The benchmark below has run. Keep its reference-code harness, resolved inputs,
results, and reproduction
instructions in the existing separate
`dynamics-of-stellar-systems/tnt-dynamite-comparison` repository. Do not make
DYNAMITE a TNT runtime or unit-test dependency.

- [x] Pin the TNT and DYNAMITE revisions, numerical-library versions, float64
  initialization, potential parameters and constants, units, initial
  conditions, reference periods, physical duration, and sampling settings.
  Use identical exported launch coordinates in both integrations rather than
  independently generating two nearly identical start spaces.
- [x] Compare accelerations first at representative points, especially near
  the centre. DYNAMITE's interpolated force evaluation and TNT's Galax force
  evaluation can differ independently of their integrators. Document those
  differences; do not attribute every trajectory discrepancy to Dopri8.
- [x] Start with a small set of regular spherical/axisymmetric orbits, then
  triaxial box/tube starts and eccentric, central-mass-dominated orbits.
  Include inner and outer shells and difficult cases rather than only median
  examples. Compare short-time phase coordinates where meaningful.
- [x] Match physical durations using the agreed clocks. For sample-by-sample
  comparisons, explicitly align output times or use a controlled diagnostic
  sampling schedule. Matching random seeds across Fortran and JAX does not
  imply matching sampling offsets.
- [x] Compare 200-period statistics: intrinsic spatial occupancy, velocity
  moments, and simple common velocity histograms. These comparison reducers
  can be small diagnostic tools; a complete observational projection layer
  is not a prerequisite. Long-time pointwise agreement is not an appropriate
  pass criterion for chaotic orbits.
- [x] Vary solver tolerance, output sample count, and integration duration
  independently. Test both implementations against tighter reference runs;
  the same numerical tolerance value need not mean the same achieved error.
  Distinguish integration error from finite-time/sampling variation.
- [ ] Report sampled/endpoint energy drift with an explicitly common
  normalization, failed or retried solves, adaptive step counts, runtime,
  and memory. Separate JAX compilation time from repeated execution time.
  Exclude or clearly flag any DYNAMITE result that substitutes another orbit
  after a failure; it is not a valid reference for the requested launch.
  Energy, solver attempts/counts, and timing are recorded; peak-memory
  profiling remains deferred. Fortran native CPU and containerized JAX wall
  timings are not a fair speed ranking.
- [ ] Agree measurable acceptance thresholds from convergence results and
  the intended observable accuracy. Record outliers and limitations as well
  as typical agreement. Calibrate TNT's defaults and any energy acceptance
  policy from this evidence rather than inheriting a nominal tolerance.

Exit condition: a reproducible small benchmark identifies the sources of
disagreement and supports explicit production settings, or exposes specific
numerical issues to resolve before building more functionality.

This condition is met by identifying the following numerical follow-ups:

- [ ] **Gaussian force accuracy:** define and validate a production policy for
  large radius-to-width ratios and extreme flattening. The agreed next step
  is to investigate and configure Gaussian quadrature using TNT's existing
  direct static evaluator. In the matched native
  parameter audit, fixed quadrature order 50 differs from DYNAMITE direct
  acceleration by up to `3.65e-3`; orders 200/400 reduce the largest measured
  difference to about `5.84e-8`. The order-200 variant is a comparison control,
  not an implemented TNT production policy or a universal recommendation.
- [ ] **Direct versus interpolated forces:** after establishing a reliable
  direct evaluator, compare direct force evaluation with a precomputed force
  grid and interpolation, as used in DYNAMITE. Measure force accuracy,
  effects on integrated orbits/retained contributions, and runtime. Separate
  grid construction/loading and JAX compilation costs from repeated force
  evaluation and orbit-integration time. This concerns the force-evaluation
  method, not selection of a different orbit integrator; a TNT interpolation
  grid has not yet been chosen for implementation.
- [x] **Working default accuracy:** set both packaged configuration and
  `OrbitIntegrationSettings` defaults to `1e-9`, retaining explicit overrides.
  The largest normalized energy drift over twelve 200-period launches falls
  from `1.52e-5` at `1e-8` to `6.76e-7` at `1e-9`, for about 27% more total
  adaptive steps. This choice is based on the benchmark, while
  [Diffrax's guidance](https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/#stiff-problems)
  supports using float64 for tolerances around `1e-8` or tighter. The archived
  report/site retain their original `1e-8` working recommendation.
- [ ] **Acceptance and float32 policy:** validate broader production cases
  before establishing energy-rejection thresholds. Float32 currently requires
  users to choose a looser orbit tolerance explicitly; there is no automatic
  adjustment or precision/tolerance compatibility guard. Initial-condition
  searches also use independent fixed tolerances, so review and validate the
  full workflow before claiming float32 orbit-library support. Keep any future
  compatibility enforcement explicit rather than silently changing accuracy.
- [ ] **Time and sampling convergence:** diagnose sensitive orbits using the
  intended retained quantities. Some raw spatial split-half differences are
  large in both codes; outer-box discrepancies do not uniformly disappear
  at 400 periods. Spatial histograms in this benchmark precede symmetry
  operations and counter-rotating populations; do not treat them as final
  symmetrized library error estimates. Formal chaos diagnosis was not done.

The comparison checkout for this chat is
`/Users/maindl/.codex/visualizations/2026/10/15/01a13ee6-91e7-7b92-b380-3b5de052396c/tnt-dynamite-comparison`.
Its reproduction scripts, report, full-precision summary, compact NPZ traces,
and standalone scientific figures are separate from TNT's runtime/tests.

## Subsequent implementation milestones

- [ ] **Intrinsic contributions and counter-rotating populations.** Establish
  the density-grid contribution contract, normalization, stable IDs, and
  population bookkeeping. For supported static potentials, time reversal
  changes `(q, v)` to `(q, -v)`: the spatial occupancy is shared while odd
  velocity moments change sign. Define mirrored contributions from the
  temporary samples and test the symmetry, with separate selectable weights
  where the model requires independent rotation populations. Do not blindly
  duplicate box populations or assume every observational transform is a
  symmetric velocity-bin reversal.
- [ ] **Observational projections, one data family at a time.** Implement
  viewing-frame transforms, spatial apertures and point-spread-function
  treatment, then line-of-sight velocity distributions and the configured
  observational quantities. Add proper-motion projections where required.
  Preserve each aperture's fraction of total orbit time; do not independently
  normalize every occupied aperture to one. Verify projections and mirrored
  populations together on small controlled examples.
- [ ] **Model execution and weight fitting.** Connect
  `Potential.generate_orbit_library`, kinematics `design_matrix`,
  `ModelIterator`, and non-negative least-squares fitting to the retained
  contribution contract. Replace the old post-binning mass-rescaling path
  with contributions computed at every requested common mass scale. Validate
  a small end-to-end model before expanding the library size.
- [ ] **Larger dither bundles.** Implement launch-space sampling, member
  averaging, shared shell clocks, member IDs, failure semantics, and optional
  member trajectories; compare the resulting bundle normalization with
  DYNAMITE. Keep `n_dither>1` rejected until this is complete.
- [ ] **Production execution.** Benchmark batching and memory on realistic
  libraries. Design persistence/resume and cache invalidation explicitly if
  needed; the selected NPZ trajectory format is diagnostic output, not a
  checkpoint protocol. Specify failure/retry behavior without silently
  replacing failed orbits.
- [ ] **Optional integrators.** Benchmark Tsit5 as a first alternative after
  Dopri8 has a baseline. Consider fixed-step velocity Verlet separately for
  supported separable Hamiltonians and long-time statistics. Add user-facing
  solver selection only when accuracy/cost evidence justifies it.

## Presentation site

Published owner-private presentation:
<https://tnt-orbit-integration-validation.funbass128.chatgpt.site>.
Sign in with the owning account to view it. The site includes interactive
orbit/run/projection selection, energy/convergence plots, distribution
diagnostics, section-based presentation navigation, the report, JSON data,
and five standalone scientific figures bundled as a PDF.

- [x] Build a scientific results site from versioned, offline-generated data
  and figures. Keep numerical execution in the reproducible comparison
  workflow; the site presents its results.
- [x] Show the comparison setup and provenance, representative orbit views,
  energy-drift and convergence plots, occupancy/velocity-statistic
  differences, failures/outliers, and performance with compilation separated.
- [x] Clearly distinguish implemented features, measured results, open
  questions, and planned follow-ups. Include reproduction links and explain
  why long-time chaotic trajectories can separate while their useful
  distributions agree. Do not present planned checks as completed validation.

The public API and current limitations are documented in
[`docs/source/orbit_library.md`](../docs/source/orbit_library.md).
