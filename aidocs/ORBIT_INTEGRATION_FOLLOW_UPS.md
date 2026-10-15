# Orbit integration follow-ups

Updated: 2026-10-15.

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
- Programmatic integration and reduction are implemented on
  `codex/orbit-integration`, through commit `a685064`. Analytic tests and a
  triaxial-potential reduction test pass. Production accuracy defaults have
  not been calibrated against DYNAMITE. The earlier DYNAMITE comparison
  validates start-space sampling only.
- The user chose to wait for comparison results before building a presentation
  site. No site creation or publication is part of this documentation step.

## Recommended next milestone: integrator comparison

This is the recommended work order, not a claim that the comparison has run.
Keep its reference-code harness, resolved inputs, results, and reproduction
instructions in the existing separate
`dynamics-of-stellar-systems/tnt-dynamite-comparison` repository. Do not make
DYNAMITE a TNT runtime or unit-test dependency.

- [ ] Pin the TNT and DYNAMITE revisions, numerical-library versions, float64
  initialization, potential parameters and constants, units, initial
  conditions, reference periods, physical duration, and sampling settings.
  Use identical exported launch coordinates in both integrations rather than
  independently generating two nearly identical start spaces.
- [ ] Compare accelerations first at representative points, especially near
  the centre. DYNAMITE's interpolated force evaluation and TNT's Galax force
  evaluation can differ independently of their integrators. Document those
  differences; do not attribute every trajectory discrepancy to Dopri8.
- [ ] Start with a small set of regular spherical/axisymmetric orbits, then
  triaxial box/tube starts and eccentric, central-mass-dominated orbits.
  Include inner and outer shells and difficult cases rather than only median
  examples. Compare short-time phase coordinates where meaningful.
- [ ] Match physical durations using the agreed clocks. For sample-by-sample
  comparisons, explicitly align output times or use a controlled diagnostic
  sampling schedule. Matching random seeds across Fortran and JAX does not
  imply matching sampling offsets.
- [ ] Compare 200-period statistics: intrinsic spatial occupancy, velocity
  moments, and simple common velocity histograms. These comparison reducers
  can be small diagnostic tools; a complete observational projection layer
  is not a prerequisite. Long-time pointwise agreement is not an appropriate
  pass criterion for chaotic orbits.
- [ ] Vary solver tolerance, output sample count, and integration duration
  independently. Test both implementations against tighter reference runs;
  the same numerical tolerance value need not mean the same achieved error.
  Distinguish integration error from finite-time/sampling variation.
- [ ] Report sampled/endpoint energy drift with an explicitly common
  normalization, failed or retried solves, adaptive step counts, runtime,
  and memory. Separate JAX compilation time from repeated execution time.
  Exclude or clearly flag any DYNAMITE result that substitutes another orbit
  after a failure; it is not a valid reference for the requested launch.
- [ ] Agree measurable acceptance thresholds from convergence results and
  the intended observable accuracy. Record outliers and limitations as well
  as typical agreement. Calibrate TNT's defaults and any energy acceptance
  policy from this evidence rather than inheriting a nominal tolerance.

Exit condition: a reproducible small benchmark identifies the sources of
disagreement and supports explicit production settings, or exposes specific
numerical issues to resolve before building more functionality.

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

## Presentation site after the first comparison

- [ ] Build a scientific results site from versioned, offline-generated data
  and figures. Keep numerical execution in the reproducible comparison
  workflow; the site presents its results.
- [ ] Show the comparison setup and provenance, representative orbit views,
  energy-drift and convergence plots, occupancy/velocity-statistic
  differences, failures/outliers, and performance with compilation separated.
- [ ] Clearly distinguish implemented features, measured results, open
  questions, and planned follow-ups. Include reproduction links and explain
  why long-time chaotic trajectories can separate while their useful
  distributions agree. Do not present planned checks as completed validation.

The public API and current limitations are documented in
[`docs/source/orbit_library.md`](../docs/source/orbit_library.md).
