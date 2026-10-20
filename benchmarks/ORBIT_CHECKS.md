# Representative direct-force orbit checks

This follow-up checks the configured Gaussian quadrature order 800 in full
200-reference-period orbit integrations. It uses the production Dopri8
integrator, float64, and all 50,000 configured output samples. It does not
implement interpolation or run a new DYNAMITE comparison.

## Findings and recommendation

- The configured order 800 and accuracy `1e-9` remain working development
  choices for these four launches. Maximum normalized sampled/endpoint energy
  drift is `6.43e-8`; this is distinct from the solver's state tolerance.
- Doubling quadrature order leaves the first three cases' radial, spatial,
  and velocity histogram counts unchanged. The outer triaxial box retains
  measurable long-window sensitivity despite tiny first-period differences.
- Tightening the solver to `1e-11` reduces energy drift by about 31-75 times,
  for about 1.50-1.75 times the warm CPU cost here. Final observable accuracy
  targets and energy-rejection thresholds remain open.
- Bounded energy diagnostics remove the large all-state quadrature allocation:
  the complete matrix now fits the original 4 GB VM allocation. The original
  8 GB trial exposed host memory pressure rather than establishing production
  memory requirements.

Proceed with the planned direct-versus-grid experiment using this bounded
static direct evaluator as the reference. Keep time/sampling convergence of
the outer box as an explicit follow-up before assigning final observable
error bounds. These four launches complement the independent force audit;
they do not certify an entire parameter domain or production library.

## Reproduce

The harness reuses the existing comparison repository's physical-model
builder, exported launches and DYNAMITE shell clocks, and diagnostic bins.
It requires an explicit checkout of
`dynamics-of-stellar-systems/tnt-dynamite-comparison`; the result records its
revision and input hashes. Mount that checkout read-only, replacing
`/absolute/path/to/tnt-dynamite-comparison` below:

```shell
docker --context colima compose run --rm --volume /absolute/path/to/tnt-dynamite-comparison:/comparison:ro dev python benchmarks/orbit_checks.py --comparison-checkout /comparison --work-dir /workspace/benchmarks/work/orbit-checks-new --output /workspace/benchmarks/results/orbit_checks-new.json
```

Choose a new work directory and output filename; earlier results are never
overwritten. Run scientific processes sequentially. Source edits are visible
through the normal workspace mount; no image rebuild is needed.

`--case`, `--variant`, and `--warm-repeats` select explicit smaller checks.
Every selection must include the `working` variant. The default matrix is:

| Variant | Gaussian order | Normalized solver accuracy |
| --- | --- | --- |
| Working | 800 | `1e-9` |
| Higher order | 1600 | `1e-9` |
| Tighter solver | 800 | `1e-11` |

All variants use identical physical parameters, launches, random keys, output
times, and 200-period durations. The four launches are the strongly triaxial
tube and box on shell E3, and oblate and strongly triaxial outer boxes on E5.
These include the central passages and sensitive outer-box cases identified
in the earlier comparison; they do not cover an entire production library.
Each launch represents one undithered member (`n_dither=1`).
The higher-order run is a convergence control using the same quadrature
family, not an independent exact solution.

Both galaxy models have minimum Gaussian axis ratio about 0.54 and minimum
Gaussian width about 0.0957 kpc. They therefore complement the much thinner
and more distant probes in the independent force audit, rather than repeating
that audit's extreme shape/radius envelope. This checks complete integrations
in representative galaxy models; it does not establish universal force or
orbit accuracy.

## Quantities and timing

The portable JSON retains all-sample radial, Cartesian spatial, and velocity
histograms, velocity second moments, and mean angular momentum. Coordinates
are normalized by the established comparison's launch length `L` and speed
`V = 2*pi*L/reference_period`. Histograms are raw intrinsic diagnostics,
before spatial symmetries, counter-rotating populations, or observational
projection. Their differences are not final observable errors.
The interior bin widths are `L/12` radially, `0.375*L` per Cartesian spatial
axis, and `0.1*V` per velocity component, with overflow bins. Zero total
variation records identical counts at this diagnostic resolution; the
underlying trajectories and continuous moments can still differ.

Histogram total variation is half the summed absolute probability difference,
between zero and one. Velocity components are compared separately. Split-half
differences compare the first and second halves of the same trajectory;
they show finite-time variation rather than a formal uncertainty estimate.
First-period position/velocity differences are normalized component-wise root
mean square differences. Long-time pointwise agreement is not used as an
acceptance criterion for sensitive orbits.

Energy diagnostics use `abs(E-E_launch)/V**2`, evaluated at every saved state
and the exact endpoint. They do not measure every internal adaptive stage.
The harness rejects failed or non-finite solves; it does not retry or replace
launches. Solver step counts include both accepted and rejected steps.

Timing synchronizes the entire production solve, including sampling and
energy diagnostics, and excludes later statistics and file I/O. Each row
records one first call and three warm repeats. Later first calls may reuse
compiled kernels. Model building and numerical-control setup are recorded
separately. Linux process peak resident memory is cumulative across the
matrix, not a per-orbit peak or a device-allocation measurement. Results also
record the VM's visible memory capacity.

Raw selected trajectories and individual row records go in the ignored
`benchmarks/work/` directory. The final JSON retains trajectory hashes and
portable statistics; it can be inspected without loading the raw trajectories.

## Measured results

Run date: 2026-10-17. The complete matrix contains 12 variants and 48
successful solves (one first call and three warm repeats per variant).
Full-precision statistics, physical parameters, diagnostics, timings and
provenance are retained in [`orbit_checks.json`](results/orbit_checks.json).

All triaxial rows below use the strongly triaxial galaxy model. Energy
errors are `abs(E-E_launch)/V**2`; accepted/rejected counts sum to the total
adaptive step count. First calls include any compilation required at that
point; subsequent rows can reuse compiled kernels.

| Launch | Variant | Maximum energy error | Accepted / rejected steps | First call (s) | Warm median [range] (s) |
| --- | --- | --- | --- | --- | --- |
| Oblate box E5 | Working | 2.884e-08 | 14,712 / 6,506 | 20.04 | 13.83 [13.41-14.94] |
| Oblate box E5 | Order 1600 | 2.885e-08 | 14,712 / 6,506 | 36.43 | 32.84 [32.80-34.42] |
| Oblate box E5 | Accuracy 1e-11 | 8.824e-10 | 24,656 / 10,639 | 26.94 | 22.52 [22.28-22.80] |
| Triaxial tube E3 | Working | 6.427e-08 | 4,318 / 1,242 | 4.27 | 4.08 [4.06-4.19] |
| Triaxial tube E3 | Order 1600 | 6.465e-08 | 4,317 / 1,236 | 9.13 | 9.12 [9.07-9.19] |
| Triaxial tube E3 | Accuracy 1e-11 | 8.936e-10 | 7,448 / 1,502 | 6.20 | 6.10 [6.06-6.97] |
| Triaxial box E3 | Working | 1.740e-08 | 9,924 / 4,278 | 9.95 | 9.43 [9.21-9.47] |
| Triaxial box E3 | Order 1600 | 1.740e-08 | 9,924 / 4,278 | 21.08 | 24.56 [23.09-25.95] |
| Triaxial box E3 | Accuracy 1e-11 | 2.324e-10 | 16,451 / 6,761 | 14.90 | 16.51 [15.58-17.14] |
| Triaxial box E5 | Working | 2.599e-08 | 14,100 / 6,213 | 15.33 | 13.16 [13.04-14.36] |
| Triaxial box E5 | Order 1600 | 2.592e-08 | 14,125 / 6,228 | 29.81 | 29.43 [29.39-30.40] |
| Triaxial box E5 | Accuracy 1e-11 | 8.344e-10 | 23,666 / 10,138 | 21.86 | 21.19 [21.01-21.98] |

### Retained-statistic controls

Total variation values below are percentages; the JSON stores fractions.
Velocity variation is the largest of the three component histogram distances.
The moment difference is the largest absolute difference in `<v_i**2>/V**2`.

| Launch | Working versus | Radial TV | Spatial TV | Maximum velocity TV | Second-moment difference |
| --- | --- | --- | --- | --- | --- |
| Oblate box E5 | Order 1600 | 0.000% | 0.000% | 0.000% | 2.112e-09 |
| Oblate box E5 | Accuracy 1e-11 | 0.002% | 0.004% | 0.002% | 4.088e-07 |
| Triaxial tube E3 | Order 1600 | 0.000% | 0.000% | 0.000% | 1.123e-10 |
| Triaxial tube E3 | Accuracy 1e-11 | 0.002% | 0.000% | 0.012% | 5.478e-09 |
| Triaxial box E3 | Order 1600 | 0.000% | 0.000% | 0.000% | 3.059e-12 |
| Triaxial box E3 | Accuracy 1e-11 | 0.006% | 0.002% | 0.002% | 2.289e-08 |
| Triaxial box E5 | Order 1600 | 0.026% | 2.872% | 0.814% | 2.109e-03 |
| Triaxial box E5 | Accuracy 1e-11 | 0.040% | 5.988% | 1.342% | 4.516e-03 |

First-period normalized position/velocity RMS differences remain below
`2.2e-10` for the order-1600 control and `2.2e-9` for the tighter solver.
The outer triaxial box subsequently develops larger differences over the
200-period window: spatial TV is 2.872% against order 1600 and 5.988%
against accuracy `1e-11`. Its working-run spatial split-half TV is 37.18%.
That comparison provides finite-window context, not an error bar or proof
of acceptable observable accuracy. No formal chaos classification was made.

### Memory and numerical equivalence

The final matrix completed in the restored 4 GB, two-CPU Colima VM.
Its cumulative Linux process peak resident memory was **0.811 GiB**
(`850580 KiB`), compared with about 7.3 GiB in the earlier two-variant
full-batch pilot at 8 GB. These are process peaks across different run sets,
not per-orbit or GPU memory requirements.

All ten available completed trajectories from before diagnostic batching
were checked against their reruns: saved times and all six phase-space
components are bitwise identical, retained statistics and step counts are
identical, and the largest absolute normalized energy-diagnostic change is
`5.30e-16`. Regression tests also compare diagnostics with direct full-array
energy evaluation below a batch boundary, at the boundary, and with a final
partial batch whose endpoint has the largest energy error.

The final outer-box order-1600 run completed with 20,353 steps and warm
median 29.43 seconds. The interrupted full-batch attempt has no completed
row for that control, so it supplies no paired numerical result or reliable
speedup estimate. Resource pressure and changed memory allocation prevent
isolated throughput claims from the before/after timings.

## Launch coverage and time variation

At working settings, the extrema of the stored samples and the differences
between the two contiguous half-windows are:

| Launch | Minimum sampled radius (pc) | Maximum sampled radius (kpc) | Spatial split-half total variation |
| --- | --- | --- | --- |
| Oblate box E5 | 23.62 | 15.838 | 33.14% |
| Strongly triaxial tube E3 | 184.37 | 0.29570 | 49.36% |
| Strongly triaxial box E3 | 0.9478 | 0.39688 | 17.68% |
| Strongly triaxial box E5 | 25.38 | 15.830 | 37.18% |

These raw spatial histograms have appreciable finite-window variation even
where numerical-control histograms agree. Convergence of final retained
contributions needs its own checks with the intended symmetry operations,
binning, integration duration, and observational quantities.

## Resource setup

The first full-sample working pilot was killed by Docker's out-of-memory
handler in the original 4 GB Colima VM. That attempt produced no numerical
result. On 2026-10-17, Colima was increased to 8 GB on the 16 GB Intel Mac,
with its two-CPU allocation retained. The VM and SSH configurations were
backed up before the change. Keep resource changes explicit when interpreting
timings against earlier measurements.

The raw six-component trajectory has about 2.4 MB of numerical data. Tracing
the original full-batch energy diagnostics showed logical float64 arrays of shape
`(800, 50002)`, about 320 MB each, across the six Gaussian components. At
order 1600 those logical arrays double in size. Compiler fusion affects which
arrays are materialized, so summing traced array sizes is not a peak-memory
measurement.

The host showed substantial memory compression and swap counters during
the original full-batch matrix. Ten variants completed; the final outer-box
order-1600 control had produced no completed row after 1,396 seconds when
the decision was made to stop the container. This was an interrupted resource
investigation, with no completed numerical result or step count for that
control. Compact timings, diagnostics, source hashes, and the interruption
record are retained in
[`orbit_checks_full_batch_attempt.json`](results/orbit_checks_full_batch_attempt.json).
Those wall times cannot establish isolated throughput or a solver speed ranking.

The production diagnostics now use `jax.lax.map` with batches of at most 128
states. All configured samples, the launch, and the exact endpoint remain
included; solver settings, force evaluation, sample times, and diagnostic
normalization retain their definitions. The final matrix uses this bounded
evaluation in the restored 4 GB VM, retaining two CPUs. Its source hashes
identify the actual files, including the
uncommitted diagnostic change; the Git revision identifies the baseline,
and `tnt_tracked_changes` lists tracked edits at measurement time.

## Validation

- All 48 final-matrix solves succeeded. A separate artifact audit checked
  source/input and trajectory hashes, output shapes/dtypes and time grids,
  histogram normalization, unchanged physical parameters between controls,
  repeated diagnostics, and all ten available pre-change trajectories.
- `pytest -q tests/unit_tests/test_orbit_integration.py`: 16 passed.
- `pytest -q tests/unit_tests/test_orbit_library_assembly.py`: 6 passed.
- `ruff check .` and the Sphinx HTML build with warnings treated as errors
  passed. The tests retain an existing upstream TensorFlow Probability
  deprecation warning. The full repository test suite was not run.
