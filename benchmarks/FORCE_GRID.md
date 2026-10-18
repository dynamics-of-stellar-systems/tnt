# Experimental direct-versus-grid comparison

This work is isolated on `codex/force-grid-benchmarks`, based on orbit
integration commit `69b4c2e`. It does not select a production evaluator or
change configuration defaults.

## Findings and recommendation

Retain the direct static evaluator. The fine trilinear grid has sampled
relative force differences around `2.6e-4` and maximum normalized physical
energy drift `4.0e-5` to `2.7e-4`, versus `1.7e-8` to `6.4e-8` for direct
integration. At the agreed state tolerance `1e-9`, it also requires many
more adaptive steps. The first three fine-grid whole-solve medians are
1.44-1.52 times slower than direct evaluation; the final case's wider timing
spread is being checked separately.

Coarse grids offer a runtime saving with substantially larger numerical
differences. These results do not support enabling this prototype as a
production default. If interpolation remains the next numerical priority,
investigate a smoother method, preferably deriving forces from one
interpolated scalar potential, then repeat the same controls. Retain explicit
error targets and the direct evaluator as the reference. This experiment
does not determine the best possible grid method or validate DYNAMITE's
exact table implementation.

## Task list

- [x] Implement and validate a benchmark-only, reflection-symmetric force grid
  with explicit domain rejection.
- [x] Compare off-grid forces and synchronized timings at three resolutions.
- [x] Compare matched 200-period orbits, retained intrinsic statistics, direct
  energy diagnostics, setup costs and warm execution; record results and limits.
- [ ] Recheck the noisy final-case timing in rotated order and export summary
  figures without discarding the original observations.

## Prototype scope

The first experiment follows DYNAMITE's coordinate and storage choice:
uniform grids in log radius, polar angle and azimuth within the positive
octant; trilinear interpolation of `log(-a_i/x_i)`; reconstruction as
`a_i = -x_i * exp(interpolated coefficient)`. It assumes a static, coaxial,
reflection-symmetric model. Its axes are the model's intrinsic Cartesian axes.
It is not a literal DYNAMITE port or a DYNAMITE runtime benchmark.

The source inspected is DYNAMITE commit
`0bd10a9586936bf6b9bcfdcd3679477d812a01d0`,
[`legacy_fortran/interpolpotent.f90`](https://github.com/dynamics-of-stellar-systems/dynamite/blob/0bd10a9586936bf6b9bcfdcd3679477d812a01d0/legacy_fortran/interpolpotent.f90).
DYNAMITE uses 640 radial by 64 by 64 angular nodes, moves angular endpoint
samples inward, and evaluates the direct force outside the grid. This
experiment instead rejects out-of-domain queries. Endpoint coefficients are
evaluated with each positive coordinate at least `1e-7 * radius`, a declared
approximation to the symmetry-plane limit that is checked against direct
axis/plane probes. No automatic fallback is provided.

The tabulated force need not be the gradient of one scalar interpolated
potential. Energy diagnostics therefore always use the unchanged direct
physical potential; they capture force interpolation effects as well as
integration error. Parameter gradients and production configuration/model
wiring are outside this experiment.

A table belongs to one fixed physical model and must be rebuilt when that
model changes. Saved tables do not preserve an automatic-differentiation
graph linking their values to model parameters. Parameter-gradient accuracy,
cache invalidation across models, GPU performance, rotated components,
central/orbit-domain coverage and a production boundary policy need their
own validation before runtime integration.

## Force protocol and results

Run date: 2026-10-18. Both the oblate and strongly triaxial six-Gaussian
galaxy models include their unchanged Plummer black hole and NFW halo.
Float64 and Gaussian quadrature order 800 are retained. Each model uses
4,645 fixed probes: 4,096 draws uniform in log radius/polar angle/azimuth
with random octant signs, plus 61 radii on nine axial/planar/diagonal/near-axis
directions. Radius covers `1e-5` to `200 kpc`; these probes are shared across
all grids and are not a complete parameter-domain validation.

| Resolution | Radial / polar / azimuth nodes | Table size | Maximum relative force error, oblate / triaxial | Warm construction, oblate / triaxial |
| --- | --- | --- | --- | --- |
| Coarse | 80 / 12 / 12 | 0.264 MiB | 0.3958% / 0.3936% | 0.269 / 0.486 s |
| Medium | 160 / 24 / 24 | 2.109 MiB | 0.1002% / 0.1031% | 1.812 / 1.854 s |
| Fine | 320 / 48 / 48 | 16.875 MiB | 0.02465% / 0.02507% | 14.969 / 15.464 s |

Errors compare against the order-800 direct evaluator, not an exact solution.
They use vector norms, avoiding division by zero components on symmetry
planes. The fine grid's 99th percentile relative error is about `2.0e-4` in
both models. Increasing resolution reduces sampled force error by about
four times per level. This is much larger than the direct evaluator's
measured quadrature error and the `1e-9` state tolerance; those are distinct
error sources.

Each timing uses the same 2,048 off-node queries, one first call and three
synchronized warm repeats. Sequential compiled scans cost about 90.7 ms
for direct forces and 3.0-4.3 ms for grids: about 21-30 times faster for
this request pattern. Batched timings are also retained. Neither ratio
predicts a complete orbit or production-library speedup.

Grid lowering/compilation, first execution, one warm rebuild, and table I/O
are separated. Compiled kernels can be reused between models with the same
structure; the second model's compilation entries therefore include cache
reuse. Setup measurements are individual observations, not confidence
intervals. The process peak resident memory is 1.205 GiB in the existing
4 GB, two-CPU Linux VM; no memory setting was changed.

The direct scalar/batch discrepancy reaches `1.14e-11` relative near the
centre, consistent with cancellation-sensitive NFW arithmetic. It is
recorded separately and checked against an explicit `1e-9` protocol bound,
not treated as interpolation error or a physical acceptance threshold.

Full probes, forces, errors, timings, physical models and source/input hashes
are retained in [`grid_forces.json`](results/grid_forces.json). Raw tables
and individual records live in ignored `benchmarks/work/` directories.

## Reproduce

Mount an explicit read-only checkout of the established
`tnt-dynamite-comparison` repository. Choose fresh work/output paths:

```shell
docker --context colima compose run --rm --volume /absolute/path/to/tnt-dynamite-comparison:/comparison:ro dev python -m benchmarks.grid_benchmark --comparison-checkout /comparison --work-dir /workspace/benchmarks/work/grid-forces-new --output /workspace/benchmarks/results/grid_forces-new.json
```

`--case`, `--resolution`, `--warm-repeats`, `--r-min`, and `--r-max` select
explicit variants; no automatic tuning, fallback, or output overwriting is
performed. The model builder, exported launches, and provenance machinery
are shared with `benchmarks/orbit_checks.py`; force timing functions are
shared with `benchmarks/gaussian_forces.py`. Run scientific processes
sequentially.

The orbit phase consumes those exact table files and verifies their hashes,
specifications, physical model, numerical source files, library versions and
comparison inputs. Use the same case selection as the force phase:

```shell
docker --context colima compose run --rm --volume /absolute/path/to/tnt-dynamite-comparison:/comparison:ro dev python -m benchmarks.grid_orbits --comparison-checkout /comparison --force-result /workspace/benchmarks/results/grid_forces-new.json --force-work-dir /workspace/benchmarks/work/grid-forces-new --work-dir /workspace/benchmarks/work/grid-orbits-new --output /workspace/benchmarks/results/grid_orbits-new.json
```

## Orbit protocol

The four launches are the oblate outer box E5, strongly triaxial tube E3,
strongly triaxial box E3, and strongly triaxial outer box E5. Every direct
and grid solve uses the identical exported launch, DYNAMITE shell clock,
random key, 200-period duration, 50,000 output times, float64, Dopri8,
normalized tolerance `1e-9`, and maximum one million adaptive steps.
Each variant has one first compiled execution and three warm repeats.
Compilation is measured separately and can reuse kernels between launches.
The unchanged production solver is shared through a benchmark-only adapter;
the public runtime API still rejects that adapter.

Whole-solve timings include uniform output samples and energy diagnostics
at every sample plus the exact endpoint. Model creation, table loading,
compilation, statistics and artifact I/O are recorded separately or excluded.
The scalar direct potential is retained for all energy diagnostics:
`abs(E-E_launch)/V**2`, where `V` is the existing launch/reference-clock
normalization. These are not energy errors divided by initial energy and
are not the solver's local state tolerance.

Retained statistics use the same intrinsic radial, Cartesian and velocity
bins and moments as the direct-orbit benchmark. Histogram total variation
is half the sum of absolute probability differences. These precede spatial
symmetry operations, counter-rotation and observational projection. Split-half
variation provides finite-window context rather than formal error bars.
First-period position/velocity differences are normalized root mean square
component differences; long-window pointwise agreement is not required for
sensitive orbits. A further 2,000 uniformly indexed samples per direct orbit
probe the grid forces along exactly matched physical paths.

Raw trajectories remain in ignored work directories. The portable orbit
artifact retains full histograms/moments, diagnostics, repeated timings,
force probes, artifact/source hashes and control differences. Timings and
statistics do not assume a production library or an accepted interpolation
error target.

## Orbit results

All 64 solves succeeded. Full-precision data are retained in
[`grid_orbits.json`](results/grid_orbits.json). The original warm timing
observations, including the noisy final-case observation, are preserved below.

| Launch | Evaluator | Maximum normalized energy drift | Accepted / rejected steps | Warm median [range] (s) |
| --- | --- | --- | --- | --- |
| Oblate box E5 | direct | 2.884e-08 | 14,712 / 6,506 | 14.13 [13.95-14.74] |
| Oblate box E5 | coarse | 4.561e-03 | 64,616 / 146,115 | 5.51 [5.10-5.53] |
| Oblate box E5 | medium | 1.109e-03 | 89,239 / 204,663 | 18.40 [17.74-19.89] |
| Oblate box E5 | fine | 2.735e-04 | 110,902 / 222,717 | 20.62 [20.27-20.75] |
| Triaxial tube E3 | direct | 6.427e-08 | 4,318 / 1,242 | 4.25 [4.25-4.47] |
| Triaxial tube E3 | coarse | 6.603e-04 | 30,268 / 62,963 | 2.53 [2.46-2.84] |
| Triaxial tube E3 | medium | 1.499e-04 | 38,257 / 62,931 | 6.50 [6.50-6.54] |
| Triaxial tube E3 | fine | 3.960e-05 | 45,838 / 47,553 | 6.14 [6.13-6.19] |
| Triaxial box E3 | direct | 1.740e-08 | 9,924 / 4,278 | 9.70 [9.39-9.82] |
| Triaxial box E3 | coarse | 1.172e-03 | 46,545 / 104,740 | 3.56 [3.47-3.59] |
| Triaxial box E3 | medium | 2.630e-04 | 63,644 / 143,416 | 12.86 [12.66-13.76] |
| Triaxial box E3 | fine | 6.531e-05 | 79,020 / 152,426 | 14.77 [14.62-14.94] |
| Triaxial box E5 | direct | 2.599e-08 | 14,100 / 6,213 | 18.11 [14.89-93.17] |
| Triaxial box E5 | coarse | 4.434e-03 | 62,415 / 137,746 | 5.89 [5.64-7.08] |
| Triaxial box E5 | medium | 1.042e-03 | 85,461 / 189,703 | 25.90 [24.47-26.90] |
| Triaxial box E5 | fine | 2.586e-04 | 106,806 / 207,554 | 28.50 [26.16-30.09] |

The fine-grid solves require 15.5-16.8 times as many total adaptive steps
as direct integration. Roughly two-thirds of grid step attempts are rejected.
The force is continuous but its derivatives change at interpolation-cell
boundaries. Increased adaptive stepping erodes the cheap-force advantage;
we have not profiled the individual sources of per-step overhead.

### Intrinsic retained-statistic differences

Values below compare each grid with the freshly measured direct orbit.
Histogram distances are percentages. Velocity total variation is the largest
of its three component distances; second-moment differences use `V**2` units.

| Launch | Grid | Radial TV | Spatial TV | Maximum velocity TV | Second-moment difference |
| --- | --- | --- | --- | --- | --- |
| Oblate box E5 | coarse | 0.162% | 17.116% | 3.864% | 1.195e-02 |
| Oblate box E5 | medium | 0.046% | 20.116% | 2.952% | 8.108e-03 |
| Oblate box E5 | fine | 0.074% | 14.010% | 2.410% | 6.482e-03 |
| Triaxial tube E3 | coarse | 0.096% | 3.898% | 1.730% | 5.166e-04 |
| Triaxial tube E3 | medium | 0.046% | 0.642% | 0.300% | 1.303e-04 |
| Triaxial tube E3 | fine | 0.028% | 0.544% | 0.224% | 4.051e-05 |
| Triaxial box E3 | coarse | 0.196% | 3.604% | 0.856% | 3.295e-04 |
| Triaxial box E3 | medium | 0.204% | 1.592% | 0.498% | 1.289e-04 |
| Triaxial box E3 | fine | 0.174% | 0.584% | 0.204% | 7.261e-05 |
| Triaxial box E5 | coarse | 0.182% | 14.498% | 4.066% | 1.764e-02 |
| Triaxial box E5 | medium | 0.134% | 14.390% | 4.426% | 8.115e-03 |
| Triaxial box E5 | fine | 0.114% | 11.340% | 2.888% | 1.088e-02 |

Fine-grid first-period position and velocity RMS differences remain below
`1.5e-4` in launch-normalized units. Along the 8,000 uniformly indexed
direct-trajectory points, the fine grid reaches maximum relative force
difference `2.59e-4`; the three resolutions produce another 24,000 matched
force comparisons.

The outer-box spatial histogram differences are 14.01% (oblate) and
11.34% (triaxial) even at the fine resolution. Their direct split-half
distances are 33.14% and 37.18%, respectively. The earlier order-1600 /
accuracy-1e-11 controls gave 0% / 0.004% for the oblate box and 2.872% /
5.988% for the triaxial box. Split-half variation is not an acceptance
threshold: it does not make a numerical difference acceptable. Time/sampling
convergence and projected-observable error targets remain separate work.

Increased resolution improves force and energy diagnostics systematically;
some long-window histogram distances are not monotonic. These observations
do not formally classify any orbit as chaotic.

### Setup and resource observations

Table loading and device transfer cost 0.006-0.149 s here, excluding the
separately recorded hash verification. This cannot replace per-model table
construction when physical parameters change. All setup and compilation
observations remain in the portable artifacts; compilation can be reused
between matching model structures. The full orbit matrix has process peak
resident memory 1.100 GiB in the unchanged 4 GB, two-CPU VM.

The final direct outer-box timing series spans 14.89-93.17 s; the other
cases have much smaller spreads. The cause of that outlier has not been
isolated. An additional rotated-order timing control is being measured
and will be reported separately. No original observation is discarded.

Build-only break-even estimates in the JSON divide warm grid-construction
time by warm time saved per identical repeated launch. A null result means
no warm solve saving was observed. They exclude compiler cost and accuracy
acceptance and do not estimate production-library throughput.

## Validation

`pytest -q tests/unit_tests/test_force_grid.py`: eight tests passed in the
Linux development container. They check analytic Plummer convergence and
radial endpoints, Gaussian nodes/axes/planes and reflection symmetry, units,
JIT domain rejection, unsupported models, and a circular-orbit comparison
using the production solver with direct physical-energy evaluation.
Ruff passed for the prototype and its tests. The full test suite was not run.
An artifact audit verified all source/table/probe hashes, dtypes/shapes,
timing medians, convergence and recorded errors. Independent SciPy linear
interpolation agrees with the JAX grid on all 27,870 force comparisons to
better than `1e-12` relative. These checks validate the implementation and
retained artifacts, not final observable accuracy.

The orbit artifact audit verified all 16 variant hashes, source/input
fingerprints, finite float64 output shapes, matched uniform clocks, physical
parameters, repeated diagnostics, step counts, all-sample histograms/moments,
control distances and first-period errors. All four new direct trajectories
and their saved times are bitwise identical to the previous direct benchmark.
