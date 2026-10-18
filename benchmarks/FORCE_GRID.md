# Experimental direct-versus-grid comparison

This work is isolated on `codex/force-grid-benchmarks`, based on orbit
integration commit `69b4c2e`. It does not select a production evaluator or
change configuration defaults.

## Task list

- [x] Implement and validate a benchmark-only, reflection-symmetric force grid
  with explicit domain rejection.
- [x] Compare off-grid forces and synchronized timings at three resolutions.
- [ ] Compare matched 200-period orbits, retained intrinsic statistics, direct
  energy diagnostics, setup costs and warm execution; record results and limits.

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

Errors use vector norms, avoiding division by zero components on symmetry
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
