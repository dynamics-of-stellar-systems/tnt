# Experimental direct-versus-grid comparison

This work is isolated on `codex/force-grid-benchmarks`, based on orbit
integration commit `69b4c2e`. It does not select a production evaluator or
change configuration defaults.

## Task list

- [x] Implement and validate a benchmark-only, reflection-symmetric force grid
  with explicit domain rejection.
- [ ] Compare off-grid forces and synchronized timings at three resolutions.
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

## Validation

`pytest -q tests/unit_tests/test_force_grid.py`: eight tests passed in the
Linux development container. They check analytic Plummer convergence and
radial endpoints, Gaussian nodes/axes/planes and reflection symmetry, units,
JIT domain rejection, unsupported models, and a circular-orbit comparison
using the production solver with direct physical-energy evaluation.
Ruff passed for the prototype and its tests. The full test suite was not run.
