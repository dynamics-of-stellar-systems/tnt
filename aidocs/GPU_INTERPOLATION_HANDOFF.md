# GPU interpolation handoff — 2026-11-17

## Objective and agreed scope

Compare **direct forces, cubic force interpolation, and quintic potential
interpolation on the cluster GPUs**, in both float64 and float32. Linear force
and cubic Hermite potential interpolation remain optional diagnostic controls.
The user explicitly wants float32 investigated, not only float64.

Keep the existing scientific implementation and boundary conditions for this
comparison. Defer new asymptotic boundary conditions, broad refactoring and
production integration until the current implementation has been measured.
Compare at matched solver tolerance **and matched achieved accuracy**. Do not
infer accuracy from energy conservation alone, or from long-time trajectory
separation alone.

## Repository checkpoint and navigation

- Working branch: `codex/gpu-force-interpolation`.
- CPU checkpoint: `e413d16`; quintic implementation: `b8c0fa5`.
- These branches/commits were local when the handoff was written. Push the GPU
  branch from the laptop before trying to fetch it on the cluster.
- `benchmarks/force_grid.py`: all interpolators, grid construction and experimental
  orbit adapter. These are outside the production potential registry.
- `tests/unit_tests/test_force_grid.py`: 49 tests passed at the checkpoint,
  including an independent SciPy B-spline comparison.
- `benchmarks/interpolation_library/run.py`: CPU comparison runner; currently
  enables float64 unconditionally. It is not yet a GPU/precision benchmark CLI.
- `benchmarks/interpolation_library/README.md`: reproduction and data layout.
- `benchmarks/interpolation_library/protocol.json`: completed round-two protocol.
- `benchmarks/interpolation_library/analysis-round2.ipynb`: all five methods,
  with executed outputs. `analysis.ipynb` and `protocol-round1.json` are historical.
- `aidocs/KNOWLEDGE.md` and `aidocs/ORBIT_INTEGRATION_FOLLOW_UPS.md`: broader
  architecture and numerical constraints; older benchmark discussions precede
  the quintic follow-up, so use the current protocol for this experiment.

## What was implemented and why

Cubic force interpolates `log(-a_i/x_i)` with tensor cubic B-splines, then
reconstructs the Cartesian force. Its force is C2 at interior grid joins but
is not constrained to be conservative.

Cubic Hermite potential matches potential and derivative jets, but its potential
is only C1 and force C0 at interior grid joins. This caused concern for adaptive
high-order integration; the quintic follow-up substantially reduced solver work.

Quintic potential retains the same eight nodal jets:
`Phi`, `Phi_u`, `Phi_theta`, `Phi_phi`, `Phi_u_theta`, `Phi_u_phi`,
`Phi_theta_phi`, and `Phi_u_theta_phi`, with `u = log(r/r_min)`.
They are evaluated from the physical potential with JAX autodiff. Actual stored
jets use dimensionless grid-index coordinates, including spacing factors.
Globally solved nodal curvatures extend the tensor to 27 entries per node.
Neighbouring pieces are constrained to share third derivatives; pure third
physical derivatives are not supplied as nodal data.

This construction is equivalent to quintic B-splines with **double interior
knots**, not simple knots: potential C3, force C2 at regular interior joins.
There are no doubled physical sampling points. At both radial endpoints the
fourth derivative in log radius vanishes. At angular endpoints third derivatives
vanish alongside the symmetry-imposed zero first derivatives. These are the
existing conditions to retain. Higher Cartesian regularity at spherical poles
has not been established. Direct fallback is a hard radial switch and does not
guarantee a globally smooth Hamiltonian across that boundary.

Grid memory: linear/cubic force 2.109 MiB; cubic Hermite 5.625 MiB; quintic
18.984 MiB. The user considers this reasonable relative to orbit-library storage.

## Fixed reference experiment and findings

Strongly triaxial six-Gaussian stellar potential + NFW halo + softened Plummer
central mass; Gaussian quadrature order 800. Use the committed round-two configs.
There are 24 identical launches, 200 shared reference periods, 50,000 saved
samples, grid `(160,24,24)` over `1e-7–200 kpc`, and direct fallback outside.
Float64 Dopri8 tolerances: `1e-5`, `1e-7`, `1e-9`, `1e-11`.
All 480 cases (five methods) succeeded. The original methods and quintic ran
separately on different dates; their wall times are not controlled repeated
measurements.

At `1e-11`, median maximum physical-energy errors divided by each orbit's
float64 direct-reference mean-square velocity, and median attempted steps:

| Method | Energy error | Steps |
| --- | ---: | ---: |
| Direct | 2.51e-9 | 16,715 |
| Cubic force | 5.30e-7 | 34,648 |
| Cubic Hermite | 1.02e-6 | 129,053.5 |
| Quintic potential | 1.39e-8 | 20,107.5 |

Quintic and cubic force have zero median weighted density error at the tightest
two tolerances, but individual orbit errors remain large, including sensitivity
in direct integrations. The statistic can hide errors affecting less than half
the reference probability mass; zero median does not mean zero density error.
JSON integrator energy errors use a launch velocity scale; the notebook converts
them to the reference mean-square velocity. Preserve that distinction.

## First cluster steps

1. Inspect the actual scheduler, allocation rules, GPU model, driver, available
   memory and existing Python/CUDA environment before proposing job commands.
   These details are unknown; do not assume Slurm or a particular GPU vendor.
   Use allocated compute resources for scientific runs.
2. Preserve the locked dependency baseline where compatible. Record any GPU
   backend/package changes, exact source/config hashes, hardware and devices.
   Verify actual accelerator availability; do not silently time a CPU fallback.
3. Reproduce a small CPU float64 case, then the same GPU float64 case using the
   committed launch archive, before expanding to the full comparison.
4. Add explicit precision selection for the benchmark. Audit the unconditional
   x64 setting, numerics config, constants, input arrays and table/evaluator
   dtypes. Establish precision before importing Galax, which may create constants
   at import. Use separate processes for precision configurations.
5. Investigate GPU float32 for all three candidate methods. Measure its attainable
   accuracy floor with suitable tolerances; do not blindly reuse `1e-11`.
   Use the shared float64 launches/reference clocks initially to isolate force
   and integration precision from launch-generation changes. This does not
   validate float32 initial-condition generation for production.
6. Separate construction precision from storage/evaluation/integration precision:
   construct a table in float64 and cast it to float32, then compare with genuine
   float32 construction. Check actual array/compute dtypes, not just the x64 flag.
   Examine cancellation when differentiating the potential and near-axis/small
   radius behaviour. Current grid loading helpers include float64 assumptions.
7. Measure synchronized warm force evaluation and complete-orbit performance.
   Separate compilation, construction, transfer, diagnostics and file export;
   record memory, steps and repeated/interleaved timings. The historical runner's
   integration timing includes direct physical-energy diagnostics but excludes
   subsequent full energy-history export and disk I/O.
8. Measure serial latency and batched throughput separately. Benchmark batching
   needs investigation; existing fallback code avoids `vmap(cond)` because it
   can execute both direct and interpolated branches. Preserve fallback semantics.

## Artifacts to move

The committed `input/launches-round2.npz` under the interpolation-library folder
contains the exact launches, velocities, reference periods and sampler labels.
Its checksum and units are in `input/launches-round2.json`.

All portable records/results and embedded notebook outputs are in Git. Full
trajectories, grids and logs are ignored and **will not arrive with a clone**:

- Minimum existing trajectory references: the 24 files
  `benchmarks/interpolation_library/work/round2-20261111/orbit-*-direct-1e-11.npz`.
- Full notebook refresh: both `work/round2-20261111/` and
  `work/round2-quintic-20261117/` (about 1.4 GiB combined).
- The remaining old work directories are historical and unnecessary for this
  GPU comparison. All raw data were retained locally; no archive was deleted.

Use fresh run directories on the cluster. Existing resume provenance contains
source paths and hashes; changing checkout paths or source/config invalidates it.
Do not overwrite the CPU reference results or reuse their records as new GPU runs.
