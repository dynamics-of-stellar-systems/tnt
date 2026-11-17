# Small-library interpolation comparison

For cluster continuation, start with the
[GPU handoff](../../aidocs/GPU_INTERPOLATION_HANDOFF.md).

The experiment uses a strongly triaxial six-Gaussian stellar potential,
NFW halo and Plummer central mass. The YAML files in `input/` contain TNT
potential settings plus benchmark-specific MGE, sampling and integration fields. The current CPU baseline is
`input/config-round2.yaml`; `input/config.yaml` describes the first round. The runner
constructs TNT objects directly; this file is not a general TNT configuration.
`input/potential.json` preserves the earlier comparison's model as provenance.
No DYNAMITE code or input format is used by the runner.

## Historical first-round matrix

- `XZGridFromBoundary` and `StationaryGrid`, each `(2, 4, 2)`:
  16 launches each, 32 total, one member per bundle.
- Shared direct-force launches and boundary-derived shell clocks.
- Direct, trilinear force, cubic-spline force, and Hermite potential evaluators.
- Identical `(160, 24, 24)` grids over `1e-5–200 kpc` for the interpolators.
- Float64 Dopri8; nominal tolerances `1e-5`, `1e-7`, `1e-9`, `1e-11`;
  200 reference periods and 50,000 saved samples per orbit.
- 512 solves overall; no timing repeats. Construction and compilation are
  recorded separately from integration, which includes direct-energy diagnostics.

The first-round three-tolerance suite is in `work/suite-20261111`. The separate
`1e-5` extension is in `work/suite-1e-5-20261111`. The original run's exact config
is preserved in its work directory; the extension uses `input/config-1e-5.yaml`.
Both share tables and launches measured in `work/pilot-20261111`, with setup
provenance recorded. No historical results are overwritten.

## Analysis

Open `analysis.ipynb` and run all cells to refresh the live snapshot. It reads
both suite directories, reports pending and failed cases, and plots:

- construction costs and complete-library integration times;
- physical-energy drift and adaptive step counts versus tolerance;
- selected-orbit trajectories, physical and Hermite energy histories;
- trajectory-error histories against direct integration at `1e-11`.

Change the orbit index and selected tolerance in the first code cell. The
notebook does not launch integrations. Partial totals are not treated as full
library timings. Single executions do not provide robust speed rankings, and
long-time trajectory separation alone does not establish observable error.

## Execution and artifacts

From the repository root, with the scientific environment available:

```shell
python -m benchmarks.interpolation_library.run --phase pilot --work benchmarks/interpolation_library/work/new-pilot
python -m benchmarks.interpolation_library.run --phase suite --work benchmarks/interpolation_library/work/new-suite --reuse-setup benchmarks/interpolation_library/work/new-pilot
```

On macOS prefix long commands with `caffeinate -i`. Run scientific processes
sequentially. Exact source and config hashes protect resuming; use a fresh work
directory after source/config changes. Per-orbit JSON records permit continuing
an interrupted run, with existing records skipped, including recorded failures.

Large trajectories, tables, full sampled energy histories and run logs remain in
ignored `work/`. Trajectory NPZs store time in Myr, position in kpc, velocity in
kpc/Myr and signed energy drift divided by the launch velocity scale squared.
Hermite outputs also contain interpolated-Hamiltonian drift. JSON integration
energy diagnostics additionally include the exact endpoint, which is not a
saved trajectory sample. Portable round-two baseline and quintic summaries are
committed in `results/`. Raw trajectories remain outside Git.

## Completed second round

`input/config-round2.yaml` extends the common interpolation domain inward to
`1e-7 kpc` and explicitly enables direct-force fallback outside `[1e-7,200] kpc`.
All 384 original second-round solves and the 96-solve quintic extension
completed successfully. The earlier targeted smoke check used previously failing
inner and outer stationary launches at `1e-5`.

The fallback is a hard switch, so Hermite's piecewise energy evaluator should
not be interpreted as a globally smooth Hamiltonian across the radial boundary.
The output field `saved_samples_outside_grid` counts output samples, not internal
adaptive-stage force calls. Use a fresh work directory for the second round.

## Second-round analysis

`analysis-round2.ipynb` combines `work/round2-20261111` (384 successful
solves, 24 launches) with the 96-solve quintic extension in
`work/round2-quintic-20261117`. It preserves the timing, energy/step and selected-trajectory
figures, omits selected energy histories, and retains second-half density
marginals. Its final figure shows reference-density-weighted median absolute
3D-density error for every orbit versus tolerance, with one panel per evaluator.
The original `analysis.ipynb` remains the first-round analysis.

## Quintic follow-up (2026-11-17)

`python -m benchmarks.interpolation_library.quintic_probe` reuses the preserved
round-two model, cubic Hermite jets and launches. It evaluates 2,048 reproducible
off-node forces (isotropic directions, log-uniform radii 0.001–100 kpc), then
integrates launches 0 and 16 for 200 periods at `1e-9`. New trajectories live in
`work/quintic-probe-20261117`; portable results and input hashes are in
`results/quintic-probe-20261117.json`. Existing results are preserved.

Median relative force errors: linear 1.35e-4, cubic force 6.54e-8, cubic Hermite
5.06e-6, quintic potential 7.18e-9. The quintic 95th percentile is 1.03e-7.
Quintic physical-energy errors for the two launches are 8.22e-8 and 3.41e-7,
versus cubic Hermite 2.25e-7 and 3.09e-6. Attempted steps drop from 13,715 to
4,896 and from 61,103 to 19,185. These are targeted checks, not a complete
library comparison or robust timing measurement. Grid size increases from
5.625 MiB for cubic Hermite to 18.984 MiB for quintic potential.

The runner also accepts an optional config `methods` list containing `quintic`;
the historical default matrix is unchanged. Quintic outputs use the existing
`hermite_energy_drift` field for their interpolated Hamiltonian.


## Complete quintic second-round extension (2026-11-17)

The unchanged quintic implementation was run on all 24 round-two launches at
`1e-5`, `1e-7`, `1e-9` and `1e-11`: **96/96 solves succeeded**. Potential,
grid geometry, radial boundary conditions, direct fallback, seeds, 200-period
durations and 50,000 saved samples are unchanged. The launch NPZ was copied
byte-for-byte from round two; `launch-origin.json` records the source hashes.
Only quintic was newly integrated. The original 384 results are preserved.

`input/config-round2-quintic.yaml` selects the added method. The work directory
archives the exact config, evaluator/runner sources, grid, provenance and log.
Run a new reproduction in a fresh work directory using the same archived launch
NPZ and `python -m benchmarks.interpolation_library.run --phase suite --config
benchmarks/interpolation_library/input/config-round2-quintic.yaml --work <directory>`.
Without a copied launch archive, the runner regenerates launches deterministically
from the model and sampler settings. The seed controls integration sampling.
Use `input/launches-round2.npz` for an exact matched comparison.

`analysis-round2.ipynb` includes quintic in every existing comparison, with five
panels in the final density figure, and embeds freshly executed outputs for all
480 cases. Portable records, density statistics, comparisons and verification
are in `results/quintic-round2-20261117.json`. Every new trajectory was checked
for finite arrays, 50,000 samples and output times identical to its direct
reference. The new timing measurements were made separately from the original
methods, on a different date; they are not controlled repeated benchmarks.

At `1e-11`, the median maximum physical-energy error in the notebook's reference
mean-square-velocity normalization is 2.51e-9 (direct), 5.30e-7 (cubic force),
1.02e-6 (cubic Hermite), and 1.39e-8 (quintic potential). Median attempted step
counts are 16,715, 34,648, 129,053.5, and 20,107.5, respectively. Quintic's
median weighted density error is zero at `1e-9` and `1e-11`, as for cubic force;
individual orbit errors remain large, and a zero weighted median is not zero
error in the full density distribution.


## CPU checkpoint and cluster transfer

`protocol.json` describes the completed 480-case CPU comparison.
`protocol-round1.json` preserves the earlier planning snapshot unchanged,
including its historical `pilot_running` status. The complete 384 original
per-orbit records are in `results/round2-baseline-20261111.json`; the quintic
extension and density comparisons are in `results/quintic-round2-20261117.json`.

The byte-identical 24-launch archive is committed as `input/launches-round2.npz`;
its checksum, units and origin are in `input/launches-round2.json`. To reproduce
quintic on a fresh checkout from the repository root:

```shell
mkdir -p benchmarks/interpolation_library/work/new-quintic
cp benchmarks/interpolation_library/input/launches-round2.npz benchmarks/interpolation_library/work/new-quintic/launches.npz
python -m benchmarks.interpolation_library.run --phase suite --config benchmarks/interpolation_library/input/config-round2-quintic.yaml --work benchmarks/interpolation_library/work/new-quintic
```

The runner currently enables float64. Precision selection and GPU controls belong
to the next experiment; neither float32 performance nor accuracy has been measured.
Use fresh work directories across checkouts: saved resume provenance includes
source paths and exact source/config hashes.

The two completed round-two work directories remain preserved locally and
ignored by Git (about 1.4 GiB combined). Retain or back them up before removing
local files. To recompute every notebook plot on the cluster, transfer both
`work/round2-20261111/` and `work/round2-quintic-20261117/`. For a smaller GPU
validation transfer, the committed launch archive plus the 24
`round2-20261111/orbit-*-direct-1e-11.npz` reference trajectories are sufficient
for trajectory-reference comparisons; full notebook refresh still needs all
480 trajectories. The embedded notebook outputs and portable JSON summaries
can be inspected without those raw arrays.

Earlier suite, pilot, smoke and probe work directories are retained as historical
artifacts. They are not needed for the current round-two GPU comparison.
