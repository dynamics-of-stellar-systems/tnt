# Gaussian force audit

For the subsequent full 200-period checks of the configured order-800
evaluator, see [Representative orbit checks](ORBIT_CHECKS.md).

Run date: 2026-10-17. This audit investigates TNT's pinned Galax direct
Gaussian evaluator, independently of orbit-solver tolerance and interpolation.

## Reproduce

From the repository root, using the Linux development environment:

```shell
docker --context colima compose run --rm dev python benchmarks/gaussian_forces.py --output /workspace/benchmarks/results/gaussian_forces-new.json
```

Use a new output filename: the script refuses to overwrite existing results.
It requires the project's pinned scientific environment, including SciPy
(installed by JAX). It does not require DYNAMITE or introduce a runtime
force-evaluation alternative. Run one scientific process at a time in the
memory-limited development container.

`results/gaussian_forces.json` records the harness/source hashes, package
versions, CPU backend, all probes, reference/evaluated forces, per-probe
errors, and seven synchronized warm timing repeats. First calls are recorded
separately; later first calls can reuse compiled kernels across shapes.
`--orders` and `--repeats` allow explicit alternate sweeps.

## Reference and coverage

The reference analytically differentiates the ellipsoidal Gaussian potential
and uses independent scalar adaptive force integrals (`scipy.integrate.quad`,
absolute tolerance `1e-13`, relative tolerance `2e-12`). Galax differentiates
its fixed-order potential quadrature instead. The reference uses
`u=s*max(1,r/sigma)` to resolve the outer-field integral and neglects only
the exponentially bounded tail beyond `u=12`. Integration warnings/errors
are rejected. An independent spherical shell-theorem calculation cross-checks
the reference, including the far field.

The audit uses `G=M=sigma=1`: radius means physical radius divided by each
Gaussian's width, not galaxy radius in kpc. It covers 11 shapes: spherical;
oblate axis ratios 0.8, 0.3, 0.1, 0.03, 0.01; and triaxial `(p,q)` of
`(0.8,0.6)`, `(0.7,0.3)`, `(0.3,0.1)`, `(0.1,0.03)`, `(0.1,0.01)`.
Both Galax Gaussian classes are exercised for axisymmetric shapes.

There are 78 probes per class/shape/order: the origin, plus 11 radii from
`1e-4` to `1e4` widths in seven directions, including principal axes,
off-axis, near-plane, and signed coordinates. Five orders across 17
class/shape combinations give 85 rows and 6,630 force comparisons. Errors
are vector-norm relative errors; zero-force origin probes use absolute
errors rather than division by zero.

## Results and working choice

| Order | Largest relative force error, all probes | Largest error for q >= 0.03 | Median serial microseconds per force |
| --- | --- | --- | --- |
| 50 | `1.00` | `1.00` | 2.13 |
| 100 | `1.16` | `1.16` | 2.69 |
| 200 | `1.13e-1` | `1.13e-1` | 3.97 |
| 400 | `2.30e-4` | `2.30e-4` | 6.40 |
| 800 | `1.86e-9` | `8.95e-12` | 11.07 |

These maxima include deliberately demanding probes. Strong flattening creates
a central boundary layer in the integral; large radius/width creates a narrow
outer-field contribution. A higher fixed order need not improve every probe
monotonically before convergence. For `q>=0.03` and sampled radii up to 1,000
widths, orders 200 and 400 reach about `5.54e-7` and `1.95e-11`, respectively.
However, order 200 has about 11% error at 10,000 widths, and the `q=0.01`
central cases need more nodes too.

The evidence supports **800 as a configurable working default** for MGE
Gaussian potential quadrature. This is distinct from the orbit solver's
normalized `accuracy=1e-9`. Users choosing a lower order need convergence
checks over their actual radii/widths and shapes. The sampled envelope is
not a universal guarantee: thinner shapes, larger ratios, untested directions,
potential/parameter derivatives, and float32 need separate investigation.
There is no automatic order selection, error estimator, or regime switch.

Each timing covers 2,496 force requests (32 repeats of the probes), either
batched or in a compiled serial scan. The median paired order-800/order-200
warm cost ratios are approximately 3.0 serial and 3.3 batched. The table gives
medians across class/shape combinations, each based on seven warm repeats.
These Linux CPU force-only measurements do not predict full orbit/library
runtime, GPU throughput, peak memory, or direct-versus-grid performance.
Grid construction and interpolation remain a separate follow-up.
