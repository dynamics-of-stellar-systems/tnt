# Orbit library

An orbit sampler generates the grid of phase-space initial conditions that
get integrated and weighted against kinematic data as part of
[model search](model_search.md). `orbit_library_settings.orbit_sampler.type`
names one of three registered samplers (`tnt.orbit_library`).

This part of TNT is under active development. Orbit sampling, standalone
single-orbit integration, and bounded-memory assembly with explicit reducers
are implemented. Automatic observational projection and model-search wiring
(`Potential.generate_orbit_library`) and weighting remain unfinished -- see
[What's implemented today](#whats-implemented-today).

## Single-orbit integration

`integrate_orbit(potential, position, velocity, reference_period, key, settings)`
integrates an already-built Galax potential with Diffrax `Dopri8` and returns
a separate `OrbitTrajectory`. `OrbitIntegrationSettings` defaults to 200
reference periods, 50,000 samples, normalized accuracy `1e-5`, and a maximum
of one million adaptive steps. These numerical defaults need calibration on
realistic galaxy models; they are not a certified energy-error target.

The solve covers `orbital_periods * reference_period`. The caller supplies
the clock explicitly; it is not estimated from an individual launch radius.
The reference convention is DYNAMITE's refined tube-boundary clock per energy
shell, shared across stationary and tube populations. A reference period is
not a measured recurrence period for a general triaxial orbit.

`XZGridFromBoundaryOrbitSampler.generate_initial_conditions(potential)`
returns those clocks alongside the launches. On inward-propagated irregular
shells it still performs the inner search needed for each shell's clock;
outer boundary searches remain skipped. The older `generate_ics` array-only
API retains its sampling behavior and does not request those extra clocks.

The stationary and origin samplers' `generate_initial_conditions` require an
explicit unitful `reference_periods` vector of shape `(nE,)`, on the same energy
grid. For example, take every `nI2*nI3`-th period from the boundary population's
per-bundle clocks. They never silently substitute their launch-radius periods.

Sampling follows DYNAMITE: for `N` samples and duration `D`, sample spacing is
`D/(N+4)` and the first sample is at `(1+u)` spacings, with `u` drawn from the
explicit JAX key. Adaptive solver steps are not retained as histogram samples.
The exact integration endpoint is evaluated separately for diagnostics.
`Dopri8` is an 8/7 method, not DYNAMITE's DOP853; convergence rather than
identical adaptive steps is the comparison target.

The solve normalizes positions and velocities using a launch length scale
and its reference-period velocity scale. Tolerances therefore do not depend
on whether equivalent coordinates are expressed in pc/yr or kpc/Myr.
Diagnostics report solver success, adaptive step counts, and maximum/final
energy drift divided by the squared characteristic speed. This avoids
division by the initial energy, which can be zero by convention. Maximum
drift includes the saved samples and endpoint, not all internal solver stages.
An unsuccessful trajectory must not be reduced or saved as a valid orbit.
No automatic retry or substitute orbit is used.

Only TNT's curated static Galax potentials (including the Gaussian components
used by TNT MGEs) and native `CompositePotential` compositions are supported.
Nonconstant parameter
objects, rotating Monari bars, arbitrary custom and transformed potentials
raise explicitly. The numerical kernel supports JIT and differentiation
through initial conditions; sampling preparation remains eager, and hard
histogram bins do not gain useful coordinate derivatives merely by using JAX.

Establish float64 precision before importing Galax, normally by importing TNT
first. Galax can create constant arrays during import; enabling JAX float64
later does not promote existing float32 constants. The scientific tests set
the precision before importing Galax, as in the existing sampler test suite.

Numerical regression tests cover 200-period harmonic motion, an eccentric
Kepler ellipse over 20 periods, gradients through initial conditions and a
potential parameter, equivalent units, zero-energy launches, and explicit
solver failures. An integration test also reduces a nested triaxial Gaussian
plus central-mass potential into normalized radial fractions. These checks do
not calibrate the default production accuracy or replace a comparison of
long-time orbit statistics against DYNAMITE's compiled integrator. The
existing DYNAMITE comparison below covers start-space sampling only.

## Library assembly and optional trajectories

`build_orbit_library` takes an already-built Galax potential, typed initial
conditions, named reducer functions, an explicit JAX key, and integration
settings. It integrates one orbit at a time. A reducer consumes a temporary
`OrbitTrajectory` and returns one orbit's contribution array, without a bundle
axis. The builder stacks those results into `OrbitLibrary.contributions` and
retains diagnostics and reference periods. It does not retain raw trajectories.
Reducers own their bin definitions and units and must return finite,
consistently shaped arrays. At least one reducer is required; there is no
placeholder observational result if concrete projection is unavailable.

For example, given a Galax `potential` and an `XZGridFromBoundaryOrbitSampler`
named `sampler`, this produces an illustrative spatial fraction per bundle:

```python
from pathlib import Path

import jax
import jax.numpy as jnp
from tnt.orbit_library import OrbitIntegrationSettings, build_orbit_library

starts = sampler.generate_initial_conditions(potential)

def positive_x_fraction(trajectory):
    return jnp.mean(trajectory.positions.ustrip("kpc")[:, 0] > 0)

libraries = build_orbit_library(
    potential,
    starts,
    {"positive_x_fraction": positive_x_fraction},
    jax.random.key(42),
    OrbitIntegrationSettings(orbital_periods=200),
    mass_scales=(1.0, 2.0),
    trajectory_directory=Path("orbit-trajectories"),
    save_bundle_ids=(0,),
)
library = libraries[1.0]
```

The illustrative fraction is not a replacement for a scientific kinematics
projection. Concrete kinematics classes still own viewing geometry, spatial
apertures, instrument blurring, and velocity-distribution transformations.
Their `design_matrix` boundary will consume retained contributions rather than
requiring all raw trajectories at once. These scientific implementations,
their connection to `ModelIterator`, and NNLS weight solving remain deferred.

The return value maps each requested positive mass scale to its library.
For scale `s`, the same temporary positions, velocities multiplied by
`sqrt(s)`, and times divided by `sqrt(s)` are passed to each reducer. No
integration is repeated, and no already-binned velocity counts are rescaled.
This requires the same spatial shape with every component's mass scaled
together. The diagnostics' dimensionless energy errors remain invariant.

Only requested bundle IDs are written to separate
`bundle-<id>-member-0.npz` files. `OrbitLibrary.trajectories` contains small
`TrajectoryReference` records with absolute paths, not coordinate arrays.
Files always contain the base-mass trajectory; `potential_mass_scale` on each
library states the velocity/time conversion needed to compare them with its
contributions. Existing files are rejected before integration and exclusively
created during writing. The builder does not load them for weight solving.

The NPZ record contains times, positions, velocities, the original launch
coordinates, and a JSON `metadata` scalar. Metadata records the intrinsic
Cartesian frame, units, reference period, bundle/member IDs, Dopri8 settings,
the exact Galax potential's components/constant parameters/constants, a
potential fingerprint, diagnostics, and the per-orbit random key. Load with
`numpy.load(path, allow_pickle=False)` and decode the metadata with `json.loads`.
This is diagnostic output, not a resumable model-search checkpoint format.

Bundle IDs are zero-based and stable. Folding the supplied key with each ID
preserves that orbit's sampling when a subset is integrated. No files are
written by default. Solver failure or nonfinite reducer output raises
`OrbitIntegrationError`; no substitute contribution is returned. A later
failure can leave previously completed selected trajectory files on disk;
they remain individually identified diagnostic artifacts.

## Orbit samplers

Every sampler shares the same fields: `rmin`/`rmax` (an explicit length
`Quantity` pair bounding a logarithmic grid of `nE` energy shells, each
shell's energy read off the potential along the `+x` axis), and `nI2`/`nI3`,
two further grid dimensions whose meaning depends on the sampler (below).
`n_bundles()` is `nE * nI2 * nI3` for all three. Field names and roles
match DYNAMITE's own (`orbitstart_f.f90`'s `nEner`/`nI2`/`nI3`) exactly:
`nI2` is always the (first) angular grid count, `nI3` always the second
non-energy dimension -- a second angle for `StationaryGrid`, a radial
count for the two `(x, z)`-plane samplers.

All samplers require `0 < rmin < rmax` and positive grid counts.
`XZGridFromBoundary` additionally requires `nI2 >= 4`; the simpler
samplers support a single angular row.

### `StationaryGrid` -- box orbits

Box orbits only, launched from rest exactly on the equipotential surface
(Schwarzschild 1979). `nI2`/`nI3` are an open, bin-centred grid of
`theta, phi in (0, pi/2)` -- never exactly `0` or `pi/2`, which would
respectively collapse every `phi` to one duplicate point on the `z`-axis, or
launch an orbit exactly in the equatorial plane (able to pass through the
centre of a cuspy potential).

### `XZGridFromOrigin` -- tube orbits, uniform grid

Orbits confined to the `(x, z)` plane, launched with velocity purely along
`y` from energy conservation (van den Bosch et al. 2008, MNRAS 385, 647,
sec. 4.3). `nI2` is an open, bin-centred grid of `theta in (0, pi/2)`; at
each `theta`, `nI3` orbits span `[r_floor, r_outer(theta)]` -- `r_floor` a
fixed near-origin floor (a cuspy or BH-dominated potential has
`v_y -> infinity` as `r -> 0`), `r_outer` the equipotential radius in that
direction -- via the "nearly closed" fractional spacing `(k - 0.9) /
(nI3 - 0.8)`, since landing exactly on the equipotential is itself
degenerate (`v_y = 0` there). No search for where tube-orbit support
actually begins or ends: every shell samples the same fixed range, faster
than `XZGridFromBoundary`, whose boundary search adapts the radial grid
to the located orbit-family transitions.

### `XZGridFromBoundary` -- tube orbits, DYNAMITE-matching boundary search

The same `(x, z)`-plane population as `XZGridFromOrigin`, but for a
*regular* energy shell, `nI3`'s radii are bracketed between two located
edges (`boundin`, `boundmid` -- the box/short-axis-tube and
short-axis-tube/long-axis-tube boundaries) rather than spanning the full
range -- van den Bosch et al. (2008) sec. 4.3's scheme, ported directly
from DYNAMITE's own Fortran reference (`orbitstart_f.f90`), matching its
early-stopping search and per-step bracket formulas. Not every shell has a
clean four-region structure: a shell comes back *irregular* when the
`boundin` search hits its own bracket edge, or a long-axis tube is never
found along the sweep. Irregularity isn't purely local to one shell either
-- walking outermost-to-innermost, from the first locally irregular shell
on, every shell inward of it delegates directly to `XZGridFromOrigin`'s own
uniform-grid procedure, without running its own boundary search at all.

### Numerical checks and comparison with DYNAMITE

Every sampler validates the equipotential bracket, root-solver result, and
residual. The residual tolerance is `1e-7` relative to the potential change
across the bracket, so a zero-energy shell remains valid. An unsupported
bracket or failed root search raises `EquipotentialSearchError`.

The boundary sampler also rejects unusable tube-width searches. Numerical
integration failure or exhaustion of the step-attempt budget before either
the crossing target or time ceiling is reached makes a trial unusable. If
the minimization finds no usable candidate, the public sampler raises
`EquipotentialSearchError`. Physical irregularity retains the documented
origin-grid delegation above; a numerical failure is reported explicitly.

A comparison against DYNAMITE's compiled `orbitstart` binary covers four
NGC6278-based viewing geometries (oblate, mild/strong triaxial, near-prolate)
and six energy shells each, from `0.01` to `100 kpc`. Box-orbit median
relative errors are approximately `2.9e-7` to `4.0e-7`, with maximum errors
of `4.2e-6` to `4.6e-6`. Boundary-sampled tube-orbit median relative errors
are approximately `1.1e-5` to `7.8e-5`; the maximum is approximately
`3.1%` in each geometry, at an innermost-shell near-origin point. The median
agreement therefore does not describe every sampled point.

The [public comparison](https://github.com/dynamics-of-stellar-systems/tnt-dynamite-comparison/tree/0df175616073749a5b1453acb326733a62d4dbbb/orbit-start-spaces)
preserves resolved inputs, raw DYNAMITE output, saved arrays, a reproduction
script, and a notebook. It pins TNT
`c1b63564866f41855b01a0fb56701792e7215cbb` and DYNAMITE
`0bd10a9586936bf6b9bcfdcd3679477d812a01d0`.

## Counter-rotating orbits

Every `AbstractOrbitSampler` declares `_add_reverse_copies`: `True` for
`XZGridFromOrigin`/`XZGridFromBoundary` (their orbits have a definite sense
of circulation, and allowing net rotation in the fit needs the opposite
sense too), `False` for `StationaryGrid` (a box orbit has no such sense to
mirror). This isn't a configuration option -- which orbit family a sampler
produces isn't a free modeling choice independent of picking the sampler
type.

For a time-independent potential, a tube orbit's counter-rotating partner
is free to construct once integrated: negating every stored velocity
component at a fixed recorded position gives another exact solution of the
same equations of motion, with no re-integration needed. This is *not*
generally true for a time-dependent potential (e.g. a rotating bar/pattern
speed, not yet supported here). Construction and reduction of the reverse
population remain unimplemented; the current builder returns only the
supplied population, and must not be treated as a complete rotation-capable
Schwarzschild library without this follow-up.

## What's implemented today

The retained `OrbitLibrary` contract now contains named per-bundle
contribution arrays, stable zero-based bundle IDs, unitful reference periods,
and integration diagnostics. Raw trajectories are separate `OrbitTrajectory`
objects, with explicit times, positions, velocities, and units. Initial
conditions use `OrbitInitialConditions` and retain a size-one member axis;
only one orbit per bundle is supported in this first integration milestone.
The builder produces contributions through explicit reducer functions;
concrete observational projection implementations remain separate follow-up.

Library execution reduces one temporary trajectory at a time,
optionally saves selected trajectories to separate files, and releases the raw
samples. Aggregated contributions remain independent per fitted bundle.
An aperture histogram must retain its fraction of the total sampled time,
including time outside the aperture, rather than being normalized to one.
Concrete projection formulas remain owned by the kinematics classes.

Already-binned velocity contributions cannot be generically mass-rescaled.
`OrbitLibrary.rescaled` explicitly raises: contributions at different mass
scales must be calculated while the temporary trajectories remain available.
Any future histogram rebinning approximation needs its own explicit contract.

- **`StationaryGrid`, `XZGridFromOrigin`, `XZGridFromBoundary`**: `generate_ics`
  and `n_bundles` both work, producing bundle-centre initial conditions from
  an already-built `galax` potential. Configuration validation
  (`_validate_orbit_sampler`) checks this same schema.
- **Builders**: `build_orbit_sampler` consumes the same validated configuration
  schema. `build_orbit_dithering` constructs `CubicOrbitDithering` only for
  `n_dither=1`; larger values raise `NotImplementedError` at construction.
- **Not implemented**: per-bundle dithering beyond one; the
  counter-rotating mirror's actual construction; automatic observational
  projection, model-search integration, and
  `Potential.generate_orbit_library`.
