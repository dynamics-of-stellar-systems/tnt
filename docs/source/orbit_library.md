# Orbit library

An orbit sampler generates the grid of phase-space initial conditions that
get integrated and weighted against kinematic data as part of
[model search](model_search.md). `orbit_library_settings.orbit_sampler.type`
names one of three registered samplers (`tnt.orbit_library`).

This part of TNT is under active development. Orbit sampling and standalone
single-orbit integration are implemented. Production library assembly and
weighting (`Potential.generate_orbit_library`) remain unfinished -- see
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
used by TNT MGEs) and composites of them are supported. Nonconstant parameter
objects, rotating Monari bars, arbitrary custom and transformed potentials
raise explicitly. The numerical kernel supports JIT and differentiation
through initial conditions; sampling preparation remains eager, and hard
histogram bins do not gain useful coordinate derivatives merely by using JAX.

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
speed, not yet supported here) -- the construction and its application to
an integrated `OrbitLibrary` aren't implemented yet, since `OrbitLibrary`
assembly itself isn't.

## What's implemented today

The retained `OrbitLibrary` contract now contains named per-bundle
contribution arrays, stable zero-based bundle IDs, unitful reference periods,
and integration diagnostics. Raw trajectories are separate `OrbitTrajectory`
objects, with explicit times, positions, velocities, and units. Initial
conditions use `OrbitInitialConditions` and retain a size-one member axis;
only one orbit per bundle is supported in this first integration milestone.
These types establish the storage boundary; contribution production and
library assembly are implemented in subsequent steps.

Production execution will reduce a bounded number of temporary trajectories,
optionally save selected trajectories to separate files, and release the raw
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
- **Not implemented**: `build_orbit_sampler`/`build_orbit_dithering` (no
  configuration-to-sampler wiring yet); `AbstractOrbitDithering`'s own
  per-bundle dithering (declared, not consumed by any sampler); the
  counter-rotating mirror's actual construction; library assembly and
  `Potential.generate_orbit_library`.
