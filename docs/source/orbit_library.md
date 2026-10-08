# Orbit library

An orbit sampler generates the grid of phase-space initial conditions that
get integrated and weighted against kinematic data as part of
[model search](model_search.md). `orbit_library_settings.orbit_sampler.type`
names one of three registered samplers (`tnt.orbit_library`).

This part of TNT is under active development. Orbit *sampling* (this page)
is implemented; orbit *integration* and weighting
(`Potential.generate_orbit_library`, `OrbitLibrary` itself) are not -- see
[What's implemented today](#whats-implemented-today).

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
than `XZGridFromBoundary` but without its orbit-family completeness
guarantee.

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

Verified directly against a real run of DYNAMITE's compiled `orbitstart`
binary across four viewing geometries (oblate, mild/strong triaxial, near
prolate) and six energy shells each: box orbits agree with DYNAMITE to 5-6
significant figures, boundary-searched tube orbits to 3-4, in every shell
and geometry.

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

- **`StationaryGrid`, `XZGridFromOrigin`, `XZGridFromBoundary`**: `generate_ics`
  and `n_bundles` both work, producing bundle-centre initial conditions from
  an already-built `galax` potential. Configuration validation
  (`_validate_orbit_sampler`) checks this same schema.
- **Not implemented**: `build_orbit_sampler`/`build_orbit_dithering` (no
  configuration-to-sampler wiring yet); `AbstractOrbitDithering`'s own
  per-bundle dithering (declared, not consumed by any sampler); the
  counter-rotating mirror's actual construction; `OrbitLibrary` itself and
  `Potential.generate_orbit_library`.
