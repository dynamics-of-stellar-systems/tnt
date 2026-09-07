# Data preparation

TNT combines several observational datasets into a single model of one
galaxy. It assumes each dataset has already been prepared so that:

- **Common origin.** Every dataset shares one spatial origin, which should
  coincide with the galaxy center.
- **Per-dataset orientation.** Each dataset may carry its own orientation,
  always given as a position angle in the standard astronomical convention
  (measured from north through east):
  - for an MGE, `major_axis_pa` in `[0, 180)` degrees -- the PA of the
    photometric major axis;
  - for any other dataset, `y_axis_pa` in `[0, 360)` degrees -- the PA of the
    *positive* y-axis of its spatial binning grid. TNT fixes the grid parity:
    the positive x-axis is 90 degrees east of the positive y-axis.
- **Galaxy rest frame.** Every kinematic dataset is in the galaxy rest frame,
  with the systemic velocity removed.

## Common origin

Every dataset shares one origin, which should coincide with the galaxy
center. In principle this is the minimum of the gravitational potential; in
practice it is approximated by an observational proxy. Use the photometric
center where it is well defined, and the kinematic center where it is not.

A spatial binning's `min_x`, `min_y`, `x_extent`, and `y_extent` are measured
from this origin, and any kinematic or population data referencing that
binning inherits it. Recenter data that was reduced on a different origin
before configuring it.

## MGE `major_axis_pa`

`major_axis_pa` should be the on-sky position angle (north through east) of
the MGE's photometric major axis.

If the MGE was fit with [`mgefit`](https://pypi.org/project/mgefit/), the
canonical fitting sequence is:

```python
import mgefit as mge

f = mge.find_galaxy(image)
s = mge.sectors_photometry(image, f.eps, f.theta, f.xpeak, f.ypeak)
m = mge.fit_sectors(s.radius, s.angle, s.counts, f.eps)  # m.sol: counts, sigma, q
```

`find_galaxy` defines `f.pa` as the standard astronomical PA (north through
east) of the major axis, assuming the image y-axis points north. Use `f.pa`
as `major_axis_pa` (wrapped into `[0, 180)`). Do not use `f.theta` -- that is
the image-frame angle (`f.pa = 270 - f.theta`) that `sectors_photometry`
consumes. If the image y-axis was not north, or east and west are swapped
from the standard orientation, `f.pa` is in a rotated or mirrored frame and
must be corrected first.

`fit_sectors` fits a single position angle, so an MGE built from it has no
twist and its `PA_twist` column is all zero. `fit_sectors_twist` fits a
per-Gaussian PA (`m.sol[3]`) in mgefit's image-frame angular coordinates,
which run opposite to astronomical PA -- that is the `270 -` in
`f.pa = 270 - f.theta`. TNT reads each component's on-sky PA as
`major_axis_pa + PA_twist` (north through east), so mgefit's per-Gaussian
angles must be negated to become `PA_twist`. For any other fitting tool,
confirm the same: `PA_twist` must increase north through east.

If you need a literature MGE that has no PA stored with it, the PA still has
to be approximated. One practical option is to estimate it from a white-light
image collapsed from an IFU cube, which usually carries WCS coordinates, and
fit a PA from north as you would from any other photometric image. Do not use
the kinematic PA as a substitute for the photometric PA. These can genuinely
differ in a triaxial galaxy, and that misalignment is a signal the model is
meant to fit, not a nuisance to define away.

## Spatial binning `y_axis_pa`

Every non-MGE dataset takes its geometry from a spatial binning. Its
orientation is `y_axis_pa` in `[0, 360)` degrees, the on-sky PA of the
binning grid's *positive* y-axis. TNT fixes the grid parity: the positive
x-axis is 90 degrees east of the positive y-axis (e.g. x points east when y
points north).

Data of the opposite parity -- positive x pointing *west* of positive y --
must be converted before it is given to TNT:

- reverse the `bins_file` array along its first (`npix_x`) axis,
  `bins = bins[::-1, :]`, and set `min_x` to `-(min_x + x_extent)`;
- negate any x-directed vector quantity in the corresponding `data_file`
  (for example a proper-motion `vx`).

Check the sign convention the reduction pipeline actually used (for example
the sign of a FITS `CDELT1`) rather than assuming either parity.

## Kinematic data

**All kinematic data:** remove the systemic velocity so the data is in the
galaxy rest frame.

**Proper motions:** detailed preparation instructions will be added here. For
now, note that if the spatial binning was converted for parity as above, the
`vx` component must be negated to match.
