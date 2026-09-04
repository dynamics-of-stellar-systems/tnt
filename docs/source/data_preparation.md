# Data preparation

TNT combines photometric (MGE) and observational (kinematic, population, or
proper-motion) data. Getting their relative orientation right on the sky is
a data preparation step: an MGE declares its own `major_axis_pa` (in
[`MGEs`](configuration.md)) and a spatial binning declares its own
`y_axis_pa` (in [`spatial_binnings`](configuration.md)). This page covers
what to set them to.

## MGE `major_axis_pa`

`major_axis_pa` should be the on-sky position angle (north through east) of
the MGE's photometric major axis. If the MGE was fit with `mgefit`, this is
`f.pa` if `f = mgefit.find_galaxy(...)` -- not `f.theta`, the different,
internal angle that `sectors_photometry` itself takes as an argument.
`find_galaxy` only calls `f.pa` astronomical because it assumes the input
image's y-axis is aligned with true north; if it isn't, correct `f.pa` for
the image's own orientation before using it as `major_axis_pa`.

## `y_axis_pa`, and a fixed x-axis

A spatial binning's grid has two perpendicular axes, but only one position
angle is declared: `y_axis_pa`, the on-sky PA of the grid's positive y-axis.
TNT fixes the positive x-axis to be 90 degrees east of the positive y-axis
(e.g. x points east when y points north). A `bins_file` built with the
opposite parity should be flipped along its x-axis (with `min_x` adjusted to
match) before use -- check the sign convention the specific pipeline used
rather than assuming either.

## An MGE with no measured PA

If an MGE's photometric PA has not been measured, it still needs to be
approximated to relate that MGE to any observational data at all -- there is
no configuration-free default. One practical option is to estimate it from
the observational data itself: collapse an IFU cube into a white-light image,
which typically carries WCS coordinates, and fit a PA from north the same way
one would from any other photometric image.
