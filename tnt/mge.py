"""Multi-Gaussian Expansion (MGE) models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Self

import astropy.units as au
import equinox as eqx
import jax
import jax.numpy as jnp
from astropy.table import QTable
from jax.ops import segment_sum
from jax.scipy.special import erf
from unxt import Quantity

from tnt import quantity_conversions
from tnt.spatial_binnings import ProjectedBinning, SphericalGrid
from tnt.units import reference_unit, validate_dimension, validate_position_angle

# Half-open on-sky domain (degrees) of an MGE's ``major_axis_pa``: a major
# axis is an undirected line, so it is only defined modulo 180 degrees. Read by
# `tnt.configuration.validation` as static schema metadata.
MAJOR_AXIS_PA_DOMAIN_DEG: tuple[float, float] = (0.0, 180.0)


class MGEDeprojectionError(ValueError):
    """A deprojection has no solution, or violates TNT's ``0 < q <= p <= 1`` convention.

    Eager deprojection raises this error. The matching ``with_validity``
    methods return a scalar JAX boolean from the same numerical checks.
    """


def _rebased_unit(intensity_unit: au.UnitBase, length_unit: au.UnitBase) -> au.UnitBase:
    """Re-express surface intensity as ``X`` per ``length_unit**2``.

    Every length base is swapped for `length_unit` at its existing power, so
    e.g. ``Lsun / pc2`` re-based on ``kpc`` becomes ``Lsun / kpc2``.
    `uconvert`ing an intensity to this makes `AbstractMGE.get_projected_mass`'s
    per-pixel integral (carried out entirely in `coord_unit`, a
    `ProjectedBinning`'s own declared coordinate unit) dimensionally
    consistent regardless of which physical length unit `I` was declared in.
    """
    result = au.dimensionless_unscaled
    for base, power in zip(intensity_unit.bases, intensity_unit.powers, strict=True):
        replacement = length_unit if base.physical_type == "length" else base
        result = result * replacement**power
    return result


def _axial_ratios_valid(p: jax.Array, q: jax.Array) -> jax.Array:
    """Per-row finite intrinsic-axis convention, shared by eager diagnostics."""
    return jnp.isfinite(p) & jnp.isfinite(q) & (q > 0) & (q <= p) & (p <= 1)


def _check_axial_ratios(p: jnp.ndarray, q: jnp.ndarray) -> None:
    """Raise `MGEDeprojectionError` unless every component has ``0 < q <= p <= 1``.

    Catches both an unsolvable deprojection (`p`/`q` containing `nan`, which
    fails every comparison below) and a solution that's finite but violates
    TNT's intrinsic-axis convention.
    """
    valid = _axial_ratios_valid(p, q)
    if bool(jnp.all(valid)):
        return
    bad = jnp.asarray(~valid).nonzero()[0]
    details = ", ".join(
        f"component {i}: p={float(p[i])!r}, q={float(q[i])!r}" for i in bad
    )
    raise MGEDeprojectionError(
        "Deprojection violates TNT's 0 < q <= p <= 1 intrinsic-axis convention "
        f"(or has no real solution) for: {details}."
    )


def _deprojected_valid(model: Deprojected3DMGE) -> jax.Array:
    """Shared intrinsic-domain and density/width/mass representability checks."""
    p, q = model.p.ustrip(""), model.q.ustrip("")
    valid = jnp.all(_axial_ratios_valid(p, q))
    for quantity in (model.I, model.sigma):
        value = quantity.ustrip(quantity.unit)
        valid = valid & jnp.all(jnp.isfinite(value) & (value > 0))
    # These are the exact products used to construct the Galax Gaussians.
    mass = model.component_masses
    value = mass.ustrip(mass.unit)
    return valid & jnp.all(jnp.isfinite(value) & (value > 0))


def _guard_deprojection(
    candidate: Callable[..., tuple[Deprojected3DMGE, jax.Array]],
    arguments: tuple[Quantity, ...],
) -> tuple[Deprojected3DMGE, jax.Array]:
    """Probe without AD, then differentiate only a valid deprojection.

    Invalid results contain zeros in every leaf, and are never physical models.
    The probe may encounter undefined arithmetic, but is disconnected from AD.
    """
    shape = jax.eval_shape(lambda args: candidate(*args)[0], arguments)
    placeholder = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), shape)
    _, valid = candidate(*jax.tree.map(jax.lax.stop_gradient, arguments))
    result = jax.lax.cond(
        valid, lambda args: candidate(*args)[0], lambda _: placeholder, arguments
    )
    return result, valid


def _shape_numbers(*values: float | jax.Array) -> tuple[jax.Array, ...]:
    """Check scalar real structure, then use the configured JAX precision."""
    result = []
    for value in values:
        number = jnp.asarray(value)
        if number.ndim != 0 or not (
            jnp.issubdtype(number.dtype, jnp.integer)
            or jnp.issubdtype(number.dtype, jnp.floating)
        ):
            raise TypeError("MGE shape coordinates must be real scalars.")
        result.append(jnp.asarray(number, dtype=float))
    return tuple(result)


def _checks_valid(checks: Mapping[str, jax.Array]) -> jax.Array:
    """Combine the numerical predicates also used for eager diagnostics."""
    return jnp.all(jnp.stack(tuple(checks.values())))


def _checked_shape_conversion(candidate: Callable, arguments: tuple) -> Any:
    """Raise the first failed authoritative check at the eager boundary."""
    result, checks = candidate(*arguments)
    for message, valid in checks.items():
        if not bool(valid):
            raise MGEDeprojectionError(message)
    return result


def _guard_shape_conversion(
    candidate: Callable, arguments: tuple
) -> tuple[Any, jax.Array]:
    """Detach numerical checks and differentiate only accepted conversions."""
    shape = jax.eval_shape(lambda args: candidate(*args)[0], arguments)
    placeholder = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), shape)
    _, checks = candidate(*jax.tree.map(jax.lax.stop_gradient, arguments))
    valid = _checks_valid(checks)
    result = jax.lax.cond(
        valid, lambda args: candidate(*args)[0], lambda _: placeholder, arguments
    )
    return result, valid


def _triaxial_geometry_valid(
    theta: jax.Array,
    phi: jax.Array,
    psi: jax.Array,
    q_obs: jax.Array,
    p: jax.Array,
    q: jax.Array,
    u: jax.Array,
) -> jax.Array:
    """Bound inversion roundoff and verify the projected covariance independently.

    The 3x3 map sends intrinsic diagonal covariance to the two sky variances
    and their covariance. Its singular values bound sensitivity to roundoff.
    Require the worst-case relative error in the smallest intrinsic variance
    to be <= 50 sqrt(eps); a small residual alone cannot detect ill conditioning.
    """
    st, ct = jnp.sin(theta), jnp.cos(theta)
    sp, cp = jnp.sin(phi), jnp.cos(phi)
    x = jnp.stack([sp, -cp, jnp.zeros_like(sp)])
    y = jnp.stack([ct * cp, ct * sp, -st])
    matrix = jnp.stack([x * x, y * y, x * y])
    singular = jnp.linalg.svd(matrix, compute_uv=False)
    eps = jnp.finfo(matrix.dtype).eps
    # q**2 is the smallest variance relative to the largest one.
    argument_scale = 1 + jnp.maximum(
        jnp.abs(theta), jnp.maximum(jnp.abs(phi), jnp.abs(psi))
    )
    error_bound = eps * argument_scale * singular[0] / singular[-1] / q**2
    intrinsic = jnp.stack([jnp.ones_like(p), p**2, q**2], axis=-1) / u[:, None] ** 2
    projected = intrinsic @ matrix.T
    cs, sn = jnp.cos(psi), jnp.sin(psi)
    expected = jnp.stack(
        [
            sn**2 + q_obs**2 * cs**2,
            cs**2 + q_obs**2 * sn**2,
            (q_obs**2 - 1) * sn * cs,
        ],
        axis=-1,
    )
    tolerance = 50 * jnp.sqrt(eps)
    residual = jnp.max(jnp.abs(projected - expected), axis=-1)
    return jnp.all(
        jnp.isfinite(error_bound)
        & (error_bound <= tolerance)
        & jnp.isfinite(u)
        & (u > 0)
        & (u <= 1)
        & (residual <= tolerance * q_obs**2)
    )


# float64 noise floor for the de Zeeuw & Franx weights `w1`/`w2`/`w3` in
# `AbstractMGE.triaxial_viewing_angles` (widened there to `16 * eps` at lower
# precision): non-negative in exact arithmetic inside the valid domain, and
# accepted `u` is separated from the singular boundaries, so an excursion this
# small is roundoff and is clamped; a larger one is a genuinely degenerate
# geometry and is rejected.
_TRIAXIAL_WEIGHT_ATOL = 1e-9

# `AbstractMGE.viewing_angles_from_T_Tmaj_Tmin`'s own round-trip accuracy
# check: `(T, T_maj, T_min)` divide by `1 - p**2` and `p**2 - q**2`,
# so roundoff in recovered `(p, q, u)` can move the shape coordinates
# substantially even for an accepted interior compression (see that
# function's own docstring, and `_p_q_u_from_T_Tmaj_Tmin`).
#
# Re-audit finding (PR 67): a purely *absolute* tolerance here (the first
# version of this check) is blind to a small requested coordinate -- at
# float32, `T_maj = 0.02` recovering `~0.054` (a 168% relative change) still
# passed an absolute-only bound of `~0.0345`. Calibrated against measured
# round-trip drift the same way as the oblate q_min check in this module
# (PR 68): an ordinary interior point's drift stays below `~6e-6` relative
# and `~6e-7` absolute at float32 (and far tighter at float64), while every
# flagged bad case measured `32%-168%` relative. A combined
# `atol + rtol * |target|` bound (not relative alone) is needed because
# `T`/`T_maj`/`T_min` are inclusively bounded in `[0, 1]` and a requested
# coordinate can legitimately be exactly `0`, where a purely relative test
# is either meaningless or infinite.
_TMAJMIN_ROUNDTRIP_REL_TOL_FACTOR = 50.0
_TMAJMIN_ROUNDTRIP_ABS_TOL_FACTOR = 1e2

# `AbstractMGE.inclination_from_q_min`'s own round-trip accuracy check.
# `deproject_oblate`'s `q_intr = sqrt(q_obs**2 - cos(i)**2) / sin(i)` (used by
# both the native `inclination` parameterization and, via
# `q_min_from_inclination`, this round trip) subtracts two nearly equal
# quantities whenever the requested `q_min` is small relative to `q_obs` --
# `cos(i)**2` is then close to `q_obs**2` by construction (see
# `inclination_from_q_min`'s own `cos2_i` formula) -- so a tiny rounding error
# already present in `i` at the working precision can become a large
# *relative* error in the recovered `q_min`. A relative tolerance (not
# absolute, unlike `_TRIAXIAL_WEIGHT_ATOL`) is the right test here
# specifically because the failure mode scales with how small the requested
# `q_min` itself is, not with its absolute size. Scaled by `sqrt(eps)` rather
# than `eps` itself for the same headroom-vs-sensitivity reason as
# `triaxial_viewing_angles`'s own `4 * sqrt(eps)` margin. Calibrated against
# measured round-trip drift, not guessed: an ordinary configuration (q_obs up
# to 0.99, q_min down to 0.02) stays within ~5e-3 relative drift at float32
# and ~2e-12 at float64, while every case the PR-68 audit flagged measured
# 0.4%-6.2% -- `50 * sqrt(eps)` (~1.7% at float32, ~7.5e-7 at float64) sits
# comfortably between the two at both precisions.
_QMIN_ROUNDTRIP_TOL_FACTOR = 50.0


def _triaxial_intrinsic_axis_ratios(
    theta_r: jnp.ndarray,
    phi_r: jnp.ndarray,
    psi_r: jnp.ndarray,
    q_obs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Intrinsic axial ratios ``(p, q)`` and compression ``u`` from viewing angles.

    The general triaxial relation of de Zeeuw & Franx (1989) / Cappellari (2002)
    eqs. 6-8 / van den Bosch et al. (2008) eqs. 6-9: given the observed axial
    ratio ``q_obs`` of an ellipse and the three viewing angles (already in
    radians, with ``psi_r`` already including any component ``PA_twist``), solve
    for the intrinsic ``p = B/A``, ``q = C/A`` and the scale-length compression
    ``u = sigma_observed / sigma_intrinsic`` (``= a'/a``, the projected-to-intrinsic
    major-axis ratio). ``u`` is only 1 at special viewing angles.

    Every argument may be a scalar or an array (one entry per Gaussian). No
    validity check here -- ``p``/``q`` can come back ``nan`` for a viewing
    geometry with no real solution; the caller checks (`_check_axial_ratios`,
    or the ``pqu`` converter's own bounds).
    """
    delta = 1 - q_obs**2

    cos_theta, sin_theta = jnp.cos(theta_r), jnp.sin(theta_r)
    sec_theta = 1 / cos_theta
    cot_phi = 1 / jnp.tan(phi_r)
    tan_phi = jnp.tan(phi_r)
    cos_phi, sin_phi = jnp.cos(phi_r), jnp.sin(phi_r)
    cos_psi, sin_psi = jnp.cos(psi_r), jnp.sin(psi_r)
    cos_2psi, sin_2psi = jnp.cos(2 * psi_r), jnp.sin(2 * psi_r)

    denom = (
        2
        * sin_theta**2
        * (delta * cos_psi * (cos_psi + cot_phi * sec_theta * sin_psi) - 1)
    )
    one_minus_q2 = (
        delta
        * (2 * cos_2psi + sin_2psi * (sec_theta * cot_phi - cos_theta * tan_phi))
        / denom
    )
    p2_minus_q2 = (
        delta
        * (2 * cos_2psi + sin_2psi * (cos_theta * cot_phi - sec_theta * tan_phi))
        / denom
    )

    q_intr = jnp.sqrt(1 - one_minus_q2)
    p_intr = jnp.sqrt(q_intr**2 + p2_minus_q2)
    u = jnp.sqrt(
        jnp.sqrt(
            p_intr**2 * cos_theta**2
            + q_intr**2 * sin_theta**2 * (p_intr**2 * cos_phi**2 + sin_phi**2)
        )
        / q_obs
    )
    # Circular projected rows have the exact spherical solution wherever
    # the viewing geometry is nonsingular. Do not round u above its bound.
    u = jnp.where(q_obs == 1, 1, u)
    return p_intr, q_intr, u


def _p_q_u_candidate(
    T: jax.Array, T_maj: jax.Array, T_min: jax.Array, q_obs: jax.Array
) -> tuple[tuple[jax.Array, ...], dict[str, jax.Array]]:
    """Quenneville et al. shape algebra and its shared numerical domain."""
    eps = jnp.finfo(T.dtype).eps
    o2 = q_obs**2
    den = 1 - (1 - T) * T_min - o2 * T * T_maj
    q2 = 1 - (1 - o2) / den
    p2 = 1 - T * (1 - q2)
    u2 = 1 - T_maj * (1 - p2)
    checks = {
        "(T, T_maj, T_min) must be finite coordinates in [0, 1].": (
            jnp.all(jnp.isfinite(jnp.stack([T, T_maj, T_min])))
            & jnp.all(
                (jnp.stack([T, T_maj, T_min]) >= 0)
                & (jnp.stack([T, T_maj, T_min]) <= 1)
            )
        ),
        "(T, T_maj, T_min) denominator too close to zero to deproject "
        "reliably at this numerical precision.": jnp.abs(den) >= 4 * eps,
        "(T, T_maj, T_min) has no valid triaxial deprojection: requires a "
        "strictly positive intrinsic minor-axis ratio (q^2 > 0).": (
            (q2 > 0) & (q2 <= p2) & (p2 <= 1) & (u2 > 0) & (u2 <= 1)
        ),
    }
    return (jnp.sqrt(p2), jnp.sqrt(q2), jnp.sqrt(u2)), checks


def _p_q_u_from_T_Tmaj_Tmin(
    T: float, T_maj: float, T_min: float, q_obs: float
) -> tuple[float, float, float]:
    """Eager Quenneville et al. eq. 7 conversion, with shared JAX checks."""
    result = _checked_shape_conversion(
        _p_q_u_candidate, _shape_numbers(T, T_maj, T_min, q_obs)
    )
    return tuple(float(value) for value in result)


def _T_Tmaj_Tmin_candidate(
    p: jax.Array, q: jax.Array, u: jax.Array, q_obs: jax.Array
) -> tuple[tuple[jax.Array, ...], dict[str, jax.Array]]:
    """Quenneville et al. eqs. 3-4 and coordinate singularity checks."""
    p2, q2, u2, o2 = p**2, q**2, u**2, q_obs**2
    denominators = jnp.stack([1 - q2, 1 - p2, p2 - q2])
    result = (
        (1 - p2) / denominators[0],
        (1 - u2) / denominators[1],
        (u2 * o2 - q2) / denominators[2],
    )
    checks = {
        "(p, q, u) is too close to a coordinate singularity to convert to "
        "(T, T_maj, T_min).": (
            jnp.all(jnp.abs(denominators) >= jnp.finfo(p.dtype).eps)
            & jnp.all(jnp.isfinite(jnp.stack(result)))
        )
    }
    return result, checks


def _T_Tmaj_Tmin_from_p_q_u(
    p: float, q: float, u: float, q_obs: float
) -> tuple[float, float, float]:
    """Eager inverse shape algebra using the same coordinate checks."""
    result = _checked_shape_conversion(
        _T_Tmaj_Tmin_candidate, _shape_numbers(p, q, u, q_obs)
    )
    return tuple(float(value) for value in result)


class AbstractMGE(eqx.Module):
    """Shared structure and behaviour for MGE models.

    Each Gaussian component is described by its peak intensity ``I``, width ``sigma``,
    axial ratio ``q``, and position-angle twist ``PA_twist``, each stored in its own
    declared unit. Subclasses fix which physical dimension ``I`` represents by setting
    `_intensity_dimension` to the corresponding `tnt.units._REFERENCE_UNITS` key
    (`light_surface_brightness` or `mass_surface_density`). Not meant to be
    instantiated directly -- use `LightMGE` or `MassMGE`.

    ``I`` is a physical surface density (e.g. Lsun/pc2, Msun/pc2), independent
    of any assumed distance. File-loaded ``sigma`` starts in angular units;
    `angular_to_physical` converts it to a distance-dependent physical width.
    Converted MGEs and directly constructed MGEs can carry physical widths.

    ``major_axis_pa`` is the on-sky position angle of the MGE's major axis --
    a standard astronomical PA, measured from north through east, in
    ``[0, 180)`` degrees (a major axis is an undirected line).
    """

    _intensity_dimension: ClassVar[str]

    I: Quantity
    sigma: Quantity
    q: Quantity
    PA_twist: Quantity
    major_axis_pa: Quantity

    @classmethod
    def from_qtable(cls, table: QTable, major_axis_pa: Quantity) -> Self:
        """Build an MGE from a table, validating its columns and keeping their units.

        Each column's declared unit is checked for dimensional correctness
        against a fixed reference and then kept as-is -- no unit-system
        conversion happens here (see `tnt.units`' module docstring).

        Args:
            table: A table with columns ``I``, ``sigma``, ``q``, and ``PA_twist``, each
                carrying an astropy unit.
            major_axis_pa: The MGE major-axis PA the ``PA_twist`` column is
                measured from (see the class docstring). Not stored in the
                ECSV; supplied by configuration. A finite scalar angle in
                ``[0, 180)`` degrees (a major axis is an undirected line).

        Returns:
            An MGE with each column in its own declared unit.

        Raises:
            ValueError: If the ``I`` column's unit isn't this MGE kind's
                surface-intensity dimension, a ``sigma``/``PA_twist`` unit
                isn't an angle, any ``q`` value is outside ``(0, 1]``, or
                `major_axis_pa` isn't a finite scalar angle in ``[0, 180)``
                degrees.
        """
        validate_dimension(table["I"].unit, cls._intensity_dimension, "MGE I column")
        validate_dimension(table["sigma"].unit, "angle", "MGE sigma column")
        validate_dimension(table["PA_twist"].unit, "angle", "MGE PA_twist column")
        validate_position_angle(
            major_axis_pa,
            minimum_deg=MAJOR_AXIS_PA_DOMAIN_DEG[0],
            maximum_deg=MAJOR_AXIS_PA_DOMAIN_DEG[1],
            path="MGE major_axis_pa",
        )

        columns = {
            name: Quantity.from_(table[name])
            for name in ("I", "sigma", "q", "PA_twist")
        }

        q = columns["q"].ustrip("")
        if not bool(jnp.all((q > 0) & (q <= 1))):
            raise ValueError(f"q must satisfy 0 < q <= 1 for all components, got {q}")

        return cls(**columns, major_axis_pa=major_axis_pa)

    @classmethod
    def read(cls, path: str | Path, major_axis_pa: Quantity) -> Self:
        """Read an MGE from an ECSV file, keeping each column's declared unit.

        Args:
            path: Path to the ECSV file.
            major_axis_pa: The MGE major-axis PA the ``PA_twist`` column is
                measured from (see the class docstring). Not stored in the
                ECSV; supplied by configuration. A finite scalar angle in
                ``[0, 180)`` degrees.

        Returns:
            An MGE with each column in its own declared unit.

        Raises:
            ValueError: As for `from_qtable`, including if `major_axis_pa`
                isn't a finite scalar angle in ``[0, 180)`` degrees.
        """
        table = QTable.read(path, format="ascii.ecsv")
        return cls.from_qtable(table, major_axis_pa)

    def rescaled(self, factor: Quantity) -> Self:
        """Multiply `I` by a dimensionless factor, keeping every other field.

        Used for `TriaxialMassMGEPotential`'s `mge_mass_scale`, a normalization on
        top of an otherwise-fixed mass map (see
        `tnt.potential.triaxial_mge.TriaxialMassMGEPotential`).

        Args:
            factor: The multiplicative factor, either a single value applied
                to every component, or an array with one value per Gaussian
                component.

        Returns:
            An MGE of the same kind with ``I = self.I * factor``.
        """
        return type(self)(
            I=self.I * factor,
            sigma=self.sigma,
            q=self.q,
            PA_twist=self.PA_twist,
            major_axis_pa=self.major_axis_pa,
        )

    def angular_to_physical(self, distance: Quantity) -> Self:
        """Convert `sigma` from angular to physical (length) units.

        `I` is already a physical surface density (e.g. Lsun/pc2, Msun/pc2),
        so it's unaffected by `distance` and carried over unchanged, along
        with `q` (dimensionless) and `PA_twist` (an orientation angle, not a
        spatial size).

        Works for any angular unit `sigma` was declared in: it goes through
        the radian small-angle factor, so this doesn't depend on `sigma`
        already being in radians.

        Args:
            distance: The distance to the object.

        Returns:
            A new MGE with `sigma` in `distance`'s unit; `I` unchanged.
        """
        sigma_physical = quantity_conversions.angular_to_physical(self.sigma, distance)

        return type(self)(
            I=self.I,
            sigma=sigma_physical,
            q=self.q,
            PA_twist=self.PA_twist,
            major_axis_pa=self.major_axis_pa,
        )

    def get_projected_mass(self, binning: ProjectedBinning) -> Quantity:
        """This MGE's total in each bin of a `ProjectedBinning`.

        Each Gaussian component's surface density is integrated over every
        pixel of `binning`'s aperture grid -- exactly in one direction via
        `erf`, and via fixed-order Gauss-Legendre quadrature in the other,
        since a rotated 2D Gaussian's integral over an axis-aligned
        rectangle has no closed form in general (Cappellari 2002 appendix
        B, eqs. B6-B7) -- and the resulting per-pixel totals are then summed
        into their assigned bins. Pixels with bin ID 0 (unbinned, see
        `ProjectedBinning`) don't contribute to any bin.

        Each Gaussian's on-sky major-axis PA is ``major_axis_pa + PA_twist``
        (north through east); it is placed in `binning`'s own grid relative to
        `binning`'s `y_axis_pa`, using the fixed parity `ProjectedBinning`
        assumes (its positive x-axis is 90 degrees east of its positive
        y-axis -- see its docstring).

        Args:
            binning: The projected-plane aperture grid and pixel-to-bin
                assignment to integrate this MGE over. Its coordinates
                (`min_x`, `min_y`, `x_extent`, `y_extent`) must be physical
                (length), matching `sigma` post-`angular_to_physical`.

        Returns:
            A `Quantity` of shape ``(n_bins,)`` giving each bin's total,
            ordered by increasing bin ID (bin 1 first). ``n_bins`` is
            `binning`'s largest bin ID.

        Raises:
            astropy.units.UnitConversionError: If `sigma` and `binning`'s
                coordinates aren't dimensionally consistent.
        """
        coord_unit = binning.min_x.unit
        I_unit = _rebased_unit(self.I.unit, coord_unit)

        # Add a leading components axis: everything below broadcasts to
        # (G, Nx, Q, Ny).
        shape = (-1, 1, 1, 1)
        sigma = self.sigma.ustrip(coord_unit).reshape(shape)
        q = self.q.ustrip("").reshape(shape)
        I = self.I.uconvert(I_unit).ustrip(I_unit).reshape(shape)

        # alpha: each Gaussian's major axis, as the "mathematical" angle
        # (counterclockwise in the grid's own (x, y) plane) the quadrature
        # below is expressed in. Derived from ProjectedBinning's fixed
        # parity (positive x is 90 degrees east of positive y): the sky PA
        # of the grid's own x-axis is `y_axis_pa + 90 deg`, and PA increases
        # in the opposite rotational sense to alpha (north-through-east vs.
        # counterclockwise-from-x), so alpha = (grid x-axis PA) - (sky PA).
        sky_pa = self.major_axis_pa.ustrip("rad") + self.PA_twist.ustrip("rad").reshape(
            shape
        )
        alpha = binning.y_axis_pa.ustrip("rad") + jnp.pi / 2 - sky_pa

        # Cappellari (2002) appendix B3's "p" -- unrelated to `Deprojected3DMGE`'s
        # intrinsic axial ratio `p`, just reusing the paper's own notation.
        cappellari_p = jnp.sqrt(1 + q**2 + (1 - q**2) * jnp.cos(2 * alpha))

        x = binning.x_nodes[None, :, :, None]  # (1, Nx, Q, 1)

        def erf_arg(y: jnp.ndarray) -> jnp.ndarray:
            return ((1 - q**2) * x * jnp.sin(2 * alpha) - cappellari_p**2 * y) / (
                2 * cappellari_p * q * sigma
            )

        erf_diff = erf_arg(binning.y_lo[None, None, None, :])
        erf_diff = erf(erf_diff) - erf(erf_arg(binning.y_hi[None, None, None, :]))
        exponent = jnp.exp(-((x / (cappellari_p * sigma)) ** 2))

        integrand = (
            I * q * sigma * jnp.sqrt(jnp.pi) / cappellari_p * erf_diff * exponent
        )  # (G, Nx, Q, Ny)

        pixel_mass = jnp.einsum("iq,giqj->ij", binning.x_weights, integrand)  # (Nx, Ny)

        binned = segment_sum(
            pixel_mass.ravel(), binning.bins.ravel(), num_segments=binning.n_bins + 1
        )

        mass_unit = I_unit * coord_unit**2
        return Quantity(binned[1:], mass_unit)

    def deproject_oblate(self, inclination: Quantity) -> Deprojected3DMGE:
        """Deproject to an intrinsic 3D MGE, assuming oblate axisymmetry.

        Oblate axisymmetric MGE deprojection (Monnet, Bacon & Emsellem 1992; Cappellari
        2002 eq. 9): each projected (2D) Gaussian, with observed axial ratio ``q'`` and
        width ``sigma``, corresponds to an intrinsic 3D Gaussian with the *same*
        ``sigma`` and an intrinsic axial ratio ``q`` (short/long axis, i.e. ``C/A``)
        determined by the inclination ``i`` between the line of sight and the symmetry
        axis, via ``q**2 = (q'**2 - cos(i)**2) / sin(i)**2``. This is the oblate
        convention: ``p = B/A = 1``, ``q = C/A <= 1`` (so ``A = B >= C``) -- the
        ``p = 1`` special case of the general triaxial ellipsoid, an axisymmetric
        system having no intermediate axis. A *prolate* spheroid (long axis = symmetry
        axis) obeys a different relation and is not produced by this path.

        ``i`` is canonically in ``(0, 90]`` degrees: ``i = 90 deg`` is edge-on, where
        ``q = q'``; ``i = 0 deg`` is face-on, where the projection is circular and
        carries no shape information (``sin(i) = 0``); and ``i`` and ``180 deg - i``
        give an identical projection for any equatorially-symmetric model, so the
        upper half-space is redundant. Inclinations outside ``(0, 90]`` deg are
        rejected rather than folded through the squared trigonometry.

        Args:
            inclination: The viewing angle between the line of sight and the symmetry
                axis, in ``(0, 90]`` degrees.

        Returns:
            A `Deprojected3DMGE` with intrinsic axial ratio `q` and `p = 1` for every
            component.

        Raises:
            ValueError: If `inclination` is outside ``(0, 90]`` degrees; if `sigma`
                isn't in physical (length) units -- call `angular_to_physical` first;
                or if any component has nonzero `PA_twist` (an axisymmetric system
                can't have isophote twist).
            MGEDeprojectionError: If any component has no real solution at this
                inclination (``q' < cos(i)``), or an intrinsic `q` outside
                TNT's ``0 < q <= 1`` convention (``p`` is always 1 here).
        """
        inclination_deg = float(inclination.ustrip("deg"))
        if not 0.0 < inclination_deg <= 90.0:
            raise ValueError(
                "deproject_oblate requires an inclination in (0, 90] degrees "
                f"(got {inclination_deg} deg); i and 180 deg - i give the same "
                "projection, and i = 0 deg (face-on) carries no shape information."
            )
        if not self.sigma.unit.is_equivalent(au.m):
            raise ValueError(
                "deproject_oblate requires physical (length) sigma; "
                "call angular_to_physical(distance) first."
            )
        if not bool(jnp.all(self.PA_twist.ustrip("rad") == 0)):
            raise ValueError(
                "deproject_oblate requires PA_twist == 0 for every "
                "component (an axisymmetric system has no isophote twist)."
            )

        self._check_deprojection_structure(inclination)
        model, valid = self._oblate_candidate(inclination)
        if not bool(valid):
            _check_axial_ratios(model.p.ustrip(""), model.q.ustrip(""))
            raise MGEDeprojectionError(
                "Oblate deprojection has no reliable solution at this numerical "
                "precision boundary (requires 0 < q <= p <= 1 "
                "and representable values)."
            )
        return model

    def _check_deprojection_structure(self, *angles: Quantity) -> None:
        """Reject static units, types and shapes independently of proposal values."""
        for angle in angles:
            if not isinstance(angle, Quantity) or angle.ndim != 0:
                raise ValueError("MGE viewing angles must be scalar Quantities.")
            validate_dimension(angle.unit, "angle", "MGE viewing angle")
        n = self.I.shape
        if len(n) != 1 or n[0] == 0:
            raise ValueError("MGE columns must be nonempty one-dimensional arrays.")
        for name, dimension in (
            ("I", self._intensity_dimension),
            ("sigma", "length"),
            ("q", "dimensionless"),
            ("PA_twist", "angle"),
        ):
            value = getattr(self, name)
            if not isinstance(value, Quantity) or value.shape != n:
                raise ValueError("MGE columns must be Quantities with matching shapes.")
            validate_dimension(value.unit, dimension, f"MGE {name}")

    def deproject_oblate_with_validity(
        self, inclination: Quantity
    ) -> tuple[Deprojected3DMGE, jax.Array]:
        """Trace one native inclination; return intrinsic MGE and scalar validity.

        Invalid results are zero placeholders. Numerical probes are detached
        from AD before conditional construction. Static unit/shape errors raise.
        Cancellation in q_obs**2 - cos(i)**2 must retain relative accuracy
        within 50 sqrt(eps) at the active precision.
        """
        self._check_deprojection_structure(inclination)
        return _guard_deprojection(self._oblate_candidate, (inclination,))

    def _oblate_candidate(
        self, inclination: Quantity
    ) -> tuple[Deprojected3DMGE, jax.Array]:
        angle = inclination.ustrip("rad")
        cos_i, sin_i = jnp.cos(angle), jnp.sin(angle)
        observed = self.q.ustrip("")
        numerator = observed**2 - cos_i**2
        circular = observed == 1
        # The exact spherical solution avoids cancellation and roundoff above
        # q=1. Safe operands also prevent a masked 0/0 from poisoning AD.
        q = jnp.sqrt(jnp.where(circular, 1, numerator)) / jnp.where(circular, 1, sin_i)
        model = Deprojected3DMGE(
            I=self.I * (observed / (jnp.sqrt(2 * jnp.pi) * q)) / self.sigma,
            sigma=self.sigma,
            p=Quantity(jnp.ones_like(q), ""),
            q=Quantity(q, ""),
        )
        eps = jnp.finfo(q.dtype).eps
        valid = (
            (inclination.ustrip("deg") > 0)
            & (inclination.ustrip("deg") <= 90)
            & jnp.all(self.PA_twist.ustrip("rad") == 0)
            & jnp.all(_axial_ratios_valid(jnp.ones_like(observed), observed))
            & jnp.all(
                circular | (numerator > (observed**2 + cos_i**2) * jnp.sqrt(eps) / 50)
            )
            & _deprojected_valid(model)
        )
        return model, valid

    def inclination_from_q_min(self, q_min: float | jax.Array) -> Quantity:
        """The inclination giving the anchor component's intrinsic axial ratio.

        Inverts `deproject_oblate`'s own relation, ``q_obs**2 = q_min**2 *
        sin(i)**2 + cos(i)**2``, evaluated at this MGE's own anchor ``q_obs'
        = min(component q)`` (`_triaxial_anchor`; its `PA_twist` element is
        unused here -- `deproject_oblate` itself requires every component's
        `PA_twist` to be zero for an oblate system):

        ``cos(i)**2 = (q_obs'**2 - q_min**2) / (1 - q_min**2)``.

        The edge-on limit ``q_min == q_obs'`` has a finite angle but an
        unbounded conversion derivative, and is rejected. A near-spherical
        anchor and thin intrinsic shapes can also be numerically unreliable.

        Raises:
            MGEDeprojectionError: If this MGE's anchor is circular
                (``q_obs' == 1``), if ``q_min`` is outside
                ``0 < q_min <= q_obs'``, if ``q_min`` is too close to 1 to
                divide by reliably at the working precision, or if
                `deproject_oblate`'s own inverse relation can't recover
                ``q_min`` accurately at the resulting inclination (see
                `_QMIN_ROUNDTRIP_TOL_FACTOR`).
        """
        self._check_deprojection_structure(Quantity(0.0, "rad"))
        return _checked_shape_conversion(self._q_min_candidate, _shape_numbers(q_min))

    def inclination_from_q_min_with_validity(
        self, q_min: float | jax.Array
    ) -> tuple[Quantity, jax.Array]:
        """Convert one intrinsic anchor ratio, with zero invalid placeholders."""
        self._check_deprojection_structure(Quantity(0.0, "rad"))
        return _guard_shape_conversion(self._q_min_candidate, _shape_numbers(q_min))

    def _q_min_candidate(
        self, q_min: jax.Array
    ) -> tuple[Quantity, dict[str, jax.Array]]:
        q_obs, _ = self._triaxial_anchor()
        eps = jnp.finfo(q_min.dtype).eps
        den = 1 - q_min**2
        cos2_i = (q_obs**2 - q_min**2) / den
        inclination = Quantity(jnp.arccos(jnp.sqrt(cos2_i)), "rad")
        model, deprojection_valid = self._oblate_candidate(inclination)
        anchor = jnp.argmin(self.q.ustrip(""))
        recovered = model.q.ustrip("")[anchor]
        tolerance = _QMIN_ROUNDTRIP_TOL_FACTOR * jnp.sqrt(eps)
        checks = {
            "Oblate deprojection needs a flattened MGE; a circular MGE has no "
            "inclination to solve for.": (q_obs > 0) & (q_obs < 1),
            "q_min has no oblate deprojection: requires 0 < q_min <= q'.": (
                jnp.isfinite(q_min) & (q_min > 0) & (q_min <= q_obs)
            ),
            "The edge-on q_min == q' limit has an unbounded conversion derivative.": (
                q_min < q_obs
            ),
            "q_min is too close to 1 to deproject reliably at this numerical "
            "precision boundary.": den >= eps,
            "q_min is too close to a numerical-precision boundary in oblate "
            "deprojection to represent accurately (round-trip drift).": (
                deprojection_valid & (jnp.abs(recovered - q_min) <= tolerance * q_min)
            ),
        }
        return inclination, checks

    def q_min_from_inclination(self, inclination: Quantity) -> float:
        """The anchor component's intrinsic axial ratio at this inclination.

        The numerical inverse of `inclination_from_q_min`: deprojects the
        whole MGE (`deproject_oblate`) and reads off the anchor component's
        own intrinsic `q`, reusing all of that method's unit/`PA_twist`/
        domain validation rather than duplicating it.
        """
        deprojected = self.deproject_oblate(inclination)
        anchor = int(jnp.argmin(self.q.ustrip("")))
        return float(deprojected.q[anchor].ustrip(""))

    def deproject_triaxial(
        self, theta: Quantity, phi: Quantity, psi: Quantity
    ) -> Deprojected3DMGE:
        """Deproject to an intrinsic 3D MGE, assuming triaxiality.

        General triaxial MGE deprojection (de Zeeuw & Franx 1989; Cappellari 2002 eqs.
        6-8; van den Bosch et al. 2008 eqs. 6-9), solving for each Gaussian's intrinsic
        axial ratios ``p`` (``B/A``) and ``q`` (``C/A``) given its observed axial ratio
        and the viewing geometry. Unlike `deproject_oblate`, the intrinsic `sigma`
        is *not* generally equal to the observed one -- it's recovered via the
        scale-length compression factor ``u = sigma_observed / sigma_intrinsic`` (van
        den Bosch et al. 2008 eq. 9), which is only ever 1 at special viewing angles
        (e.g. looking down a principal axis).

        `theta`, `phi`, and `psi` are three *global* viewing angles, the same for every
        Gaussian in the MGE; each Gaussian's own angle used in the deprojection is ``psi
        + PA_twist`` (van den Bosch et al. 2008 eq. 6), where `PA_twist` is its
        isophotal twist relative to a reference component (conventionally 0 for that
        component).

        A solution isn't guaranteed to exist for arbitrary viewing angles (Cappellari
        2002 sec. 2.2.1), and a solution that exists isn't guaranteed to respect TNT's
        ``0 < q <= p <= 1`` intrinsic-axis convention (``p = B/A``, ``q = C/A``) --
        both cases raise `MGEDeprojectionError`. The matching
        `deproject_triaxial_with_validity` method exposes those checks inside
        a JAX trace, including numerical conditioning and covariance residuals.

        Args:
            theta: Global polar viewing angle, relative to the principal axes.
            phi: Global azimuthal viewing angle, relative to the principal axes.
            psi: Global rotation of the object around the line of sight.

        Returns:
            A `Deprojected3DMGE` with intrinsic axial ratios `p`, `q`, and intrinsic
            `sigma` (see above -- generally not equal to the observed `sigma`).

        Raises:
            ValueError: If `sigma` isn't in physical (length) units -- call
                `angular_to_physical` first.
            MGEDeprojectionError: If any component has no real solution at this
                viewing geometry, or intrinsic axial ratios outside TNT's
                ``0 < q <= p <= 1`` convention.
        """
        if not self.sigma.unit.is_equivalent(au.m):
            raise ValueError(
                "deproject_triaxial requires physical (length) sigma; "
                "call angular_to_physical(distance) first."
            )

        self._check_deprojection_structure(theta, phi, psi)
        model, valid = self._triaxial_candidate(theta, phi, psi)
        if not bool(valid):
            _check_axial_ratios(model.p.ustrip(""), model.q.ustrip(""))
            raise MGEDeprojectionError(
                "Triaxial deprojection has no reliable solution at this numerical "
                "precision boundary (requires 0 < q <= p <= 1 "
                "and representable values)."
            )
        return model

    def deproject_triaxial_with_validity(
        self, theta: Quantity, phi: Quantity, psi: Quantity
    ) -> tuple[Deprojected3DMGE, jax.Array]:
        """Trace one native viewing geometry, with shared accuracy/domain checks."""
        self._check_deprojection_structure(theta, phi, psi)
        return _guard_deprojection(self._triaxial_candidate, (theta, phi, psi))

    def _triaxial_candidate(
        self, theta: Quantity, phi: Quantity, psi: Quantity
    ) -> tuple[Deprojected3DMGE, jax.Array]:
        p, q, u = self._triaxial_component_ratios(theta, phi, psi)
        model = Deprojected3DMGE(
            I=self.I
            * (u**3 * self.q.ustrip("") / (jnp.sqrt(2 * jnp.pi) * p * q))
            / self.sigma,
            sigma=self.sigma / u,
            p=Quantity(p, ""),
            q=Quantity(q, ""),
        )
        observed = self.q.ustrip("")
        valid = jnp.all(_axial_ratios_valid(jnp.ones_like(observed), observed))
        valid = (
            valid
            & _deprojected_valid(model)
            & _triaxial_geometry_valid(
                theta.ustrip("rad"),
                phi.ustrip("rad"),
                psi.ustrip("rad") + self.PA_twist.ustrip("rad"),
                self.q.ustrip(""),
                p,
                q,
                u,
            )
        )
        return model, valid

    def _triaxial_component_ratios(
        self, theta: Quantity, phi: Quantity, psi: Quantity
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Per-Gaussian intrinsic ``(p, q, u)`` at global viewing angles.

        The shared core of `deproject_triaxial` and `triaxial_intrinsic_shape`:
        each Gaussian's own line-of-sight angle is ``psi + PA_twist`` (van den
        Bosch et al. 2008 eq. 6). No validity check -- the callers decide (see
        `_triaxial_intrinsic_axis_ratios`).
        """
        psi_r = psi.ustrip("rad") + self.PA_twist.ustrip("rad")
        return _triaxial_intrinsic_axis_ratios(
            theta.ustrip("rad"), phi.ustrip("rad"), psi_r, self.q.ustrip("")
        )

    def _triaxial_anchor(self) -> tuple[jax.Array, jax.Array]:
        """``(q', PA_twist)`` of the flattest Gaussian -- the ``(p, q, u)`` anchor.

        The van den Bosch relations pin the triaxial viewing geometry to one
        reference ellipse; TNT uses the component with the smallest observed
        axial ratio. Ties break by component order. `triaxial_viewing_angles`
        subtracts this ``PA_twist`` (radians) from the global ``psi`` it
        returns, so `_triaxial_component_ratios` adds it straight back for the
        anchor.
        """
        q = self.q.ustrip("")
        anchor = jnp.argmin(q)
        return q[anchor], self.PA_twist.ustrip("rad")[anchor]

    def triaxial_viewing_angles(
        self, p: float | jax.Array, q: float | jax.Array, u: float | jax.Array
    ) -> tuple[Quantity, Quantity, Quantity]:
        """Convert the flattest Gaussian's intrinsic shape to global angles.

        The de Zeeuw & Franx / van den Bosch relations require a flattened
        anchor, 0 < q < p <= 1, and max(q/q', p) < u <= min(p/q', 1).
        Ties in observed q break by row order. The anchor PA twist is removed
        from global psi and restored by per-row deprojection.

        Numerical acceptance requires an interior compression separated from
        both bounds by 4 sqrt(eps), nonsingular weights, and an accurate
        recovered anchor shape. A value requiring boundary clipping is
        rejected: clipping changes the proposed shape and its derivatives.
        Eager errors use the same checks as the with_validity counterpart.
        """
        self._check_deprojection_structure(Quantity(0.0, "rad"))
        return _checked_shape_conversion(self._pqu_candidate, _shape_numbers(p, q, u))

    def triaxial_viewing_angles_with_validity(
        self, p: float | jax.Array, q: float | jax.Array, u: float | jax.Array
    ) -> tuple[tuple[Quantity, Quantity, Quantity], jax.Array]:
        """Trace one intrinsic shape, returning angles and scalar validity."""
        self._check_deprojection_structure(Quantity(0.0, "rad"))
        return _guard_shape_conversion(self._pqu_candidate, _shape_numbers(p, q, u))

    def _pqu_candidate(
        self, p: jax.Array, q: jax.Array, u: jax.Array
    ) -> tuple[tuple[Quantity, Quantity, Quantity], dict[str, jax.Array]]:
        q_obs, anchor_twist = self._triaxial_anchor()
        eps = jnp.finfo(p.dtype).eps
        lo, hi = jnp.maximum(q / q_obs, p), jnp.minimum(p / q_obs, 1)
        margin = 4 * jnp.sqrt(eps)
        u_lo, u_hi = lo + jnp.maximum(lo, 1) * margin, hi * (1 - margin)
        # Bounded probe operands do not turn a clipped proposal into a model.
        u_calc = jnp.clip(u, u_lo, u_hi)
        p2, q2, u2, o2 = p**2, q**2, u_calc**2, q_obs**2
        d_theta = (1 - q2) * (p2 - q2)
        d_phi = (1 - u2) * (1 - o2 * u2) * (p2 - q2)
        d_psi = (1 - u2) * (u2 - p2) * (o2 * u2 - q2)
        w1 = (u2 - q2) * (o2 * u2 - q2) / d_theta
        w2 = (u2 - p2) * (p2 - o2 * u2) * (1 - q2) / d_phi
        w3 = (1 - o2 * u2) * (p2 - o2 * u2) * (u2 - q2) / d_psi
        weight_atol = jnp.maximum(_TRIAXIAL_WEIGHT_ATOL, 16 * eps)
        theta = jnp.arccos(jnp.sqrt(jnp.clip(w1, 0, 1)))
        phi = jnp.arctan(jnp.sqrt(jnp.maximum(w2, 0)))
        psi = jnp.pi - jnp.arctan(jnp.sqrt(jnp.maximum(w3, 0))) - anchor_twist
        angles = tuple(Quantity(value, "rad") for value in (theta, phi, psi))
        p_r, q_r, u_r = _triaxial_intrinsic_axis_ratios(
            theta, phi, psi + anchor_twist, q_obs
        )
        target, recovered = jnp.stack([p, q, u]), jnp.stack([p_r, q_r, u_r])
        checks = {
            "Triaxial deprojection needs a flattened anchor; a circular MGE "
            "has no viewing geometry to solve for.": (q_obs > 0) & (q_obs < 1),
            "(p, q, u) requires finite 0 < q <= p <= 1 and 0 < u <= 1.": (
                _axial_ratios_valid(p, q) & jnp.isfinite(u) & (u > 0) & (u <= 1)
            ),
            "q == p (prolate) has no unique triaxial viewing geometry.": q < p,
            "No triaxial deprojection: requires max(q/q', p) < u <= min(p/q', 1).": (
                (lo < u) & (u <= hi + jnp.maximum(1, hi) * eps)
            ),
            "The triaxial domain is too narrow to deproject reliably at this "
            "numerical precision.": u_lo < u_hi,
            "The compression is too close to a numerical-precision boundary; "
            "clipping would change the proposed shape and derivatives.": (
                (u > u_lo) & (u < u_hi)
            ),
            "Viewing geometry too close to a coordinate singularity to deproject.": (
                jnp.min(jnp.abs(jnp.stack([d_theta, d_phi, d_psi]))) >= eps
            ),
            "Degenerate viewing geometry: invalid de Zeeuw & Franx weights.": (
                (w1 >= -weight_atol)
                & (w1 <= 1 + weight_atol)
                & (w2 >= -weight_atol)
                & (w3 >= -weight_atol)
                & jnp.all(jnp.isfinite(jnp.stack([theta, phi, psi])))
            ),
            "Intrinsic shape is too close to a numerical-precision boundary "
            "to recover accurately.": jnp.all(
                jnp.abs(recovered - target) <= 100 * eps + 50 * jnp.sqrt(eps) * target
            ),
        }
        return angles, checks

    def triaxial_intrinsic_shape(
        self, theta: Quantity, phi: Quantity, psi: Quantity
    ) -> tuple[float, float, float]:
        """The flattest Gaussian's intrinsic ``(p, q, u)`` at these viewing angles.

        The anchor-component slice of `deproject_triaxial`, and the numerical
        inverse of `triaxial_viewing_angles` -- exact for an interior geometry,
        for accepted interior proposals. Compression values requiring clipping
        are rejected by `triaxial_viewing_angles`.
        The anchor's ``PA_twist`` is folded back into ``psi`` by
        `_triaxial_component_ratios`, same as for every other Gaussian.
        """
        p, q, u = self._triaxial_component_ratios(theta, phi, psi)
        anchor = int(jnp.argmin(self.q.ustrip("")))
        return float(p[anchor]), float(q[anchor]), float(u[anchor])

    def viewing_angles_from_T_Tmaj_Tmin(
        self, T: float | jax.Array, T_maj: float | jax.Array, T_min: float | jax.Array
    ) -> tuple[Quantity, Quantity, Quantity]:
        """Convert one Quenneville et al. shape to global viewing angles.

        The shared JAX checks cover [0, 1] coordinates, the algebraic
        denominator, strictly positive intrinsic thickness, pqu geometry,
        and recovered-coordinate accuracy. Each round-trip coordinate must
        agree within 100 eps + 50 sqrt(eps) times its absolute value.
        """
        self._check_deprojection_structure(Quantity(0.0, "rad"))
        return _checked_shape_conversion(
            self._tmajmin_candidate, _shape_numbers(T, T_maj, T_min)
        )

    def viewing_angles_from_T_Tmaj_Tmin_with_validity(
        self, T: float | jax.Array, T_maj: float | jax.Array, T_min: float | jax.Array
    ) -> tuple[tuple[Quantity, Quantity, Quantity], jax.Array]:
        """Trace one shape; invalid results contain zero angle placeholders."""
        self._check_deprojection_structure(Quantity(0.0, "rad"))
        return _guard_shape_conversion(
            self._tmajmin_candidate, _shape_numbers(T, T_maj, T_min)
        )

    def _tmajmin_candidate(
        self, T: jax.Array, T_maj: jax.Array, T_min: jax.Array
    ) -> tuple[tuple[Quantity, Quantity, Quantity], dict[str, jax.Array]]:
        q_obs, _ = self._triaxial_anchor()
        shape, checks = _p_q_u_candidate(T, T_maj, T_min, q_obs)
        angles, geometry_checks = self._pqu_candidate(*shape)
        checks.update(geometry_checks)
        p_r, q_r, u_r = self._triaxial_component_ratios(*angles)
        anchor = jnp.argmin(self.q.ustrip(""))
        recovered, inverse_checks = _T_Tmaj_Tmin_candidate(
            p_r[anchor], q_r[anchor], u_r[anchor], q_obs
        )
        checks.update(inverse_checks)
        eps = jnp.finfo(T.dtype).eps
        target = jnp.stack([T, T_maj, T_min])
        checks[
            "Shape coordinates are too close to a numerical-precision "
            "boundary to recover accurately (round-trip drift)."
        ] = jnp.all(
            jnp.abs(jnp.stack(recovered) - target)
            <= _TMAJMIN_ROUNDTRIP_ABS_TOL_FACTOR * eps
            + _TMAJMIN_ROUNDTRIP_REL_TOL_FACTOR * jnp.sqrt(eps) * jnp.abs(target)
        )
        return angles, checks

    def T_Tmaj_Tmin_from_viewing_angles(
        self, theta: Quantity, phi: Quantity, psi: Quantity
    ) -> tuple[float, float, float]:
        """The anchor's ``(T, T_maj, T_min)`` at these viewing angles.

        The numerical inverse of `viewing_angles_from_T_Tmaj_Tmin`:
        `triaxial_intrinsic_shape`'s anchor ``(p, q, u)``, reparameterized via
        `_T_Tmaj_Tmin_from_p_q_u`.
        """
        p, q, u = self.triaxial_intrinsic_shape(theta, phi, psi)
        q_obs, _ = self._triaxial_anchor()
        return _T_Tmaj_Tmin_from_p_q_u(p, q, u, q_obs)


class LightMGE(AbstractMGE):
    """An MGE of a surface-brightness distribution (``I`` in e.g. Lsun/pc2)."""

    _intensity_dimension: ClassVar[str] = "light_surface_brightness"

    def to_mass(self, m_over_l: Quantity) -> MassMGE:
        """Convert to a MassMGE given a mass-to-light ratio.

        `sigma`, `q`, `PA_twist`, and `major_axis_pa` are unaffected and carried over
        unchanged -- only `I` (and hence what it represents) changes.

        Args:
            m_over_l: The mass-to-light ratio (e.g. in Msun/Lsun), either a single value
                applied to every component, or an array with one value per Gaussian
                component.

        Returns:
            A `MassMGE` with ``I = self.I * m_over_l``.

        Raises:
            ValueError: If `m_over_l` is array-valued and its length doesn't match the
                number of Gaussian components.
        """
        if m_over_l.ndim > 0 and m_over_l.shape[0] != self.I.shape[0]:
            raise ValueError(
                f"m_over_l has {m_over_l.shape[0]} components, but this MGE "
                f"has {self.I.shape[0]}."
            )

        return MassMGE(
            I=self.I * m_over_l,
            sigma=self.sigma,
            q=self.q,
            PA_twist=self.PA_twist,
            major_axis_pa=self.major_axis_pa,
        )


class MassMGE(AbstractMGE):
    """An MGE of a mass surface-density distribution (``I`` in e.g. Msun/pc2)."""

    _intensity_dimension: ClassVar[str] = "mass_surface_density"


def _gaussian_radial_antiderivative(a: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    """Antiderivative of ``r**2 * exp(-a * r**2)`` with respect to ``r``.

    Args:
        a: The Gaussian's rate parameter (positive), broadcastable against `r`.
        r: The radius (finite) at which to evaluate the antiderivative.

    Returns:
        ``integral_0^r r'**2 exp(-a r'**2) dr'`` up to the (shared, cancelling)
        constant of integration -- i.e. valid for computing definite integrals
        between finite radii, or between a finite radius and 0.
    """
    return -r / (2 * a) * jnp.exp(-a * r**2) + jnp.sqrt(jnp.pi) / (4 * a**1.5) * erf(
        jnp.sqrt(a) * r
    )


class Deprojected3DMGE(eqx.Module):
    """An intrinsic (3D) MGE, produced by deprojecting a `LightMGE`/`MassMGE`.

    Each Gaussian component is described by its peak (3D) density ``I``, intrinsic width
    ``sigma``, and intrinsic axial ratios ``p`` (``B/A``) and ``q`` (``C/A``).
    `deproject_oblate` always returns the same `sigma` as the projected MGE it
    came from; `deproject_triaxial` generally does not (see its docstring). An
    oblate axisymmetric deprojection always has ``p == 1``, since axisymmetric
    ellipsoids have no intermediate axis.
    """

    I: Quantity
    sigma: Quantity
    p: Quantity
    q: Quantity

    @property
    def gaussian_widths(self) -> Quantity:
        """Floating-point widths in kpc for stable Gaussian construction.

        Preserve conversion before Galax forms powers of the widths, even
        when compilation rearranges arithmetic or stored widths are integers.
        """
        return Quantity(
            jax.lax.optimization_barrier(
                jnp.asarray(self.sigma.ustrip("kpc"), dtype=float)
            ),
            "kpc",
        )

    @property
    def component_masses(self) -> Quantity:
        """Integrated mass (or luminosity) of each intrinsic Gaussian.

        Shared by numerical validation and Galax construction; ``I`` is the
        central volume density, not the projected surface intensity.
        Calculate locally in kpc and Msun/Lsun without changing stored units.
        Interleave density and width factors to avoid overflowing sigma**3
        when the integrated mass itself is representable.
        """
        unit = au.Msun if self.I.unit.is_equivalent(au.Msun / au.kpc**3) else au.Lsun
        density = self.I.ustrip(unit / au.kpc**3)
        sigma = self.sigma.ustrip("kpc")
        # Keep compiler reassociation from combining a large declared-unit
        # width with conversion factors only after its products overflow.
        density, sigma = jax.lax.optimization_barrier((density, sigma))
        mass = (
            density
            * sigma
            * self.p.ustrip("")
            * self.q.ustrip("")
            * (2 * jnp.pi) ** 1.5
            * sigma
            * sigma
        )
        return Quantity(mass, unit)

    def spherical_mass_grid(self, grid: SphericalGrid) -> Quantity:
        """Mass in each cell of a `SphericalGrid` (one octant).

        Along any fixed direction the density is an exact 1D Gaussian in ``r``,
        so every radial bin -- including the semi-infinite outermost one -- is
        integrated analytically via `erf`. The ``theta``/``phi`` integral within
        each angular cell is done with fixed-order Gauss-Legendre quadrature.

        Args:
            grid: The spherical grid to bin the mass into, from
                `SphericalGrid`.

        Returns:
            A `Quantity` of shape ``(grid.n_r, grid.n_theta, grid.n_phi)``
            giving the mass in each cell of the octant grid.
        """
        length_unit = grid.r_edges.unit
        finite_edges = grid.r_edges.ustrip(length_unit)[:-1]  # drop the r=inf edge

        cos_theta = grid.cos_theta_nodes  # (n_theta, Q)
        sin_theta = jnp.sqrt(1 - grid.cos_theta_nodes**2)
        cos_phi = jnp.cos(grid.phi_nodes)  # (n_phi, Q)
        sin_phi = jnp.sin(grid.phi_nodes)

        # Direction-dependent factors of each component's rate parameter a(theta,
        # phi), such that a * r**2 = (x**2 + y**2/p**2 + z**2/q**2) / (2 sigma**2)
        # -- broadcast to (n_theta, Q, n_phi, Q).
        x2 = (sin_theta[:, :, None, None] * cos_phi[None, None, :, :]) ** 2
        y2 = (sin_theta[:, :, None, None] * sin_phi[None, None, :, :]) ** 2
        z2 = jnp.broadcast_to(cos_theta[:, :, None, None] ** 2, x2.shape)

        sigma = self.sigma.ustrip(length_unit)  # (G,)
        p = self.p.ustrip("")  # (G,)
        q = self.q.ustrip("")  # (G,)
        I = self.I.ustrip(self.I.unit)

        # Add a leading components axis: (G, n_theta, Q, n_phi, Q).
        shape = (-1, 1, 1, 1, 1)
        a = (x2 + y2 / p.reshape(shape) ** 2 + z2 / q.reshape(shape) ** 2) / (
            2 * sigma.reshape(shape) ** 2
        )

        # Radial integral per component and direction, for every finite edge,
        # then differenced into per-bin integrals; the last bin runs to infinity.
        antideriv = _gaussian_radial_antiderivative(a[..., None], finite_edges)
        finite_bins = antideriv[..., 1:] - antideriv[..., :-1]  # (..., n_r - 1)
        full_integral = jnp.sqrt(jnp.pi) / (4 * a**1.5)
        last_bin = full_integral - antideriv[..., -1]
        radial = jnp.concatenate([finite_bins, last_bin[..., None]], axis=-1)

        # Weight by each component's amplitude and sum over components. No
        # explicit sin(theta) factor is needed here: theta_weights already
        # include it, via the cos(theta) quadrature above.
        integrand = jnp.sum(I.reshape(shape + (1,)) * radial, axis=0)

        mass = jnp.einsum(
            "ja,kb,jakbn->njk", grid.theta_weights, grid.phi_weights, integrand
        )

        mass_unit = self.I.unit * length_unit**3
        return Quantity(mass, mass_unit)


_MGE_CLASSES: tuple[type[AbstractMGE], ...] = (LightMGE, MassMGE)


def read_mge(path: str | Path, major_axis_pa: Quantity) -> AbstractMGE:
    """Read an MGE from an ECSV file, inferring whether it's light or mass.

    The kind is inferred from the declared unit of the file's ``I`` column: whichever of
    `LightMGE` (power/length**2) or `MassMGE` (mass/length**2) it is dimensionally
    consistent with. ``sigma`` must be angular; columns keep their declared
    units (see `from_qtable`). Only ``sigma`` changes with distance when
    `angular_to_physical` is subsequently applied.

    Args:
        path: Path to the ECSV file.
        major_axis_pa: The MGE major-axis PA the ``PA_twist`` column is
            measured from; supplied by configuration, not the ECSV. A finite
            scalar angle in ``[0, 180)`` degrees.

    Returns:
        A `LightMGE` or `MassMGE`, whichever matches the file's ``I`` column.

    Raises:
        ValueError: If the ``I`` column's unit doesn't match any known MGE
            kind, or (via `from_qtable`) `major_axis_pa` isn't a finite scalar
            angle in ``[0, 180)`` degrees.
    """
    table = QTable.read(path, format="ascii.ecsv")
    intensity_unit = table["I"].unit

    for cls in _MGE_CLASSES:
        target_unit = reference_unit(cls._intensity_dimension)
        if intensity_unit is not None and intensity_unit.is_equivalent(target_unit):
            return cls.from_qtable(table, major_axis_pa)

    expected = [reference_unit(cls._intensity_dimension) for cls in _MGE_CLASSES]
    raise ValueError(
        f"Could not infer MGE kind for {path}: its I column has unit "
        f"{intensity_unit!r}, which is not equivalent to any of {expected!r}."
    )


def build_mges(
    mges: Mapping[str, Mapping[str, Any]],
    input_directory: str | Path,
    distance: Quantity,
) -> dict[str, AbstractMGE]:
    """Build the named MGEs from a resolved configuration's ``MGEs`` mapping.

    Each MGE's kind (light or mass) is inferred from its file's declared
    units -- see `read_mge`. Every MGE is converted to physical units via
    `angular_to_physical` before being returned, since every consumer (e.g.
    `tnt.potential`'s MGE composite components) needs physical `sigma` to
    build a 3D potential. `tnt.spatial_binnings.build_spatial_binnings` is
    converted to physical units the same way, so a consumer needing both
    (e.g. a future `AbstractMGE.get_projected_mass` call) can assume
    dimensional consistency without converting either itself. This
    deliberately takes already-resolved, plain-data inputs rather than a
    `tnt.configuration.Configuration`, since that class explicitly holds no
    instantiated runtime objects.

    Args:
        mges: Mapping of unique identifiers to ``{file, major_axis_pa}``
            entries, e.g. a resolved configuration's ``MGEs`` section.
            ``major_axis_pa`` is an explicit ``{value, unit}`` angle -- the
            on-sky PA (north through east) of the MGE major axis, which the
            ECSV's ``PA_twist`` column is measured relative to.
        input_directory: Directory that each filename is resolved against,
            e.g. a resolved configuration's ``io_settings.input_directory``.
        distance: The distance to the object, e.g. a resolved
            configuration's ``system_attributes.distance``.

    Returns:
        A dict mapping each identifier to its physical-unit `LightMGE` or
        `MassMGE`.
    """
    directory = Path(input_directory)
    built: dict[str, AbstractMGE] = {}
    for name, entry in mges.items():
        declared_pa = entry["major_axis_pa"]
        major_axis_pa = Quantity(declared_pa["value"], declared_pa["unit"])
        mge = read_mge(directory / entry["file"], major_axis_pa)
        built[name] = mge.angular_to_physical(distance)
    return built
