"""Multi-Gaussian Expansion (MGE) models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Self

import astropy.units as au
import equinox as eqx
import jax.numpy as jnp
import unxt as u
from astropy.table import QTable
from jax.ops import segment_sum
from jax.scipy.special import erf
from unxt import AbstractUnitSystem, Quantity

from tnt import quantity_conversions
from tnt.spatial_binnings import ProjectedBinning, SphericalGrid


class MGEDeprojectionError(ValueError):
    """A deprojection has no solution, or violates TNT's ``0 < q <= p <= 1`` convention.

    Raised eagerly, in plain Python -- not `jax.jit`/`jax.vmap`-traceable.
    Fine today since nothing calls this under a trace; see
    `aidocs/KNOWLEDGE.md` for the durable limitation and the trigger for
    revisiting it.
    """


def _check_axial_ratios(p: jnp.ndarray, q: jnp.ndarray) -> None:
    """Raise `MGEDeprojectionError` unless every component has ``0 < q <= p <= 1``.

    Catches both an unsolvable deprojection (`p`/`q` containing `nan`, which
    fails every comparison below) and a solution that's finite but violates
    TNT's intrinsic-axis convention.
    """
    valid = (q > 0) & (q <= p) & (p <= 1)
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


class AbstractMGE(eqx.Module):
    """Shared structure and behaviour for MGE models.

    Each Gaussian component is described by its peak intensity ``I``, width ``sigma``,
    axial ratio ``q``, and position-angle twist ``PA_twist``, stored as arrays converted
    to a unit system's units. Subclasses fix which physical dimension ``I`` represents
    (e.g. light or mass) by setting `_intensity_attr` to the corresponding
    `unxt.AbstractUnitSystem` attribute name. Not meant to be instantiated directly --
    use `LightMGE` or `MassMGE`.
    """

    _intensity_attr: ClassVar[str]

    I: Quantity
    sigma: Quantity
    q: Quantity
    PA_twist: Quantity

    @classmethod
    def _surface_intensity_unit(cls, unit_system: AbstractUnitSystem) -> au.UnitBase:
        """Return this MGE kind's surface-intensity unit."""
        return (
            unit_system[u.dimension(cls._intensity_attr)]
            / unit_system[u.dimension("angle")] ** 2
        )

    @classmethod
    def from_qtable(cls, table: QTable, unit_system: AbstractUnitSystem) -> Self:
        """Build an MGE from a table, validating and converting its columns.

        Args:
            table: A table with columns ``I``, ``sigma``, ``q``, and ``PA_twist``, each
                carrying an astropy unit.
            unit_system: The unit system to convert the columns into.

        Returns:
            An MGE with columns converted to `unit_system`'s units.

        Raises:
            astropy.units.UnitConversionError: If a column's unit is not
                dimensionally consistent with the expected physical type.
            ValueError: If any ``q`` value is outside ``(0, 1]``.
        """
        intensity_unit = cls._surface_intensity_unit(unit_system)
        target_units = {
            "I": intensity_unit,
            "sigma": unit_system.angle,
            "q": "",
            "PA_twist": unit_system.angle,
        }

        columns = {
            name: Quantity.from_(table[name].to(unit))
            for name, unit in target_units.items()
        }

        q = columns["q"].ustrip("")
        if not bool(jnp.all((q > 0) & (q <= 1))):
            raise ValueError(f"q must satisfy 0 < q <= 1 for all components, got {q}")

        return cls(**columns)

    @classmethod
    def read(cls, path: str | Path, unit_system: AbstractUnitSystem) -> Self:
        """Read an MGE from an ECSV file, converting into a unit system's units.

        Args:
            path: Path to the ECSV file.
            unit_system: The unit system to convert the columns into.

        Returns:
            An MGE with columns converted to `unit_system`'s units.
        """
        table = QTable.read(path, format="ascii.ecsv")
        return cls.from_qtable(table, unit_system)

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
            I=self.I * factor, sigma=self.sigma, q=self.q, PA_twist=self.PA_twist
        )

    def angular_to_physical(self, distance: Quantity) -> Self:
        """Convert `sigma` and `I` from angular to physical (length) units.

        `q` (dimensionless) and `PA_twist` (an orientation angle, not a spatial size)
        are unaffected and carried over unchanged.

        Args:
            distance: The distance to the object.

        Returns:
            A new MGE with `sigma` in `distance`'s unit and `I` converted to match.
        """
        sigma_physical = quantity_conversions.angular_to_physical(self.sigma, distance)
        solid_angle = Quantity(1.0, f"{self.sigma.unit}2")
        I_physical = self.I * solid_angle / distance**2

        return type(self)(
            I=I_physical, sigma=sigma_physical, q=self.q, PA_twist=self.PA_twist
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

        `binning`'s `PA` is measured counterclockwise from the aperture
        grid's y-axis to each component's own major axis, in the
        Cappellari/van den Bosch convention used elsewhere in this module for
        `psi` and `PA_twist` (opposite handedness to the "mathematical" angle
        from the x-axis) -- `PA_twist` away from a reference component.

        Args:
            binning: The projected-plane aperture grid and pixel-to-bin
                assignment to integrate this MGE over. Its coordinates
                (`min_x`, `min_y`, `x_extent`, `y_extent`) must be
                dimensionally consistent with `sigma` -- both angular, or
                both converted to the same physical unit via
                `angular_to_physical`/`ProjectedBinning.angular_to_physical`.

        Returns:
            A `Quantity` of shape ``(n_bins,)`` giving each bin's total,
            ordered by increasing bin ID (bin 1 first). ``n_bins`` is
            `binning`'s largest bin ID.

        Raises:
            astropy.units.UnitConversionError: If `sigma` and `binning`'s
                coordinates aren't dimensionally consistent.
        """
        coord_unit = binning.min_x.unit

        # Add a leading components axis: everything below broadcasts to
        # (G, Nx, Q, Ny).
        shape = (-1, 1, 1, 1)
        sigma = self.sigma.ustrip(coord_unit).reshape(shape)
        q = self.q.ustrip("").reshape(shape)
        I = self.I.ustrip(self.I.unit).reshape(shape)
        # PA is measured from the y-axis (Cappellari/van den Bosch convention);
        # the -pi/2 converts it to alpha, the "mathematical" angle from the
        # x-axis that the integral below is expressed in.
        alpha = (
            binning.PA.ustrip("rad")
            - jnp.pi / 2
            + self.PA_twist.ustrip("rad").reshape(shape)
        )

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

        mass_unit = self.I.unit * coord_unit**2
        return Quantity(binned[1:], mass_unit)

    def deproject_axisymmetric(self, inclination: Quantity) -> Deprojected3DMGE:
        """Deproject to an intrinsic 3D MGE, assuming axisymmetry.

        Axisymmetric MGE deprojection (Monnet, Bacon & Emsellem 1992; Cappellari 2002
        eq. 9): each projected (2D) Gaussian, with observed axial ratio ``q'`` and width
        ``sigma``, corresponds to an intrinsic 3D Gaussian with the *same* ``sigma`` and
        an intrinsic axial ratio ``q`` (short/long axis, i.e. ``C/A``) determined by the
        inclination ``i`` between the line of sight and the symmetry axis (``i = 90
        deg`` is edge-on, where ``q = q'``; ``i = 0 deg`` is face-on, where deprojection
        is undefined). This is the ``p = 1`` (``B = A``) special case of the general
        triaxial ellipsoid, since an axisymmetric system has no intermediate axis.

        Args:
            inclination: The viewing angle between the line of sight and the symmetry
                axis.

        Returns:
            A `Deprojected3DMGE` with intrinsic axial ratio `q` and `p = 1` for every
            component.

        Raises:
            ValueError: If `sigma` isn't in physical (length) units -- call
                `angular_to_physical` first -- or if any component has nonzero
                `PA_twist` (an axisymmetric system can't have isophote twist).
            MGEDeprojectionError: If any component has no real solution at this
                inclination (``q' < cos(i)``), or an intrinsic `q` outside
                TNT's ``0 < q <= 1`` convention (``p`` is always 1 here).
        """
        if not self.sigma.unit.is_equivalent(au.m):
            raise ValueError(
                "deproject_axisymmetric requires physical (length) sigma; "
                "call angular_to_physical(distance) first."
            )
        if not bool(jnp.allclose(self.PA_twist.ustrip("rad"), 0.0)):
            raise ValueError(
                "deproject_axisymmetric requires PA_twist == 0 for every "
                "component (an axisymmetric system has no isophote twist)."
            )

        cos_i = jnp.cos(inclination.ustrip("rad"))
        sin_i = jnp.sin(inclination.ustrip("rad"))
        q_obs = self.q.ustrip("")
        q_intr = jnp.sqrt(q_obs**2 - cos_i**2) / sin_i

        _check_axial_ratios(p=jnp.ones_like(q_intr), q=q_intr)

        I_3d = self.I * (q_obs / (jnp.sqrt(2 * jnp.pi) * q_intr)) / self.sigma

        return Deprojected3DMGE(
            I=I_3d,
            sigma=self.sigma,
            p=Quantity(jnp.ones_like(q_intr), ""),
            q=Quantity(q_intr, ""),
        )

    def deproject_triaxial(
        self, theta: Quantity, phi: Quantity, psi: Quantity
    ) -> Deprojected3DMGE:
        """Deproject to an intrinsic 3D MGE, assuming triaxiality.

        General triaxial MGE deprojection (de Zeeuw & Franx 1989; Cappellari 2002 eqs.
        6-8; van den Bosch et al. 2008 eqs. 6-9), solving for each Gaussian's intrinsic
        axial ratios ``p`` (``B/A``) and ``q`` (``C/A``) given its observed axial ratio
        and the viewing geometry. Unlike `deproject_axisymmetric`, the intrinsic `sigma`
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
        both cases raise `MGEDeprojectionError` (this is an eager Python check, not
        JAX-traceable; see that exception's own note if `theta`/`phi`/`psi` are ever
        evaluated under `jax.jit`/`jax.vmap`).

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

        theta_r = theta.ustrip("rad")
        phi_r = phi.ustrip("rad")
        psi_r = psi.ustrip("rad") + self.PA_twist.ustrip("rad")
        delta = 1 - self.q.ustrip("") ** 2

        cos_theta, sin_theta = jnp.cos(theta_r), jnp.sin(theta_r)
        sec_theta = 1 / cos_theta
        cot_phi = 1 / jnp.tan(phi_r)
        tan_phi = jnp.tan(phi_r)
        cos_psi, sin_psi = jnp.cos(psi_r), jnp.sin(psi_r)
        cos_2psi, sin_2psi = jnp.cos(2 * psi_r), jnp.sin(2 * psi_r)

        denom = 2 * sin_theta**2 * (
            delta * cos_psi * (cos_psi + cot_phi * sec_theta * sin_psi) - 1
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
        _check_axial_ratios(p=p_intr, q=q_intr)

        q_obs = self.q.ustrip("")
        cos_phi, sin_phi = jnp.cos(phi_r), jnp.sin(phi_r)
        u = jnp.sqrt(
            jnp.sqrt(
                p_intr**2 * cos_theta**2
                + q_intr**2 * sin_theta**2 * (p_intr**2 * cos_phi**2 + sin_phi**2)
            )
            / q_obs
        )
        sigma_intr = self.sigma / u

        I_3d = (
            self.I
            * (u**3 * q_obs / (jnp.sqrt(2 * jnp.pi) * p_intr * q_intr))
            / self.sigma
        )

        return Deprojected3DMGE(
            I=I_3d, sigma=sigma_intr, p=Quantity(p_intr, ""), q=Quantity(q_intr, "")
        )


class LightMGE(AbstractMGE):
    """An MGE of a surface-brightness distribution (``I`` in e.g. Lsun/arcsec2)."""

    _intensity_attr: ClassVar[str] = "power"

    def to_mass(self, m_over_l: Quantity) -> MassMGE:
        """Convert to a MassMGE given a mass-to-light ratio.

        `sigma`, `q`, and `PA_twist` are unaffected and carried over unchanged -- only
        `I` (and hence what it represents) changes.

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
            I=self.I * m_over_l, sigma=self.sigma, q=self.q, PA_twist=self.PA_twist
        )


class MassMGE(AbstractMGE):
    """An MGE of a mass surface-density distribution (``I`` in e.g. Msun/arcsec2)."""

    _intensity_attr: ClassVar[str] = "mass"


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
    return -r / (2 * a) * jnp.exp(-a * r**2) + jnp.sqrt(jnp.pi) / (
        4 * a**1.5
    ) * erf(jnp.sqrt(a) * r)


class Deprojected3DMGE(eqx.Module):
    """An intrinsic (3D) MGE, produced by deprojecting a `LightMGE`/`MassMGE`.

    Each Gaussian component is described by its peak (3D) density ``I``, intrinsic width
    ``sigma``, and intrinsic axial ratios ``p`` (``B/A``) and ``q`` (``C/A``).
    `deproject_axisymmetric` always returns the same `sigma` as the projected MGE it
    came from; `deproject_triaxial` generally does not (see its docstring). An
    axisymmetric deprojection always has ``p == 1``, since axisymmetric ellipsoids have
    no intermediate axis.
    """

    I: Quantity
    sigma: Quantity
    p: Quantity
    q: Quantity

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
        a = (
            x2 + y2 / p.reshape(shape) ** 2 + z2 / q.reshape(shape) ** 2
        ) / (2 * sigma.reshape(shape) ** 2)

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


def read_mge(path: str | Path, unit_system: AbstractUnitSystem) -> AbstractMGE:
    """Read an MGE from an ECSV file, inferring whether it's light or mass.

    The kind is inferred from the declared unit of the file's ``I`` column: whichever of
    `LightMGE` (power/angle**2) or `MassMGE` (mass/angle**2) it is dimensionally
    consistent with.

    Args:
        path: Path to the ECSV file.
        unit_system: The unit system to convert the columns into.

    Returns:
        A `LightMGE` or `MassMGE`, whichever matches the file's ``I`` column.

    Raises:
        ValueError: If the ``I`` column's unit doesn't match any known MGE kind.
    """
    table = QTable.read(path, format="ascii.ecsv")
    intensity_unit = table["I"].unit

    for cls in _MGE_CLASSES:
        target_unit = cls._surface_intensity_unit(unit_system)
        if intensity_unit.is_equivalent(target_unit):
            return cls.from_qtable(table, unit_system)

    expected = [cls._surface_intensity_unit(unit_system) for cls in _MGE_CLASSES]
    raise ValueError(
        f"Could not infer MGE kind for {path}: its I column has unit "
        f"{intensity_unit!r}, which is not equivalent to any of {expected!r}."
    )


def build_mges(
    mges: Mapping[str, str],
    input_directory: str | Path,
    unit_system: AbstractUnitSystem,
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
        mges: Mapping of unique identifiers to ECSV filenames, e.g. a
            resolved configuration's ``MGEs`` section.
        input_directory: Directory that each filename is resolved against,
            e.g. a resolved configuration's ``io_settings.input_directory``.
        unit_system: The unit system to convert each MGE's columns into.
        distance: The distance to the object, e.g. a resolved
            configuration's ``system_attributes.distance``.

    Returns:
        A dict mapping each identifier to its physical-unit `LightMGE` or
        `MassMGE`.
    """
    directory = Path(input_directory)
    return {
        name: read_mge(directory / filename, unit_system).angular_to_physical(distance)
        for name, filename in mges.items()
    }
