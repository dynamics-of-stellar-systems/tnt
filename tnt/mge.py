"""Multi-Gaussian Expansion (MGE) models."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Self

import astropy.units as au
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from astropy.table import QTable
from jax.scipy.special import erf
from unxt import AbstractUnitSystem, Quantity

from tnt import quantity_conversions


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

    I: Quantity  # noqa: E741
    sigma: Quantity
    q: Quantity
    PA_twist: Quantity

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
        intensity_unit = (
            getattr(unit_system, cls._intensity_attr) / unit_system.angle**2
        )
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
        I_physical = self.I * solid_angle / distance**2  # noqa: N806

        return type(self)(
            I=I_physical, sigma=sigma_physical, q=self.q, PA_twist=self.PA_twist
        )

    def physical_to_angular(self, distance: Quantity) -> Self:
        """Convert `sigma` and `I` from physical (length) to angular units.

        Inverse of `angular_to_physical`.

        Args:
            distance: The distance to the object.

        Returns:
            A new MGE with `sigma` in radians and `I` converted to match.
        """
        sigma_angular = quantity_conversions.physical_to_angular(self.sigma, distance)
        solid_angle = Quantity(1.0, f"{sigma_angular.unit}2")
        I_angular = self.I * distance**2 / solid_angle  # noqa: N806

        return type(self)(
            I=I_angular, sigma=sigma_angular, q=self.q, PA_twist=self.PA_twist
        )

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
            component. If ``q' < cos(i)`` for a component (no real solution at this
            inclination), that component's `q` is `nan` rather than raising --
            `inclination` may be a fit parameter under `jax.jit`, so this can't be a
            hard Python check; validate eagerly outside `jit` if you need one.

        Raises:
            ValueError: If `sigma` isn't in physical (length) units -- call
                `angular_to_physical` first -- or if any component has nonzero
                `PA_twist` (an axisymmetric system can't have isophote twist).
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

        I_3d = self.I * (q_obs / (jnp.sqrt(2 * jnp.pi) * q_intr)) / self.sigma  # noqa: N806

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
        2002 sec. 2.2.1); an unsolvable component produces `nan` rather than raising,
        since the viewing angles may be fit parameters under `jax.jit`.

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

        I_3d = (  # noqa: N806
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


# Fixed-order Gauss-Legendre quadrature nodes/weights on [-1, 1], used for the
# angular (theta, phi) integral in `Deprojected3DMGE.spherical_mass_grid`. A
# module-level constant since the order doesn't depend on any call's inputs.
_ANGULAR_QUAD_ORDER = 10
_GAUSS_LEGENDRE_NODES, _GAUSS_LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(
    _ANGULAR_QUAD_ORDER
)


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


class SphericalGrid(eqx.Module):
    """Cell edges of a spherical ``(r, theta, phi)`` grid, one octant.

    `r_edges` has ``n_r + 1`` edges bounding ``n_r`` bins spanning ``(0,
    infinity)``: ``r_edges[0] == 0``, ``r_edges[-1] == inf``, and the bins
    between are logarithmically spaced from `r_min` to `r_max`.

    `theta` bins (`cos_theta_edges`, ``n_theta + 1`` edges) are spaced linearly
    in ``cos(theta)`` rather than `theta` itself -- from ``cos(0) = 1`` to
    ``cos(pi/2) = 0`` -- so that every (theta, phi) cell spans equal solid
    angle at fixed r. Quenneville, Liepold & Ma (2021) sec. 4.4 found this 
    necessary to avoid under-sampling mass near the poles.
    
    `phi_edges` (``n_phi + 1`` edges) is linearly spaced over ``[0,pi/2]``.
    """

    r_edges: Quantity
    cos_theta_edges: Quantity
    phi_edges: Quantity

    def __init__(
        self, n_r: int, n_theta: int, n_phi: int, r_min: Quantity, r_max: Quantity
    ) -> None:
        """Build the grid's cell edges.

        Args:
            n_r: Number of radial bins; must be at least 2.
            n_theta: Number of polar-angle bins.
            n_phi: Number of azimuthal-angle bins.
            r_min: Inner edge of the logarithmically spaced radial region.
            r_max: Outer edge of the logarithmically spaced radial region.

        Raises:
            ValueError: If `n_r` is less than 3. (With only 2 bins, the single
                interior edge can't be both `r_min` and `r_max`.)
        """
        if n_r < 3:
            raise ValueError(f"n_r must be at least 3, got {n_r}")

        length_unit = r_min.unit
        r_min_v = r_min.ustrip(length_unit)
        r_max_v = r_max.ustrip(length_unit)

        interior_edges = jnp.geomspace(r_min_v, r_max_v, n_r - 1)
        r_edges = jnp.concatenate(
            [jnp.zeros(1), interior_edges, jnp.array([jnp.inf])]
        )

        self.r_edges = Quantity(r_edges, length_unit)
        self.cos_theta_edges = Quantity(jnp.linspace(1.0, 0.0, n_theta + 1), "")
        self.phi_edges = Quantity(jnp.linspace(0.0, jnp.pi / 2, n_phi + 1), "rad")

    @property
    def n_r(self) -> int:
        return self.r_edges.shape[0] - 1

    @property
    def n_theta(self) -> int:
        return self.cos_theta_edges.shape[0] - 1

    @property
    def n_phi(self) -> int:
        return self.phi_edges.shape[0] - 1


class Deprojected3DMGE(eqx.Module):
    """An intrinsic (3D) MGE, produced by deprojecting a `LightMGE`/`MassMGE`.

    Each Gaussian component is described by its peak (3D) density ``I``, intrinsic width
    ``sigma``, and intrinsic axial ratios ``p`` (``B/A``) and ``q`` (``C/A``).
    `deproject_axisymmetric` always returns the same `sigma` as the projected MGE it
    came from; `deproject_triaxial` generally does not (see its docstring). An
    axisymmetric deprojection always has ``p == 1``, since axisymmetric ellipsoids have
    no intermediate axis.
    """

    I: Quantity  # noqa: E741
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

        nodes = jnp.asarray(_GAUSS_LEGENDRE_NODES)
        weights = jnp.asarray(_GAUSS_LEGENDRE_WEIGHTS)

        # The sin(theta) dtheta Jacobian is exactly -d(cos(theta)), so
        # quadrature nodes/weights in cos(theta) already include it -- no extra
        # sin(theta) factor is needed when this is later contracted against the
        # integrand.
        cos_theta_edges = grid.cos_theta_edges.ustrip("")
        cos_theta_lo, cos_theta_hi = cos_theta_edges[1:], cos_theta_edges[:-1]
        cos_theta_mid = (cos_theta_hi + cos_theta_lo) / 2
        cos_theta_half = (cos_theta_hi - cos_theta_lo) / 2
        cos_theta_nodes = (
            cos_theta_mid[:, None] + cos_theta_half[:, None] * nodes[None, :]
        )
        theta_weights = cos_theta_half[:, None] * weights[None, :]  # (n_theta, Q)

        phi_edges = grid.phi_edges.ustrip("rad")
        phi_lo, phi_hi = phi_edges[:-1], phi_edges[1:]
        phi_mid, phi_half = (phi_hi + phi_lo) / 2, (phi_hi - phi_lo) / 2
        phi_nodes = phi_mid[:, None] + phi_half[:, None] * nodes[None, :]
        phi_weights = phi_half[:, None] * weights[None, :]  # (n_phi, Q)

        cos_theta = cos_theta_nodes  # (n_theta, Q)
        sin_theta = jnp.sqrt(1 - cos_theta_nodes**2)
        cos_phi = jnp.cos(phi_nodes)  # (n_phi, Q)
        sin_phi = jnp.sin(phi_nodes)

        # Direction-dependent factors of each component's rate parameter a(theta,
        # phi), such that a * r**2 = (x**2 + y**2/p**2 + z**2/q**2) / (2 sigma**2)
        # -- broadcast to (n_theta, Q, n_phi, Q).
        x2 = (sin_theta[:, :, None, None] * cos_phi[None, None, :, :]) ** 2
        y2 = (sin_theta[:, :, None, None] * sin_phi[None, None, :, :]) ** 2
        z2 = jnp.broadcast_to(cos_theta[:, :, None, None] ** 2, x2.shape)

        sigma = self.sigma.ustrip(length_unit)  # (G,)
        p = self.p.ustrip("")  # (G,)
        q = self.q.ustrip("")  # (G,)
        I = self.I.ustrip(self.I.unit)  # noqa: E741, N806

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
            "ja,kb,jakbn->njk", theta_weights, phi_weights, integrand
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
        target_unit = getattr(unit_system, cls._intensity_attr) / unit_system.angle**2
        if intensity_unit.is_equivalent(target_unit):
            return cls.from_qtable(table, unit_system)

    expected = [
        getattr(unit_system, cls._intensity_attr) / unit_system.angle**2
        for cls in _MGE_CLASSES
    ]
    raise ValueError(
        f"Could not infer MGE kind for {path}: its I column has unit "
        f"{intensity_unit!r}, which is not equivalent to any of {expected!r}."
    )
