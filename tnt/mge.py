"""Multi-Gaussian Expansion (MGE) models."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Self

import astropy.units as au
import equinox as eqx
import jax.numpy as jnp
from astropy.table import QTable
from unxt import AbstractUnitSystem, Quantity

from tnt import units


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
        sigma_physical = units.angular_to_physical(self.sigma, distance)
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
        sigma_angular = units.physical_to_angular(self.sigma, distance)
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


_MGE_CLASSES: tuple[type[AbstractMGE], ...] = (LightMGE, MassMGE)


def read_mge(path: str | Path, unit_system: AbstractUnitSystem) -> AbstractMGE:
    """Read an MGE from an ECSV file, inferring whether it's light or mass.

    The kind is inferred from the declared unit of the file's ``I`` column: whichever of
    `LightMGE` (power/angle**2) or `MassMGE` (mass/angle**2) it is dimensionally
    consistent with. This makes the check meaningful -- a file with the wrong kind of
    units for its intended use is rejected here, rather than silently accepted.

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
