"""Accuracy of TNT's triaxial MGE potential `_gradient` vs. an independent galpy build.

TNT's `TriaxialLightMGEPotential`/`TriaxialMassMGEPotential` turn a deprojected
MGE into a `galax.potential.CompositePotential` of one
`galax.potential.TriaxialGaussianPotential` per Gaussian
(`tnt.potential.triaxial_mge._galax_potential_from_deprojected`), and its
acceleration is `jax.grad` of that potential. Each galax component evaluates the
Chandrasekhar (1969) ellipsoidal `tau`-integral with a *fixed* 50-node
Gauss-Legendre rule (galax's default `integration_order`, which
`_galax_potential_from_deprojected` does not override).

galpy's `TriaxialGaussianPotential` (Emsellem et al. 1994; Bovy) is an
independent implementation of the same density,
`rho = amp / ((2 pi)^{3/2} sigma^3 b c) * exp(-m^2 / 2 sigma^2)`,
`m^2 = x^2 + y^2/b^2 + z^2/c^2`,
matching galax's `(m_tot, r_s, q1, q2) <-> (amp, sigma, b, c)` exactly. Its
force integrals also use Gauss-Legendre, but the order (`glorder`) is a free
parameter, so a high-order galpy build is a genuine accuracy reference for the
shipped order-50 galax gradient.

galpy is NOT a TNT dependency. Run this standalone:

    uv run --with galpy python aidocs/accuracy_mge_gradient_vs_galpy.py

It writes `aidocs/accuracy_mge_gradient_vs_galpy.csv` (one row per
evaluation point) and prints a regime summary.

Unit handling: galpy is configured with `ro = 1 kpc` and `vo` chosen so that
one galpy natural time unit is exactly 1 Myr; with `turn_physical_off()` its
raw forces are then already in `kpc / Myr^2`, the same as
`galax` under `unitsystem("galactic")`. A one-point cross-check against galax
at the start asserts this holds.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import numpy as np
import unxt as u

jax.config.update("jax_enable_x64", True)

from tnt.mge import Deprojected3DMGE
from tnt.potential.triaxial_mge import (
    _galax_potential_from_deprojected,
)

OUT_CSV = Path(__file__).with_suffix(".csv")

# --- galpy natural units == galactic (kpc, Myr, Msun) ----------------------
_RO = 1.0
# galpy: time_in_Gyr = ro / vo * 0.9777922216731283 ; want 1 Myr = 1e-3 Gyr.
_VO = _RO * 0.9777922216731283 / 1.0e-3

UNITS = u.unitsystem("galactic")

# --- the test MGE: 6 Gaussians spanning sigma_max / sigma_min = 600 -------
# Genuinely triaxial (0 < q <= p < 1), plausible run of shapes.
SIGMA_KPC = np.array([0.05, 0.15, 0.5, 2.0, 8.0, 30.0])
M_TOT_MSUN = np.array([2.0e9, 5.0e9, 1.0e10, 3.0e10, 5.0e10, 8.0e10])
P_RATIO = np.array([0.95, 0.90, 0.85, 0.80, 0.78, 0.75])  # b = q1
Q_RATIO = np.array([0.85, 0.80, 0.70, 0.62, 0.58, 0.55])  # c = q2

# --- radius sweep: 1e-3 kpc (deep inside sigma_min) to 1e3 kpc (far outside
#     sigma_max), along five rays (three principal axes + two skew). --------
RADII_KPC = np.logspace(-3.0, 3.0, 73)
RAYS = {
    "x-axis": np.array([1.0, 0.0, 0.0]),
    "y-axis": np.array([0.0, 1.0, 0.0]),
    "z-axis": np.array([0.0, 0.0, 1.0]),
    "diagonal": np.array([1.0, 1.0, 1.0]),
    "skew": np.array([2.0, 1.0, 0.5]),
}
RAYS = {k: v / np.linalg.norm(v) for k, v in RAYS.items()}

# galax integration orders to test: 50 is what ships; the rest show whether a
# residual is just quadrature order (and would close by raising it).
GALAX_ORDERS = (50, 100, 200, 500)
GALPY_REF_ORDER = 2000
GALPY_MATCH_ORDER = 50  # same order as shipped galax, independent implementation


def _peak_density_msun_kpc3() -> np.ndarray:
    return M_TOT_MSUN / (P_RATIO * Q_RATIO * (2.0 * np.pi) ** 1.5 * SIGMA_KPC**3)


def _galax_potential(integration_order: int):
    """The shipped MGE->galax build.

    `integration_order == 50` goes through the exact TNT code path
    (`_galax_potential_from_deprojected`, which does not set
    `integration_order`, so galax's default 50 applies). Higher orders
    rebuild the identical composite with the order raised, to show whether
    any order-50 residual is just quadrature truncation.
    """
    deprojected = Deprojected3DMGE(
        I=u.Quantity(np.asarray(_peak_density_msun_kpc3()), "Msun/kpc3"),
        sigma=u.Quantity(np.asarray(SIGMA_KPC), "kpc"),
        p=u.Quantity(np.asarray(P_RATIO), ""),
        q=u.Quantity(np.asarray(Q_RATIO), ""),
    )
    if integration_order == 50:
        return _galax_potential_from_deprojected(deprojected, UNITS)

    import galax.potential as gp

    components = {
        str(i): gp.TriaxialGaussianPotential(
            m_tot=deprojected.I[i]
            * deprojected.p[i]
            * deprojected.q[i]
            * (2.0 * np.pi) ** 1.5
            * deprojected.sigma[i] ** 3,
            r_s=deprojected.sigma[i],
            q1=deprojected.p[i],
            q2=deprojected.q[i],
            units=UNITS,
            integration_order=integration_order,
        )
        for i in range(len(SIGMA_KPC))
    }
    return gp.CompositePotential(components, units=UNITS)


def _galax_accel(pot, xyz_kpc: np.ndarray) -> np.ndarray:
    """xyz_kpc: (N, 3) -> acceleration (N, 3) in kpc / Myr^2."""
    q = u.Quantity(np.asarray(xyz_kpc), "kpc")
    t = u.Quantity(0.0, "Myr")
    grad = pot.gradient(q, t).ustrip("kpc / Myr2")
    return -np.asarray(grad)


def _galpy_potential(glorder: int):
    import astropy.units as au
    from galpy.potential import TriaxialGaussianPotential

    components = [
        TriaxialGaussianPotential(
            amp=M_TOT_MSUN[i] * au.Msun,
            sigma=SIGMA_KPC[i] * au.kpc,
            b=float(P_RATIO[i]),
            c=float(Q_RATIO[i]),
            glorder=glorder,
            ro=_RO,
            vo=_VO,
        )
        for i in range(len(SIGMA_KPC))
    ]
    for c in components:
        c.turn_physical_off()
    return components


def _galpy_accel(components, xyz_kpc: np.ndarray) -> np.ndarray:
    from galpy.potential import (
        evaluatephitorques,
        evaluateRforces,
        evaluatezforces,
    )

    xyz = np.asarray(xyz_kpc)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r_cyl = np.hypot(x, y)
    phi = np.arctan2(y, x)
    out = np.empty_like(xyz)
    for k in range(xyz.shape[0]):
        fz = evaluatezforces(components, r_cyl[k], z[k], phi=phi[k], use_physical=False)
        if r_cyl[k] == 0.0:
            # On the z-axis the aligned-ellipsoid force has no x/y part by
            # reflection symmetry; the cylindrical phi-torque / R is 0/0.
            out[k] = (0.0, 0.0, fz)
            continue
        fr = evaluateRforces(components, r_cyl[k], z[k], phi=phi[k], use_physical=False)
        ft = (
            evaluatephitorques(
                components, r_cyl[k], z[k], phi=phi[k], use_physical=False
            )
            / r_cyl[k]
        )
        out[k, 0] = fr * np.cos(phi[k]) - ft * np.sin(phi[k])
        out[k, 1] = fr * np.sin(phi[k]) + ft * np.cos(phi[k])
        out[k, 2] = fz
    return out


def _rel_l2(test: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.linalg.norm(test - ref, axis=-1) / np.linalg.norm(ref, axis=-1)


def _angle_deg(test: np.ndarray, ref: np.ndarray) -> np.ndarray:
    dot = np.sum(test * ref, axis=-1)
    cos = dot / (np.linalg.norm(test, axis=-1) * np.linalg.norm(ref, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def main() -> None:
    # Build the full grid: (ray, radius) -> xyz.
    ray_names, xyz_rows, radius_rows = [], [], []
    for name, direction in RAYS.items():
        for radius in RADII_KPC:
            ray_names.append(name)
            radius_rows.append(radius)
            xyz_rows.append(direction * radius)
    xyz = np.asarray(xyz_rows)
    radius_rows = np.asarray(radius_rows)

    # --- unit cross-check: galpy natural forces must already be kpc/Myr^2 ---
    check_xyz = np.array([[1.3, 0.7, 0.4]])
    a_galax = _galax_accel(_galax_potential(GALPY_REF_ORDER), check_xyz)[0]
    a_galpy = _galpy_accel(_galpy_potential(GALPY_REF_ORDER), check_xyz)[0]
    rel = np.linalg.norm(a_galax - a_galpy) / np.linalg.norm(a_galpy)
    assert rel < 1e-6, (
        f"unit calibration failed: rel diff {rel:.2e}\n{a_galax}\n{a_galpy}"
    )
    print(f"unit cross-check ok (galax order {GALPY_REF_ORDER} vs galpy: {rel:.2e})")

    # --- reference: high-order galpy, checked converged against 2x order --
    ref = _galpy_accel(_galpy_potential(GALPY_REF_ORDER), xyz)
    ref_check = _galpy_accel(_galpy_potential(2 * GALPY_REF_ORDER), xyz)
    ref_selfconv = _rel_l2(ref, ref_check)

    series = {
        f"galax-{order}": _galax_accel(_galax_potential(order), xyz)
        for order in GALAX_ORDERS
    }
    series[f"galpy-{GALPY_MATCH_ORDER}"] = _galpy_accel(
        _galpy_potential(GALPY_MATCH_ORDER), xyz
    )

    rows = []
    for i in range(xyz.shape[0]):
        row = {
            "ray": ray_names[i],
            "radius_kpc": radius_rows[i],
            "ref_accel_mag": float(np.linalg.norm(ref[i])),
            "ref_selfconv_rel_l2": float(ref_selfconv[i]),
        }
        for label, accel in series.items():
            row[f"{label}_rel_l2"] = float(_rel_l2(accel[i : i + 1], ref[i : i + 1])[0])
            row[f"{label}_angle_deg"] = float(
                _angle_deg(accel[i : i + 1], ref[i : i + 1])[0]
            )
        rows.append(row)

    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")

    # --- regime summary -------------------------------------------------
    sigma_min, sigma_max = SIGMA_KPC.min(), SIGMA_KPC.max()
    regimes = {
        f"inside sigma_min ({sigma_min:g} kpc)": radius_rows < sigma_min,
        "within the MGE": (radius_rows >= sigma_min) & (radius_rows <= sigma_max),
        f"outside sigma_max ({sigma_max:g} kpc)": radius_rows > sigma_max,
    }
    labels = [*series.keys(), "ref_selfconv"]
    print(
        f"\ngalpy reference self-convergence (order {GALPY_REF_ORDER} vs "
        f"{2 * GALPY_REF_ORDER}): max {ref_selfconv.max():.2e} -- must sit well "
        "below every error below for the reference to be trustworthy."
    )
    print("\nmax relative L2 error of the acceleration, by regime:")
    header = "  " + f"{'regime':<34}" + "".join(f"{lab:>14}" for lab in labels)
    print(header)
    for regime, mask in regimes.items():
        cells = "".join(
            f"{max(r[f'{lab}_rel_l2'] for r, m in zip(rows, mask) if m):>14.2e}"
            for lab in labels
        )
        print(f"  {regime:<34}{cells}")

    print("\nmax direction error (deg) of the acceleration, by regime:")
    header_a = "  " + f"{'regime':<34}" + "".join(f"{lab:>14}" for lab in series)
    print(header_a)
    for regime, mask in regimes.items():
        cells = "".join(
            f"{max(r[f'{lab}_angle_deg'] for r, m in zip(rows, mask) if m):>14.2e}"
            for lab in series
        )
        print(f"  {regime:<34}{cells}")

    print("\nradius (kpc) and ray where each series is worst:")
    for lab in series:
        worst = max(rows, key=lambda r: r[f"{lab}_rel_l2"])
        print(
            f"  {lab:<12} rel_l2 {worst[f'{lab}_rel_l2']:.2e} "
            f"at r = {worst['radius_kpc']:.3g} kpc ({worst['ray']})"
        )


if __name__ == "__main__":
    main()
