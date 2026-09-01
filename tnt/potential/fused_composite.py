"""A `galax.potential` extension: one shared Gauss-Legendre quadrature
across every quadrature-based child of a composite potential, instead of
one independent quadrature per child.

Supports the three `galax.potential` types whose `_potential` is a
Gauss-Legendre quadrature of the Chandrasekhar (1969) ellipsoidal
``tau``-integral -- `TriaxialGaussianPotential`,
`AxisymmetricGaussianPotential` (its ``q1 = 1`` special case) and
`TriaxialNFWPotential`. All three share the integrand's denominator
``sqrt(((q1^2-1)s^2 + 1)((q2^2-1)s^2 + 1))`` and the ellipsoidal coordinate
``xi^2(s)``; they differ only in the profile factor ``Delta psi(xi^2)`` --
``2 exp(-xi^2/2)`` for the Gaussians, ``2/(1 + xi)`` for the NFW -- and in
the per-child prefactor ``-2 pi G rho0 r_s^2 q1 q2``. Children of different
profile types may be mixed (e.g. a stellar Gaussian MGE plus a triaxial NFW
halo); a mixed composite evaluates both profile factors over the whole
quadrature tensor and selects per component.

Deliberately implemented with zero TNT imports -- only `galax`, `jax`,
`unxt`, `equinox` -- so it stands entirely on its own as a `galax.potential`
extension. It lives here for now because this is where it's developed and
tested first. If galax's maintainers are interested, this is intended to
become a PR into `galax.potential` itself, not a permanent TNT fork of
composite-potential logic -- see the follow-up issues tracking that
decision and wiring this into `tnt.potential.triaxial_mge` (deliberately
out of scope here).

The fusion sums every component's integrand at each of the shared quadrature
nodes and integrates once ("integral of sum") -- mathematically identical to
the shipped composite's "sum of integrals", since integration is linear and
every component shares one fixed quadrature order. The reduction is done as a
single vectorized `jnp.sum` over one `(nodes, ..., N)` tensor ("array-fused").
Per GitHub discussion #34 (dynamics-of-stellar-systems/tnt#34), on GPU this is
~2x faster than the shipped composite for batched `gradient()` -- the
acceleration an orbit integrator queries at every Runge-Kutta stage -- and
wins at every batch size; on CPU it is a mild regression at large batches.
The cost is memory: the intermediate is `O(nodes * N * batch)`, where the
composite is `O(N + batch)`, so a large enough `N x batch` can exhaust device
memory. Batch-dimension chunking to bound the footprint is a follow-up, not
implemented here.

`FusedCompositePotential` fuses `_potential`/`_gradient` only.
`_density`/`_laplacian`/`_hessian` are deliberately left inherited from
`galax.potential.AbstractCompositePotential` -- correct, just not fused --
so this remains a safe, fully-functional drop-in for
`galax.potential.CompositePotential` wherever every component is one of the
three supported types, even though only two of its five potential methods
are actually fast. Combining with any other potential (`|`, `+`) also falls
back to `AbstractCompositePotential`'s inherited behavior, which always
builds a plain `galax.potential.CompositePotential` -- correct, since the
fusion premise needs children that share the quadrature.
"""

from __future__ import annotations

import functools as ft
from dataclasses import KW_ONLY
from types import MappingProxyType
from typing import Any, ClassVar

import equinox as eqx
import galax.potential as gp
import galax.potential.custom_types as gt
import jax
import quaxed.numpy as jnp
import unxt as u
from galax.potential._src.base import default_constants
from galax.potential._src.composite import ArgPotential, UnitsOptionEnum
from galax.potential._src.jax import vectorize_method
from galax.potential._src.params.attr import CompositeParametersAttribute
from galax.potential._src.utils import GaussLegendreIntegrator
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap
from zeroth import zeroth

# The `galax.potential` types whose `_potential` is a Gauss-Legendre
# quadrature of the same ellipsoidal `tau`-integral, grouped by their profile
# factor `Delta psi(xi^2)`.
_GAUSSIAN_TYPES: tuple[type, ...] = (
    gp.TriaxialGaussianPotential,
    gp.AxisymmetricGaussianPotential,
)
_NFW_TYPES: tuple[type, ...] = (gp.TriaxialNFWPotential,)
_SUPPORTED_TYPES: tuple[type, ...] = _GAUSSIAN_TYPES + _NFW_TYPES


class FusedCompositePotential(gp.AbstractCompositePotential):
    """A `CompositePotential` of Gauss-Legendre-quadrature components,
    evaluated with one shared quadrature instead of one per component.

    Every child must be a `TriaxialGaussianPotential`,
    `AxisymmetricGaussianPotential` or `TriaxialNFWPotential` (the
    `galax.potential` types built on that quadrature); the three may be
    mixed. See the module docstring for the motivation and scope.
    Construction mirrors `galax.potential.CompositePotential` exactly (same
    `units="first"` default, same per-child unit-system check), plus one
    extra invariant this class needs that plain `CompositePotential`
    doesn't: every child must share the same `integration_order`, since
    fusing depends on all of them integrating against one shared node set.
    """

    parameters: ClassVar = CompositeParametersAttribute(MappingProxyType({}))

    _data: dict[str, gp.AbstractSinglePotential]
    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(static=True, converter=u.unitsystem)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )
    integration_order: int = eqx.field(static=True, default=None)
    _integrator: GaussLegendreIntegrator = eqx.field(default=None)

    def __init__(
        self,
        potentials: ArgPotential = (),
        /,
        *,
        units: Any = UnitsOptionEnum.FIRST,
        constants: Any = default_constants,
        **kwargs: gp.AbstractSinglePotential,
    ) -> None:
        data: dict[str, gp.AbstractSinglePotential] = dict(potentials, **kwargs)
        for key, potential in data.items():
            if not isinstance(potential, _SUPPORTED_TYPES):
                msg = (
                    f"component {key!r} is a {type(potential).__name__}; "
                    "FusedCompositePotential only supports the Gauss-Legendre "
                    "quadrature potentials TriaxialGaussianPotential, "
                    "AxisymmetricGaussianPotential and TriaxialNFWPotential."
                )
                raise TypeError(msg)
        object.__setattr__(self, "_data", data)

        usys = (
            zeroth(data.values()).units
            if units is UnitsOptionEnum.FIRST
            else u.unitsystem(units)
        )
        if not all(p.units == usys for p in data.values()):
            msg = "all potentials must have the same unit system"
            raise ValueError(msg)
        object.__setattr__(self, "units", usys)

        object.__setattr__(
            self,
            "constants",
            ImmutableMap({k: v.decompose(usys) for k, v in constants.items()}),
        )
        self._apply_unitsystem()

        order = zeroth(data.values()).integration_order
        if not all(p.integration_order == order for p in data.values()):
            msg = (
                "all components must share the same integration_order -- the "
                "fused quadrature can only reuse one shared set of nodes."
            )
            raise ValueError(msg)
        object.__setattr__(self, "integration_order", order)
        object.__setattr__(
            self, "_integrator", GaussLegendreIntegrator.for_order(order)
        )

    # ==========================================================================
    # Fused potential + gradient

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        """Sum every component's integrand at each shared node, integrate once.

        Mathematically identical to `AbstractCompositePotential._potential`
        (summing each child's own independently-integrated potential) --
        see the module docstring. Arbitrary leading batch dims on `xyz`/`t`
        are supported, matching the per-child `_potential`'s own generality;
        the node reduction is a single `jnp.sum` over one `(nodes, *batch, N)`
        tensor, which degenerates to `(nodes, N)` for an unbatched call.
        """
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)

        children = tuple(self.values())
        length_unit = self.units["length"]
        u1 = self.units["dimensionless"]
        energy_unit = self.units["specific energy"]

        def q1_of(c: gp.AbstractSinglePotential) -> Any:
            # AxisymmetricGaussianPotential is the q1 = 1 special case and has
            # no `q1` field; the other two carry it explicitly.
            if isinstance(c, gp.AxisymmetricGaussianPotential):
                return jnp.ones_like(c.q2(t, ustrip=u1))
            return c.q1(t, ustrip=u1)

        def prefactor_of(c: gp.AbstractSinglePotential) -> Any:
            q1c = 1.0 if isinstance(c, gp.AxisymmetricGaussianPotential) else c.q1(t)
            return (
                -2.0
                * jnp.pi
                * c.constants["G"]
                * c.rho0(t)
                * c.r_s(t) ** 2
                * q1c
                * c.q2(t)
            ).ustrip(energy_unit)

        r_s = jnp.stack(
            [c.r_s(t, ustrip=length_unit) for c in children], axis=-1
        )  # (*Bt, N)
        q1 = jnp.stack([q1_of(c) for c in children], axis=-1)
        q2 = jnp.stack([c.q2(t, ustrip=u1) for c in children], axis=-1)
        prefactor = jnp.stack([prefactor_of(c) for c in children], axis=-1)  # (*Bt, N)
        q1sq, q2sq = q1**2, q2**2

        x2 = xyz[..., 0:1] ** 2  # (*B, 1) -- trailing axis broadcasts against N
        y2 = xyz[..., 1:2] ** 2
        z2 = xyz[..., 2:3] ** 2

        batch_shape = jnp.broadcast_shapes(x2.shape[:-1], r_s.shape[:-1])

        # Give the node dimension its own leading axis and reduce it in one
        # `jnp.sum`: `s2` broadcasts as (nodes, 1, ..., 1) against the
        # (*batch, N) per-node term, so the whole quadrature is one tensor.
        nodes, weights = self._integrator.x, self._integrator.w
        s = nodes.reshape((nodes.shape[0],) + (1,) * (len(batch_shape) + 1))
        s2 = s * s
        denom = jnp.sqrt(
            ((q1sq - 1) * s2 + 1) * ((q2sq - 1) * s2 + 1)
        )  # (nodes, *batch, N)
        xi2 = (
            s2 * (x2 + y2 / (1 + (q1sq - 1) * s2) + z2 / (1 + (q2sq - 1) * s2)) / r_s**2
        )  # (nodes, *batch, N)

        # The only per-profile branch: Delta psi(xi^2). Which family each
        # child belongs to is fixed at construction, so this selects a code
        # path at trace time -- no runtime dispatch -- except for a genuinely
        # mixed composite, which evaluates both factors over the whole tensor
        # and picks per component (one wasted transcendental per node for the
        # minority family; mixed composites are the uncommon case).
        is_nfw = tuple(isinstance(c, gp.TriaxialNFWPotential) for c in children)
        if not any(is_nfw):
            delta_psi = 2.0 * jnp.exp(-xi2 / 2)
        elif all(is_nfw):
            delta_psi = 2.0 / (1.0 + jnp.sqrt(xi2))
        else:
            nfw_mask = jnp.asarray(is_nfw)  # (N,) -- broadcasts against last axis
            delta_psi = jnp.where(
                nfw_mask,
                2.0 / (1.0 + jnp.sqrt(xi2)),
                2.0 * jnp.exp(-xi2 / 2),
            )

        integrand = delta_psi / denom
        per_node = jnp.sum(prefactor * integrand, axis=-1)  # (nodes, *batch)
        w = weights.reshape((weights.shape[0],) + (1,) * len(batch_shape))
        return jnp.sum(w * per_node, axis=0)  # (*batch,)

    @vectorize_method(signature="(3),()->(3)")
    @ft.partial(jax.jit)
    def _gradient(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz3:
        """`jax.grad` of the fused `_potential` -- no hand-derived gradient.

        Mirrors `AbstractPotential._gradient`'s own default (which
        `AbstractCompositePotential` otherwise shadows with a per-child sum
        of gradients); `vectorize_method` strips batch dims down to the
        core `(3,)`/`()` signature before calling, so this only ever needs
        `_potential`'s single-point case to be correct -- already fused via
        the shared quadrature above.
        """
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        return jax.grad(self._potential)(xyz, t)
