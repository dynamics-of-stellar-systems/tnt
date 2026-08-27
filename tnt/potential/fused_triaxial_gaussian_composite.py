"""A `galax.potential` extension: one shared Gauss-Legendre quadrature
across every `TriaxialGaussianPotential` child of a composite potential,
instead of one independent 50-point quadrature per child.

Deliberately implemented with zero TNT imports -- only `galax`, `jax`,
`unxt`, `equinox` -- so it stands entirely on its own as a `galax.potential`
extension. It lives here for now because this is where it's developed and
tested first; per GitHub discussion #34
(dynamics-of-stellar-systems/tnt#34), "scan-fused" (summing every
component's integrand at each shared node before integrating once, via
`jax.lax.scan` over the nodes) is 4-11x faster than the shipped composite
for batched gradient evaluation (= orbit-integration acceleration) on the
actual VSC-5 cluster CPU deployment target, even though the opposite
ranking holds on a laptop. If galax's maintainers are interested, this is
intended to become a PR into `galax.potential` itself, not a permanent TNT
fork of composite-potential logic -- see the follow-up issues tracking
that decision, GPU-hardware validation, and wiring this into
`tnt.potential.triaxial_mge` (deliberately out of scope here).

`FusedTriaxialGaussianCompositePotential` fuses `_potential`/`_gradient`
only. `_density`/`_laplacian`/`_hessian` are deliberately left inherited
from `galax.potential.AbstractCompositePotential` -- correct, just not
fused -- so this remains a safe, fully-functional drop-in for
`galax.potential.CompositePotential` wherever every component is a
`TriaxialGaussianPotential`, even though only two of its five potential
methods are actually fast. Combining with any other potential (`|`, `+`)
also falls back to `AbstractCompositePotential`'s inherited behavior,
which always builds a plain `galax.potential.CompositePotential` --
correct, since the fusion premise needs homogeneous
`TriaxialGaussianPotential` children.
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


class FusedTriaxialGaussianCompositePotential(gp.AbstractCompositePotential):
    """A `CompositePotential` of `TriaxialGaussianPotential` components,
    evaluated with one shared quadrature instead of one per component.

    See the module docstring for the motivation and scope. Construction
    mirrors `galax.potential.CompositePotential` exactly (same
    `units="first"` default, same per-child unit-system check), plus one
    extra invariant this class needs that plain `CompositePotential`
    doesn't: every child must share the same `integration_order`, since
    fusing depends on all of them integrating against one shared node set.
    """

    parameters: ClassVar = CompositeParametersAttribute(MappingProxyType({}))

    _data: dict[str, gp.TriaxialGaussianPotential]
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
        **kwargs: gp.TriaxialGaussianPotential,
    ) -> None:
        data: dict[str, gp.TriaxialGaussianPotential] = dict(potentials, **kwargs)
        for key, potential in data.items():
            if not isinstance(potential, gp.TriaxialGaussianPotential):
                msg = (
                    f"component {key!r} is a {type(potential).__name__}, not a "
                    "TriaxialGaussianPotential -- "
                    "FusedTriaxialGaussianCompositePotential only supports "
                    "homogeneous TriaxialGaussianPotential children."
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
        are supported, matching `TriaxialGaussianPotential._potential`'s own
        generality; degenerates to a single scalar reduction inside the
        `lax.scan` body for an unbatched call.
        """
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)

        children = tuple(self.values())
        length_unit = self.units["length"]
        dimensionless_unit = self.units["dimensionless"]
        r_s = jnp.stack(
            [c.r_s(t, ustrip=length_unit) for c in children], axis=-1
        )  # (*Bt, N)
        q1 = jnp.stack(
            [c.q1(t, ustrip=dimensionless_unit) for c in children], axis=-1
        )
        q2 = jnp.stack(
            [c.q2(t, ustrip=dimensionless_unit) for c in children], axis=-1
        )
        prefactor = jnp.stack(
            [
                (
                    -2.0
                    * jnp.pi
                    * c.constants["G"]
                    * c.rho0(t)
                    * c.r_s(t) ** 2
                    * c.q1(t)
                    * c.q2(t)
                ).ustrip(self.units["specific energy"])
                for c in children
            ],
            axis=-1,
        )  # (*Bt, N)
        q1sq, q2sq = q1**2, q2**2

        x2 = xyz[..., 0:1] ** 2  # (*B, 1) -- trailing axis broadcasts against N
        y2 = xyz[..., 1:2] ** 2
        z2 = xyz[..., 2:3] ** 2

        def node_step(
            carry: gt.BBtSz0, node: tuple[gt.Array, gt.Array]
        ) -> tuple[gt.BBtSz0, None]:
            s, w = node
            s2 = s * s
            denom = jnp.sqrt(
                ((q1sq - 1) * s2 + 1) * ((q2sq - 1) * s2 + 1)
            )  # (*Bt, N)
            xi2 = (
                s2
                * (
                    x2
                    + y2 / (1 + (q1sq - 1) * s2)
                    + z2 / (1 + (q2sq - 1) * s2)
                )
                / r_s**2
            )  # (*batch, N)
            integrand = 2.0 * jnp.exp(-xi2 / 2) / denom
            node_value = jnp.sum(prefactor * integrand, axis=-1)  # (*batch,)
            return carry + w * node_value, None

        batch_shape = jnp.broadcast_shapes(x2.shape[:-1], r_s.shape[:-1])
        carry0 = jnp.zeros(batch_shape)
        total, _ = jax.lax.scan(
            node_step, carry0, (self._integrator.x, self._integrator.w)
        )
        return total

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
