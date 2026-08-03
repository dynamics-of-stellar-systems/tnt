"""Observed kinematic data sets and how to compare a model against them.

Signature-only scaffold: every method raises `NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import equinox as eqx
import jax.numpy as jnp
from unxt import AbstractUnitSystem

from tnt.mge import LightMGE, MassMGE
from tnt.orbit_library import OrbitLibrary
from tnt.spatial_binnings import ProjectedBinning


class AbstractKinematicData(eqx.Module):
    """One named, observed kinematic data set.

    `_type` matches a `kinematic_data.<name>` config entry's `type` field.
    `binning` is the `ProjectedBinning` the observations are already binned
    into (`kinematic_data.<name>.binning`, resolved against
    `spatial_binnings`). `mge` is at most one resolved MGE
    (`kinematic_data.<name>.mge`, resolved against `MGEs`) -- never more
    than one, but `None` where a data set doesn't need one at all (e.g.
    `ProperMotionsKinematicData`, which compares velocities directly rather
    than a LOSVD/mass-model projection).
    """

    _type: ClassVar[str]
    binning: ProjectedBinning
    mge: LightMGE | MassMGE | None

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        binnings: Mapping[str, ProjectedBinning],
        mges: Mapping[str, LightMGE | MassMGE],
        unit_system: AbstractUnitSystem,
    ) -> Self:
        """Build one kinematic data set from its resolved config entry.

        Args:
            settings: A resolved `kinematic_data.<name>` config entry.
            binnings: Named `ProjectedBinning`s to resolve `settings.binning`
                against.
            mges: Named MGEs to resolve `settings.mge` against, if present --
                this data set stores at most the one MGE it names, never
                the whole registry.
            unit_system: The unit system to convert this data set's
                quantities into.
        """
        raise NotImplementedError

    def design_matrix(self, orbit_library: OrbitLibrary) -> jnp.ndarray:
        """Each orbit's predicted observable, projected into this data set's bins.

        The building block `AbstractWeightSolver` assembles into its linear
        system -- the one place an `OrbitLibrary` is projected onto a
        `ProjectedBinning`.
        """
        raise NotImplementedError


class GaussHermiteKinematicData(AbstractKinematicData):
    """A Gauss-Hermite LOSVD moment expansion kinematic data set."""

    _type: ClassVar[str] = "gauss_hermite"


class BayesLOSVDKinematicData(AbstractKinematicData):
    """A `BayesLOSVD`-derived, binned LOSVD kinematic data set."""

    _type: ClassVar[str] = "bayes_losvd"


class ProperMotionsKinematicData(AbstractKinematicData):
    """A proper-motions kinematic data set."""

    _type: ClassVar[str] = "proper_motions"


def build_kinematic_data(
    kinematic_data: Mapping[str, Mapping[str, Any]],
    binnings: Mapping[str, ProjectedBinning],
    mges: Mapping[str, LightMGE | MassMGE],
    unit_system: AbstractUnitSystem,
) -> dict[str, AbstractKinematicData]:
    """Build the named `AbstractKinematicData` sets from a resolved configuration.

    Args:
        kinematic_data: A resolved configuration's `kinematic_data` section.
        binnings: Named `ProjectedBinning`s, e.g. from
            `tnt.spatial_binnings.build_spatial_binnings`.
        mges: Named MGEs, e.g. from `tnt.mge.build_mges`.
        unit_system: The unit system to convert each data set's quantities into.

    Returns:
        A dict mapping each identifier to its `AbstractKinematicData`.
    """
    raise NotImplementedError
