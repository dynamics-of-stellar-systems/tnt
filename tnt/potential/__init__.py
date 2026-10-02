"""Galactic potentials, assembled from named, `galax`-backed components.

`potential.<name>.type` names either a curated `galax.potential` class (see
`registry._SUPPORTED_GALAX_TYPES`, e.g. `"NFWPotential"`, `"PlummerPotential"`)
or one of four TNT-specific MGE composite potentials -- triaxial
(`"TriaxialLightMGEPotential"`/`"TriaxialMassMGEPotential"`) or oblate
axisymmetric (`"OblateLightMGEPotential"`/`"OblateMassMGEPotential"`) -- each
built from a named MGE, and pairs every supported class with each native
parameter's mass-rescale exponent.
`parameterization` is a separate, optional concern: when omitted,
`parameters` use the resolved type's own native constructor kwargs, with
physical dimensions read directly from `registry._SUPPORTED_GALAX_TYPES` (see
`raw_parameter_dimensions`). When given, `parameterization` names a
registered conversion from some other raw parameter convention into those
same native fields.

Each parameter keeps its own declared unit all the way through
construction -- `Potential.resolve`/`build` never coerce it into a shared
internal unit system (`galax`'s own `ParameterField` machinery already
converts generically at evaluation time; see `ResolvedPotentialComponent.build`).
Resolving a component's static structure (`type`/`parameterization`/`mge`)
is split from building it at a given point in parameter space
(`Potential.resolve`/`Potential.build`), so a caller building many
`Potential`s from the same configuration -- e.g. `ModelIterator`, once per
proposed point -- resolves once and reuses the result.

`Potential.generate_orbit_library` remains `NotImplementedError`.

Split across submodules by concern: `registry` (curated `galax` types, their
native parameters' dimensions/mass-rescale exponents, and TNT component /
parameterization registration), `nfw` (the `concentration_m200`
parameterization's self-contained numerics, and its `register_parameterization`
call), `components` (the abstract base and the native-`galax` component,
resolution/dispatch and eager runtime domain validation included),
`triaxial_mge`/`oblate_mge` (the MGE-backed
composite types, one sibling module per deprojection convention), and
`core` (`Potential` itself and the module-level helpers around it).
"""

from __future__ import annotations

from tnt.potential import nfw as _nfw  # noqa: F401 -- registers NFW parameterization
from tnt.potential.components import (
    AbstractPotentialComponent,
    GalaxPotentialComponent,
    ResolvedPotentialComponent,
)
from tnt.potential.core import Potential, build_potential, raw_potential_parameters
from tnt.potential.oblate_mge import (
    OblateLightMGEPotential,
    OblateMassMGEPotential,
)
from tnt.potential.registry import (
    InvalidPotentialParametersError,
    NativeParameter,
    raw_parameter_dimensions,
)
from tnt.potential.triaxial_mge import (
    TriaxialLightMGEPotential,
    TriaxialMassMGEPotential,
)

__all__ = [
    "AbstractPotentialComponent",
    "GalaxPotentialComponent",
    "InvalidPotentialParametersError",
    "NativeParameter",
    "OblateLightMGEPotential",
    "OblateMassMGEPotential",
    "Potential",
    "ResolvedPotentialComponent",
    "TriaxialLightMGEPotential",
    "TriaxialMassMGEPotential",
    "build_potential",
    "raw_parameter_dimensions",
    "raw_potential_parameters",
]
