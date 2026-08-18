"""Unit tests for `tnt.parameter_generator`.

`tnt.parameter_generator` imports `tnt.all_models`, which (via `tnt.model`/
`tnt.potential`) imports `galax.potential` at module level, and this venv's
installed `galax`/`equinox` versions are mutually incompatible
(`ImportError: cannot import name '_has_dataclass_init' from
'equinox._module'`) -- an unrelated, pre-existing environment issue. Stub
out `galax`/`galax.potential` before importing anything from `tnt` that
would pull in that chain, so these tests can run regardless. Remove this
stub once the real dependency conflict is fixed.
"""

from __future__ import annotations

import sys
import types

if "galax" not in sys.modules:
    _fake_galax = types.ModuleType("galax")
    _fake_galax_potential = types.ModuleType("galax.potential")

    class _FakeAbstractPotentialBase:
        pass

    _fake_galax_potential.AbstractPotentialBase = _FakeAbstractPotentialBase
    _fake_galax.potential = _fake_galax_potential
    sys.modules["galax"] = _fake_galax
    sys.modules["galax.potential"] = _fake_galax_potential

from tnt import configuration_validation
from tnt.parameter_generator import _GENERATOR_CLASSES


def test_generator_settings_keys_match_the_real_classes() -> None:
    """`configuration_validation._GENERATOR_SETTINGS_KEYS` is duplicated,
    plain-data information -- it can't import these classes directly (see
    its own comment for why: preparation-phase code shouldn't depend on
    execution-phase modules). This is the regression test that keeps that
    duplicated data in sync with each class's own
    `_required_generator_settings`.
    """
    expected = {
        generator_cls._type: generator_cls._required_generator_settings
        for generator_cls in _GENERATOR_CLASSES
    }
    assert configuration_validation._GENERATOR_SETTINGS_KEYS == expected
