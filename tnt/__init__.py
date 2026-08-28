"""Top-level package for TNT."""

import logging as _logging

import tnt.numerics as _numerics  # Apply TNT's JAX defaults before other imports.
from tnt.configuration import Configuration, configuration_session
from tnt.logging import LoggingSession, configure_logging, configure_worker_logging
from tnt.units import UnitSystems

del _numerics

__all__ = [
    "Configuration",
    "LoggingSession",
    "UnitSystems",
    "configuration_session",
    "configure_logging",
    "configure_worker_logging",
]

_package_logger = _logging.getLogger(__name__)
if not any(
    isinstance(handler, _logging.NullHandler) for handler in _package_logger.handlers
):
    _package_logger.addHandler(_logging.NullHandler())
