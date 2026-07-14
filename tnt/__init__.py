"""Top-level package for TNT."""

import logging as _logging

from tnt.configuration import Configuration, configuration_session
from tnt.logging import LoggingSession, configure_logging, configure_worker_logging

__all__ = [
    "Configuration",
    "LoggingSession",
    "configure_logging",
    "configure_worker_logging",
    "configuration_session",
]

_package_logger = _logging.getLogger(__name__)
if not any(
    isinstance(handler, _logging.NullHandler) for handler in _package_logger.handlers
):
    _package_logger.addHandler(_logging.NullHandler())
