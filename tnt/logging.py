"""Explicit, application-safe logging for TNT executions."""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import sys
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

_LOGGER_NAME = "tnt"
_OWNED_HANDLER_ATTRIBUTE = "_tnt_owned_handler"
_SESSION_LOCK = threading.RLock()
_ACTIVE_SESSION: LoggingSession | None = None


class LoggingSession:
    """Own TNT-local handlers and restore the previous package logger state."""

    def __init__(
        self,
        *,
        log_queue: Any,
        queue_handler: logging.handlers.QueueHandler,
        listener: logging.handlers.QueueListener,
        output_handlers: list[logging.Handler],
        logfile_path: Path | None,
        previous_level: int,
        previous_propagate: bool,
    ) -> None:
        """Store resources created by :func:`configure_logging`."""
        self.worker_queue = log_queue
        self.logfile_path = logfile_path
        self._queue_handler = queue_handler
        self._listener = listener
        self._output_handlers = output_handlers
        self._previous_level = previous_level
        self._previous_propagate = previous_propagate
        self._closed = False

    @property
    def closed(self) -> bool:
        """Return whether this session has released its logging resources."""
        return self._closed

    def close(self) -> None:
        """Stop logging and restore the TNT package logger."""
        global _ACTIVE_SESSION

        with _SESSION_LOCK:
            if self._closed:
                return

            package_logger = logging.getLogger(_LOGGER_NAME)
            if self._queue_handler in package_logger.handlers:
                package_logger.removeHandler(self._queue_handler)

            self._listener.stop()
            self._queue_handler.close()
            for handler in self._output_handlers:
                handler.close()
            self.worker_queue.close()
            self.worker_queue.join_thread()

            if _ACTIVE_SESSION is self:
                package_logger.setLevel(self._previous_level)
                package_logger.propagate = self._previous_propagate
                _ACTIVE_SESSION = None
            self._closed = True

    def __enter__(self) -> Self:
        """Return this active logging session."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the session when leaving a context manager."""
        self.close()


def configure_logging(configuration: Mapping[str, Any]) -> LoggingSession:
    """Configure isolated TNT file and terminal logging for one execution.

    The function changes only the ``tnt`` package logger. It never changes,
    shuts down, or reloads the root logger. Calling it again closes the prior
    TNT logging session before installing a new one.

    Args:
        configuration: Fully resolved TNT configuration mapping.

    Returns:
        An active logging session. Use it as a context manager or call
        :meth:`LoggingSession.close` when execution finishes.

    Raises:
        TypeError: If required configuration sections have the wrong type.
        ValueError: If required logging or output settings are absent.
    """
    global _ACTIVE_SESSION

    logging_settings = _required_mapping(
        configuration,
        "logging_settings",
        "configuration",
    )
    file_settings = _required_mapping(logging_settings, "file", "logging_settings")
    console_settings = _required_mapping(
        logging_settings,
        "console",
        "logging_settings",
    )
    io_settings = _required_mapping(configuration, "io_settings", "configuration")
    output_directory = _required_string(
        io_settings,
        "output_directory",
        "io_settings",
    )
    file_enabled = _required_bool(
        file_settings,
        "enabled",
        "logging_settings.file",
    )
    file_level = _logging_level(file_settings, "level", "logging_settings.file")
    log_directory = _relative_log_directory(
        _required_string(
            file_settings,
            "directory",
            "logging_settings.file",
        )
    )
    console_enabled = _required_bool(
        console_settings,
        "enabled",
        "logging_settings.console",
    )
    console_level = _logging_level(
        console_settings,
        "level",
        "logging_settings.console",
    )

    with _SESSION_LOCK:
        if _ACTIVE_SESSION is not None:
            _ACTIVE_SESSION.close()

        output_handlers: list[logging.Handler] = []
        logfile_path: Path | None = None
        try:
            if file_enabled:
                logfile_path = _create_logfile_path(
                    Path(output_directory),
                    log_directory,
                )
                file_handler = logging.FileHandler(
                    logfile_path,
                    mode="x",
                    encoding="utf-8",
                )
                file_handler.setLevel(file_level)
                file_handler.setFormatter(_file_formatter())
                output_handlers.append(file_handler)

            if console_enabled:
                console_handler = logging.StreamHandler(stream=sys.stderr)
                console_handler.setLevel(console_level)
                console_handler.setFormatter(
                    logging.Formatter("[%(levelname)s] %(message)s")
                )
                output_handlers.append(console_handler)

            log_queue = multiprocessing.get_context().Queue()
            queue_handler = logging.handlers.QueueHandler(log_queue)
            queue_handler.setLevel(logging.DEBUG)
            setattr(queue_handler, _OWNED_HANDLER_ATTRIBUTE, True)
            listener = logging.handlers.QueueListener(
                log_queue,
                *output_handlers,
                respect_handler_level=True,
            )

            package_logger = logging.getLogger(_LOGGER_NAME)
            previous_level = package_logger.level
            previous_propagate = package_logger.propagate
            package_logger.setLevel(logging.DEBUG)
            package_logger.propagate = False
            package_logger.addHandler(queue_handler)
            try:
                listener.start()
            except BaseException:
                package_logger.removeHandler(queue_handler)
                package_logger.setLevel(previous_level)
                package_logger.propagate = previous_propagate
                queue_handler.close()
                log_queue.close()
                log_queue.join_thread()
                raise

            session = LoggingSession(
                log_queue=log_queue,
                queue_handler=queue_handler,
                listener=listener,
                output_handlers=output_handlers,
                logfile_path=logfile_path,
                previous_level=previous_level,
                previous_propagate=previous_propagate,
            )
            _ACTIVE_SESSION = session
            return session
        except BaseException:
            for handler in output_handlers:
                handler.close()
            raise


def configure_worker_logging(log_queue: Any) -> None:
    """Send TNT records from one worker process to the parent log queue.

    Args:
        log_queue: Queue exposed as ``LoggingSession.worker_queue`` by the
            parent process.
    """
    package_logger = logging.getLogger(_LOGGER_NAME)
    for handler in list(package_logger.handlers):
        if getattr(handler, _OWNED_HANDLER_ATTRIBUTE, False):
            package_logger.removeHandler(handler)
            handler.close()

    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setLevel(logging.DEBUG)
    setattr(queue_handler, _OWNED_HANDLER_ATTRIBUTE, True)
    package_logger.addHandler(queue_handler)
    package_logger.setLevel(logging.DEBUG)
    package_logger.propagate = False


def _relative_log_directory(value: str) -> Path:
    """Return a safe output-relative logging directory."""
    relative_path = Path(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            "logging_settings.file.directory must stay within "
            "io_settings.output_directory."
        )
    return relative_path


def _create_logfile_path(output_directory: Path, relative_directory: Path) -> Path:
    """Create the log directory and return a unique timestamped path."""
    log_directory = output_directory / relative_directory
    log_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return log_directory / f"tnt-{timestamp}.log"


def _file_formatter() -> logging.Formatter:
    """Return the detailed UTC formatter used for TNT logfiles."""
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ [%(levelname)s] %(name)s "
        "[%(processName)s] %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    return formatter


def _required_mapping(
    mapping: Mapping[str, Any],
    key: str,
    parent: str,
) -> Mapping[str, Any]:
    """Return one required nested mapping."""
    if key not in mapping:
        raise ValueError(f"{parent} is missing required field: {key}.")
    value = mapping[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"{parent}.{key} must be a mapping.")
    return value


def _required_string(mapping: Mapping[str, Any], key: str, parent: str) -> str:
    """Return one required non-empty string."""
    if key not in mapping:
        raise ValueError(f"{parent} is missing required field: {key}.")
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{parent}.{key} must be a non-empty string.")
    return value


def _required_bool(mapping: Mapping[str, Any], key: str, parent: str) -> bool:
    """Return one required Boolean value."""
    if key not in mapping:
        raise ValueError(f"{parent} is missing required field: {key}.")
    value = mapping[key]
    if not isinstance(value, bool):
        raise TypeError(f"{parent}.{key} must be a boolean.")
    return value


def _logging_level(mapping: Mapping[str, Any], key: str, parent: str) -> int:
    """Return a validated standard logging level number."""
    level_name = _required_string(mapping, key, parent)
    level = logging.getLevelName(level_name)
    if not isinstance(level, int) or level_name not in {
        "CRITICAL",
        "DEBUG",
        "ERROR",
        "INFO",
        "WARNING",
    }:
        raise ValueError(f"{parent}.{key} has unsupported level {level_name!r}.")
    return level
