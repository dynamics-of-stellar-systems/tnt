import logging
from pathlib import Path
from typing import Any

import pytest

from tnt.logging import configure_logging, configure_worker_logging


def _logging_configuration(output_directory: Path) -> dict[str, Any]:
    return {
        "io_settings": {"output_directory": str(output_directory)},
        "logging_settings": {
            "file": {
                "enabled": True,
                "level": "DEBUG",
                "directory": "logs",
            },
            "console": {"enabled": True, "level": "INFO"},
        },
    }


def _emit_worker_message(log_queue: Any) -> None:
    configure_worker_logging(log_queue)
    logging.getLogger("tnt.worker").info("message from worker")


def test_logging_session_writes_detailed_file_and_filtered_console(
    tmp_path: Path,
    capsys: Any,
) -> None:
    root_logger = logging.getLogger()
    root_handlers = list(root_logger.handlers)
    root_level = root_logger.level
    package_logger = logging.getLogger("tnt")
    package_level = package_logger.level
    package_propagate = package_logger.propagate

    with configure_logging(_logging_configuration(tmp_path)) as session:
        logger = logging.getLogger("tnt.test")
        logger.debug("debug detail")
        logger.info("terminal information")
        logger.error("terminal error")

        assert list(root_logger.handlers) == root_handlers
        assert root_logger.level == root_level
        assert session.logfile_path is not None
        logfile_path = session.logfile_path
        assert logfile_path.parent == tmp_path / "logs"
        assert logfile_path.name.startswith("tnt-")
        assert logfile_path.suffix == ".log"

    captured = capsys.readouterr()
    logfile = logfile_path.read_text(encoding="utf-8")

    assert "debug detail" in logfile
    assert "terminal information" in logfile
    assert "terminal error" in logfile
    assert "[MainProcess]" in logfile
    assert "tnt.test" in logfile
    assert "debug detail" not in captured.err
    assert "[INFO] terminal information" in captured.err
    assert "[ERROR] terminal error" in captured.err
    assert package_logger.level == package_level
    assert package_logger.propagate is package_propagate
    assert list(root_logger.handlers) == root_handlers
    assert root_logger.level == root_level


def test_reconfiguration_closes_the_previous_tnt_session(tmp_path: Path) -> None:
    first = configure_logging(_logging_configuration(tmp_path / "first"))
    second = configure_logging(_logging_configuration(tmp_path / "second"))

    try:
        assert first.closed
        assert not second.closed
    finally:
        second.close()

    assert second.closed


def test_invalid_reconfiguration_keeps_existing_session_active(
    tmp_path: Path,
) -> None:
    session = configure_logging(_logging_configuration(tmp_path / "active"))
    invalid_configuration = _logging_configuration(tmp_path / "invalid")
    invalid_configuration["logging_settings"]["console"]["level"] = "VERBOSE"

    try:
        with pytest.raises(ValueError, match="unsupported level"):
            configure_logging(invalid_configuration)
        assert not session.closed
    finally:
        session.close()


def test_logging_destinations_can_be_disabled(
    tmp_path: Path,
    capsys: Any,
) -> None:
    configuration = _logging_configuration(tmp_path)
    configuration["logging_settings"]["file"]["enabled"] = False
    configuration["logging_settings"]["console"]["enabled"] = False

    with configure_logging(configuration) as session:
        logging.getLogger("tnt.test").error("discarded by explicit settings")
        assert session.logfile_path is None

    assert not (tmp_path / "logs").exists()
    assert "discarded by explicit settings" not in capsys.readouterr().err


def test_worker_records_are_written_by_parent_listener(tmp_path: Path) -> None:
    with configure_logging(_logging_configuration(tmp_path)) as session:
        assert session.worker_context.get_start_method() == "spawn"
        process = session.worker_context.Process(
            target=_emit_worker_message,
            args=(session.worker_queue,),
        )
        process.start()
        try:
            process.join(timeout=30)
            assert not process.is_alive()
            assert process.exitcode == 0
            assert session.logfile_path is not None
            logfile_path = session.logfile_path
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()

    logfile = logfile_path.read_text(encoding="utf-8")
    assert "message from worker" in logfile
    assert "tnt.worker" in logfile
