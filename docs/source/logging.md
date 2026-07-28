# Logging

TNT emits records through loggers below the `tnt` namespace. Importing TNT and
reading a configuration do not install file or terminal handlers or change the
root logger. An application embedding TNT can therefore use its own logging
configuration without interference.

## Configuration and execution session

The recommended standalone entry point bootstraps logging before complete
configuration preparation. Configuration loading, resolution, validation, and
the subsequent execution body are therefore recorded in the same logfile:

```python
import logging
import tnt

with tnt.configuration_session(
    "user_config.yaml",
    workspace_root="/scratch/project/NGC6278",
) as config:
    logger = logging.getLogger("tnt.application")
    logger.info("Starting TNT execution")
    # Construct and execute the model here.
```

The bootstrap parses the user YAML and packaged defaults once, resolves the
output directory against the same workspace root used by full preparation,
validates the logging settings, starts logging, and then continues complete
configuration resolution using the already-loaded mapping. It does not parse
the configuration twice.

Malformed YAML, an invalid or missing output directory, and invalid bootstrap
logging settings cannot be recorded in the intended logfile because a valid
logging destination is not yet available. These errors are raised to the
caller and can be handled by logging the application configured before calling
TNT.

Once bootstrap logging is active, configuration or execution exceptions are
written once with their traceback and then re-raised to the caller.

## Logging an already-resolved configuration

Applications that already have a resolved configuration can activate the same
operational logging explicitly:

```python
config = tnt.Configuration().read("user_config.yaml")

with tnt.configure_logging(config.as_dict()) as logging_session:
    # Execute TNT here.
    ...
```

This lower-level form does not include the earlier configuration preparation
in the TNT logfile.

The default profile creates a uniquely timestamped logfile under
`<output_directory>/logs/`. The logfile receives `DEBUG` and higher records
with timestamps, logger and process names, and source locations. The terminal
receives `INFO` and higher records with a shorter format.

The returned `LoggingSession` is a context manager. Leaving its context stops
the listener, closes TNT's handlers, and restores the previous state of the
`tnt` package logger. Calling `configure_logging()` again closes the prior TNT
session first, preventing duplicate TNT-owned handlers.

Neither `configure_logging()` nor session cleanup changes, shuts down, or
reloads the root logger. Applications that want their own logging behavior
should simply omit this explicit setup and configure Python logging normally.

## Worker processes

TNT logging uses a central queue so worker processes do not write directly to
the same file. Pass the active session's queue to each worker and configure the
worker at process startup:

```python
import multiprocessing


def worker(log_queue):
    import logging
    import tnt

    tnt.configure_worker_logging(log_queue)
    logging.getLogger("tnt.worker").info("Worker started")


with tnt.configure_logging(config.as_dict()) as logging_session:
    process = multiprocessing.Process(
        target=worker,
        args=(logging_session.worker_queue,),
    )
    process.start()
    process.join()
```

Workers submit records to the queue; only the parent listener formats and
writes them.

## Configuration

The operational defaults are:

```yaml
logging_settings:
  file:
    enabled: true
    level: "DEBUG"
    directory: "logs"
  console:
    enabled: true
    level: "INFO"
```

Supported levels are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`. The
log directory must be relative and remain within `io_settings.output_directory`.
Either destination can be disabled explicitly.
