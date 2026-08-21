"""Read, resolve, and preserve TNT configuration files."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Any

import yaml

from tnt.config_parsing import _mapping, _optional_mapping
from tnt.configuration_validation import validate_resolved_configuration
from tnt.logging import configure_logging
from tnt.units import (
    UnitSystems,
    build_unit_systems,
    normalize_configuration_quantities,
)

CONFIG_REPOSITORY_DIRECTORY = "config_repository"
RUNS_DIRECTORY = "runs"
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
RUN_MANIFEST_FILENAME = "run_manifest.yaml"

ConfigDict = dict[str, Any]
_LOGGER = logging.getLogger(__name__)


class Configuration:
    """A resolved TNT configuration without instantiated runtime objects.

    Reading a user configuration merges it over the packaged defaults, applies
    defaults for dynamically named objects, materializes runtime paths, and
    writes portable configuration-repository artifacts to the output directory.
    It does not construct the physical system or execute modelling code.
    """

    def __init__(self) -> None:
        """Initialize an empty configuration container."""
        self.data: ConfigDict = {}
        self.portable_data: ConfigDict = {}
        self.source_path: Path | None = None
        self.workspace_root: Path | None = None
        self.run_id: int | None = None
        self.resolved_path: Path | None = None
        self.run_manifest_path: Path | None = None
        self.unit_systems: UnitSystems | None = None

    def read(
        self,
        filename: str | Path,
        workspace_root: str | Path | None = None,
    ) -> Configuration:
        """Read, resolve, and preserve a user configuration.

        Args:
            filename: YAML user-configuration path.
            workspace_root: Base directory for relative input and output paths.
                Defaults to the directory containing the invoking Python script,
                or the current working directory in an interactive session.

        Returns:
            This configuration instance after successful resolution.

        Raises:
            FileNotFoundError: If the user configuration does not exist.
            TypeError: If a required configuration mapping has the wrong type.
            ValueError: If required configuration values are absent or invalid.
        """
        root = _resolve_workspace_root(workspace_root)
        source_path, merged_config = _load_merged_configuration(filename)
        return self._resolve_and_write(
            source_path,
            merged_config,
            root,
        )

    def _resolve_and_write(
        self,
        source_path: Path,
        merged_config: ConfigDict,
        workspace_root: Path,
        logfile_path: Path | None = None,
    ) -> Configuration:
        """Resolve, validate, and preserve an already-loaded configuration."""
        _LOGGER.debug("Resolving configuration loaded from %s.", source_path)
        schema_resolved_config = _apply_schema_defaults(merged_config)
        unit_systems = build_unit_systems(
            _optional_mapping(schema_resolved_config, "units", "configuration")
        )
        schema_resolved_config = normalize_configuration_quantities(
            schema_resolved_config,
            unit_systems,
        )
        runtime_config, portable_config, output_directory = _resolve_io_directories(
            schema_resolved_config,
            workspace_root,
        )
        validate_resolved_configuration(portable_config)

        repository = output_directory / CONFIG_REPOSITORY_DIRECTORY
        run_manifest_path, resolved_path, run_id = _preserve_run(
            repository=repository,
            workspace_root=workspace_root,
            runtime_config=runtime_config,
            portable_config=portable_config,
            logfile_path=logfile_path,
        )

        _LOGGER.info("Resolved configuration preserved at %s.", resolved_path)
        _LOGGER.info("Run manifest written to %s.", run_manifest_path)

        self.data = runtime_config
        self.portable_data = portable_config
        self.source_path = source_path.resolve()
        self.workspace_root = workspace_root
        self.run_id = run_id
        self.resolved_path = resolved_path.resolve()
        self.run_manifest_path = run_manifest_path.resolve()
        self.unit_systems = unit_systems
        return self

    def as_dict(self) -> ConfigDict:
        """Return an independent copy with materialized runtime paths."""
        return deepcopy(self.data)

    def as_portable_dict(self) -> ConfigDict:
        """Return an independent copy with workspace-relative I/O paths."""
        return deepcopy(self.portable_data)

    def print(self) -> None:
        """Print the resolved runtime configuration as YAML."""
        if not self.data:
            raise RuntimeError("No configuration has been read.")
        print(_dump_yaml(self.data), end="")


@contextmanager
def configuration_session(
    filename: str | Path,
    workspace_root: str | Path | None = None,
) -> Iterator[Configuration]:
    """Prepare a configuration inside an isolated TNT logging session.

    The user YAML and packaged defaults are loaded once. A minimal bootstrap
    extracts the output directory and logging settings, starts TNT-local
    logging, and then performs complete resolution and validation.

    Args:
        filename: YAML user-configuration path.
        workspace_root: Base directory for relative input and output paths.
            Defaults to the directory containing the invoking Python script,
            or the current working directory in an interactive session.

    Yields:
        The resolved configuration while its TNT logging session remains active.

    Raises:
        FileNotFoundError: If the user configuration does not exist.
        TypeError: If bootstrap or full configuration data has the wrong type.
        ValueError: If bootstrap or full configuration data is invalid.
    """
    root = _resolve_workspace_root(workspace_root)
    source_path, merged_config = _load_merged_configuration(filename)
    bootstrap_config = _logging_bootstrap_configuration(merged_config, root)

    with configure_logging(bootstrap_config) as logging_session:
        _LOGGER.info("User configuration loaded from %s.", source_path)
        if logging_session.logfile_path is not None:
            _LOGGER.info("Detailed TNT logfile: %s.", logging_session.logfile_path)

        config = Configuration()
        try:
            config._resolve_and_write(
                source_path,
                merged_config,
                root,
                logging_session.logfile_path,
            )
        except Exception:
            _LOGGER.exception("Configuration preparation failed for %s.", source_path)
            raise

        try:
            yield config
        except Exception:
            _LOGGER.exception("TNT configuration session failed.")
            raise
        else:
            _LOGGER.info("TNT configuration session completed.")


def _load_merged_configuration(
    filename: str | Path,
) -> tuple[Path, ConfigDict]:
    """Load one user profile and merge it with packaged defaults."""
    source_path = Path(filename).expanduser()
    _LOGGER.debug("Reading user configuration from %s.", source_path)
    user_config_bytes = source_path.read_bytes()
    user_config = _read_yaml_bytes_mapping(
        user_config_bytes,
        "user configuration",
    )
    default_config = _read_packaged_defaults()
    return source_path, _deep_merge(default_config, user_config)


def _logging_bootstrap_configuration(
    config: ConfigDict,
    workspace_root: Path,
) -> ConfigDict:
    """Extract validated-enough settings needed to start TNT logging."""
    io_settings = _optional_mapping(config, "io_settings", "configuration")
    output_directory = io_settings.get("output_directory")
    if not isinstance(output_directory, str) or not output_directory.strip():
        raise ValueError("io_settings.output_directory must be a non-empty string.")

    logging_settings = _optional_mapping(config, "logging_settings", "configuration")
    runtime_output = _materialize_path(output_directory, workspace_root)
    return {
        "io_settings": {
            "output_directory": str(runtime_output),
        },
        "logging_settings": deepcopy(logging_settings),
    }


def _read_packaged_defaults() -> ConfigDict:
    """Load the packaged TNT default configuration."""
    default_resource = files("tnt.defaults").joinpath("default_config.yaml")
    with default_resource.open("r", encoding="utf-8") as stream:
        loaded = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    return _mapping(loaded, "packaged default configuration")


def _read_yaml_bytes_mapping(content: bytes, description: str) -> ConfigDict:
    """Read one UTF-8 YAML mapping from its original byte representation."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{description} must be UTF-8 encoded.") from error
    loaded = yaml.load(text, Loader=_UniqueKeySafeLoader)
    return _mapping(loaded, description)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Load safe YAML while rejecting duplicate mapping keys."""

    def construct_mapping(
        self,
        node: yaml.MappingNode,
        deep: bool = False,
    ) -> ConfigDict:
        """Construct a mapping after checking each key for uniqueness."""
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                line = key_node.start_mark.line + 1
                raise ValueError(f"Duplicate configuration key {key!r} at line {line}.")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Recursively merge mappings, with override values taking precedence."""
    result = deepcopy(base)
    for key, override_value in override.items():
        base_value = result.get(key)
        if (
            isinstance(base_value, dict)
            and isinstance(override_value, dict)
            and not _is_explicit_quantity_mapping(base_value)
        ):
            result[key] = _deep_merge(base_value, override_value)
        else:
            result[key] = deepcopy(override_value)
    return result


def _is_explicit_quantity_mapping(value: ConfigDict) -> bool:
    """Return whether a mapping is one atomic value-and-unit quantity."""
    return set(value) == {"unit", "value"}


def _apply_schema_defaults(config: ConfigDict) -> ConfigDict:
    """Apply defaults to dynamically named objects and remove schema metadata."""
    resolved = deepcopy(config)
    dynamic_defaults = _mapping(
        resolved.pop("dynamic_object_defaults", {}),
        "dynamic_object_defaults",
    )
    kinematics_type_defaults = _mapping(
        resolved.pop("kinematics_type_defaults", {}),
        "kinematics_type_defaults",
    )

    potential_defaults = _optional_mapping(
        dynamic_defaults, "potential", "dynamic_object_defaults"
    )
    parameter_defaults = _optional_mapping(
        dynamic_defaults, "parameter", "dynamic_object_defaults"
    )

    potential = _optional_mapping(resolved, "potential", "configuration")
    resolved["potential"] = {
        name: _resolve_potential(
            name,
            settings,
            potential_defaults,
            parameter_defaults,
        )
        for name, settings in potential.items()
    }

    kinematic_data = _optional_mapping(resolved, "kinematic_data", "configuration")
    resolved["kinematic_data"] = {
        name: _resolve_kinematics(
            name,
            settings,
            kinematics_type_defaults,
        )
        for name, settings in kinematic_data.items()
    }
    return resolved


def _resolve_potential(
    name: str,
    settings: Any,
    potential_defaults: ConfigDict,
    parameter_defaults: ConfigDict,
) -> ConfigDict:
    """Resolve defaults for one dynamically named potential component."""
    path = f"potential.{name}"
    settings_mapping = _mapping(
        settings,
        path,
    )
    resolved = _deep_merge(potential_defaults, settings_mapping)

    parameters = _optional_mapping(
        resolved,
        "parameters",
        path,
    )
    if parameters or "parameters" in resolved:
        resolved["parameters"] = _resolve_parameters(
            parameters,
            parameter_defaults,
            f"{path}.parameters",
        )
    return resolved


def _resolve_parameters(
    parameters: ConfigDict,
    parameter_defaults: ConfigDict,
    path: str,
) -> ConfigDict:
    """Resolve defaults for a mapping of dynamically named parameters."""
    return {
        name: _deep_merge(
            parameter_defaults,
            _mapping(settings, f"{path}.{name}"),
        )
        for name, settings in parameters.items()
    }


def _resolve_kinematics(
    name: str,
    settings: Any,
    kinematics_type_defaults: ConfigDict,
) -> ConfigDict:
    """Resolve common and type-specific defaults for one kinematics data set."""
    path = f"kinematic_data.{name}"
    settings_mapping = _mapping(settings, path)
    _validate_explicit_histogram_completeness(settings_mapping, path)
    kinematics_type = settings_mapping.get("type")
    if not isinstance(kinematics_type, str) or not kinematics_type:
        raise ValueError(f"{path}.type must be a non-empty string.")
    if kinematics_type not in kinematics_type_defaults:
        allowed_types = ", ".join(sorted(kinematics_type_defaults))
        raise ValueError(
            f"Unsupported {path}.type {kinematics_type!r}; "
            f"expected one of: {allowed_types}."
        )

    type_defaults = _mapping(
        kinematics_type_defaults[kinematics_type],
        f"kinematics_type_defaults.{kinematics_type}",
    )
    type_defaults = _replace_explicit_systematic_uncertainties(
        type_defaults,
        settings_mapping,
    )
    if _has_complete_histogram_metadata(settings_mapping):
        type_defaults = deepcopy(type_defaults)
        type_defaults.pop("histogram", None)

    return _deep_merge(type_defaults, settings_mapping)


def _replace_explicit_systematic_uncertainties(
    type_defaults: ConfigDict,
    settings: ConfigDict,
) -> ConfigDict:
    """Replace rather than merge an explicit systematic-uncertainty mapping."""
    observational_errors = settings.get("observational_errors")
    if not isinstance(observational_errors, dict):
        return type_defaults
    if "systematic_uncertainties" not in observational_errors:
        return type_defaults

    adjusted_defaults = deepcopy(type_defaults)
    default_errors = adjusted_defaults.get("observational_errors")
    if isinstance(default_errors, dict):
        default_errors.pop("systematic_uncertainties", None)
    return adjusted_defaults


def _has_complete_histogram_metadata(settings: ConfigDict) -> bool:
    """Return whether all explicit histogram metadata values are present."""
    histogram = settings.get("histogram")
    return isinstance(histogram, dict) and all(
        key in histogram for key in ("width", "center", "bins")
    )


def _validate_explicit_histogram_completeness(
    settings: ConfigDict,
    path: str,
) -> None:
    """Reject partial explicit histogram metadata before defaults are merged."""
    histogram = settings.get("histogram")
    if not isinstance(histogram, dict):
        return
    explicit_fields = {"width", "center", "bins"}
    provided_fields = explicit_fields.intersection(histogram)
    if provided_fields and provided_fields != explicit_fields:
        missing = ", ".join(sorted(explicit_fields - provided_fields))
        raise ValueError(
            f"{path}.histogram must define width, center, and bins together; "
            f"missing: {missing}."
        )


def _resolve_workspace_root(workspace_root: str | Path | None) -> Path:
    """Return the explicit root or the invoking script's directory."""
    if workspace_root is None:
        main_module = sys.modules.get("__main__")
        main_file = getattr(main_module, "__file__", None)
        if isinstance(main_file, str) and not main_file.startswith("<"):
            return Path(main_file).expanduser().resolve().parent
        return Path.cwd().resolve()

    root = Path(workspace_root).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def _resolve_io_directories(
    config: ConfigDict,
    workspace_root: Path,
) -> tuple[ConfigDict, ConfigDict, Path]:
    """Create runtime and portable forms of input and output directories."""
    runtime_config = deepcopy(config)
    portable_config = deepcopy(config)
    runtime_io = _optional_mapping(runtime_config, "io_settings", "configuration")
    portable_io = _optional_mapping(portable_config, "io_settings", "configuration")
    for key in ("input_directory", "output_directory"):
        value = portable_io.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"io_settings.{key} must be a non-empty string.")
        materialized = _materialize_path(value, workspace_root)
        runtime_io[key] = str(materialized)
        portable_io[key] = _workspace_relative_path(materialized, workspace_root)
    return runtime_config, portable_config, Path(runtime_io["output_directory"])


def _materialize_path(value: str, workspace_root: Path) -> Path:
    """Resolve one configured path against the workspace root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()


def _workspace_relative_path(path: Path, workspace_root: Path) -> str:
    """Represent an absolute path relative to the workspace root."""
    try:
        relative = os.path.relpath(path, workspace_root)
    except ValueError as error:
        raise ValueError(
            f"Configured path {path} cannot be represented relative to "
            f"workspace root {workspace_root}."
        ) from error
    return Path(relative).as_posix()


def _preserve_run(
    *,
    repository: Path,
    workspace_root: Path,
    runtime_config: ConfigDict,
    portable_config: ConfigDict,
    logfile_path: Path | None,
) -> tuple[Path, Path, int]:
    """Atomically publish one immutable manifest/configuration run bundle."""
    runs_directory = repository / RUNS_DIRECTORY
    runs_directory.mkdir(parents=True, exist_ok=True)
    run_id = _next_index(runs_directory)
    run_directory = runs_directory / f"{run_id:04d}"
    resolved_path = run_directory / RESOLVED_CONFIG_FILENAME
    manifest_path = run_directory / RUN_MANIFEST_FILENAME
    temporary_directory = Path(mkdtemp(dir=runs_directory, prefix=".run-"))
    try:
        temporary_resolved_path = temporary_directory / RESOLVED_CONFIG_FILENAME
        temporary_manifest_path = temporary_directory / RUN_MANIFEST_FILENAME
        _write_yaml_immutably(
            portable_config,
            temporary_resolved_path,
            "resolved configuration",
        )
        manifest = _build_run_manifest(
            workspace_root=workspace_root,
            runtime_config=runtime_config,
            repository=repository,
            resolved_path=resolved_path,
            logfile_path=logfile_path,
            run_id=run_id,
        )
        _write_yaml_immutably(manifest, temporary_manifest_path, "run manifest")
        os.replace(temporary_directory, run_directory)
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)
    return manifest_path, resolved_path, run_id


def _next_index(directory: Path) -> int:
    """Return the first index after every indexed entry in `directory`."""
    indices = [
        int(path.name)
        for path in directory.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    return max(indices, default=-1) + 1


def _repository_relative(path: Path, repository: Path) -> str:
    """Represent an artifact path relative to the configuration repository."""
    return path.relative_to(repository).as_posix()


def _build_run_manifest(
    *,
    workspace_root: Path,
    runtime_config: ConfigDict,
    repository: Path,
    resolved_path: Path,
    logfile_path: Path | None,
    run_id: int,
) -> ConfigDict:
    """Build provenance for this concrete TNT run."""
    io_settings = _optional_mapping(runtime_config, "io_settings", "configuration")
    orbit_settings = _optional_mapping(
        runtime_config,
        "orbit_library_settings",
        "configuration",
    )
    configured_seed = orbit_settings.get("random_seed")
    effective_seed = (
        configured_seed
        if isinstance(configured_seed, int) and configured_seed >= 0
        else None
    )
    return {
        "manifest_version": 3,
        "run_id": run_id,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "tnt": {
            "version": _package_version("tnt"),
            **_git_provenance(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "dependencies": {
            "jax": _package_version("jax"),
            "pyyaml": _package_version("PyYAML"),
            "unxt": _package_version("unxt"),
        },
        "execution": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "working_directory": str(Path.cwd().resolve()),
            "workspace_root": str(workspace_root),
            "entrypoint": _entrypoint(),
            "scheduler": {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "pbs_job_id": os.environ.get("PBS_JOBID"),
            },
        },
        "configuration": {
            "resolved": _repository_relative(resolved_path, repository),
            "logfile": str(logfile_path.resolve()) if logfile_path else None,
            "input_directory": io_settings["input_directory"],
            "output_directory": io_settings["output_directory"],
        },
        "randomness": {
            "configured_orbit_library_seed": configured_seed,
            "effective_orbit_library_seed": effective_seed,
            "status": "fixed" if effective_seed is not None else "pending_generation",
        },
    }


def _entrypoint() -> str | None:
    """Return the invoking Python file without recording arbitrary arguments."""
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if not isinstance(main_file, str) or main_file.startswith("<"):
        return None
    return str(Path(main_file).expanduser().resolve())


def _package_version(package: str) -> str | None:
    """Return installed package metadata when available."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_provenance() -> ConfigDict:
    """Return commit and working-tree state when TNT is running from Git."""
    source_root = Path(__file__).resolve().parents[1]
    if not (source_root / ".git").exists():
        return {"git_commit": None, "git_working_tree_dirty": None}
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=source_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"git_commit": None, "git_working_tree_dirty": None}
    commit = commit_result.stdout.strip()
    return {
        "git_commit": (commit if commit_result.returncode == 0 and commit else None),
        "git_working_tree_dirty": (
            bool(status_result.stdout.strip())
            if status_result.returncode == 0
            else None
        ),
    }


def _dump_yaml(config: ConfigDict) -> str:
    """Serialize a configuration deterministically as block-style YAML."""
    return yaml.safe_dump(
        config,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )


def _generated_yaml_bytes(config: ConfigDict, description: str) -> bytes:
    """Serialize generated YAML with a descriptive header."""
    content = f"# Generated by TNT: {description}.\n{_dump_yaml(config)}"
    return content.encode("utf-8")


def _write_yaml_immutably(
    config: ConfigDict,
    destination: Path,
    description: str,
) -> None:
    """Write generated YAML atomically without replacing an existing file."""
    _write_bytes_immutably(
        _generated_yaml_bytes(config, description),
        destination,
    )


def _write_bytes_immutably(content: bytes, destination: Path) -> None:
    """Publish complete bytes atomically without replacing `destination`."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
