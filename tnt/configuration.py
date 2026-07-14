"""Read, resolve, and preserve TNT configuration files."""

from __future__ import annotations

import os
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

from tnt.configuration_validation import validate_resolved_configuration

CONFIG_REPOSITORY_DIRECTORY = "config_repository"
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"

ConfigDict = dict[str, Any]


class Configuration:
    """A resolved TNT configuration without instantiated runtime objects.

    Reading a user configuration merges it over the packaged defaults, applies
    defaults for dynamically named objects, and writes the resulting standalone
    configuration to the configured output directory. It does not construct the
    physical system or execute any modelling code.
    """

    def __init__(self) -> None:
        """Initialize an empty configuration container."""
        self.data: ConfigDict = {}
        self.source_path: Path | None = None
        self.resolved_path: Path | None = None

    def read(self, filename: str | Path) -> Configuration:
        """Read, resolve, and preserve a user configuration.

        Args:
            filename: YAML user-configuration path.

        Returns:
            This configuration instance after successful resolution.

        Raises:
            FileNotFoundError: If the user configuration does not exist.
            TypeError: If a required configuration mapping has the wrong type.
            ValueError: If required configuration values are absent or invalid.
        """
        source_path = Path(filename).expanduser()
        user_config = _read_yaml_mapping(source_path, "user configuration")
        default_config = _read_packaged_defaults()
        merged_config = _deep_merge(default_config, user_config)
        resolved_config = _apply_schema_defaults(merged_config)
        output_directory = _resolve_io_directories(resolved_config)
        validate_resolved_configuration(resolved_config)

        resolved_path = (
            output_directory / CONFIG_REPOSITORY_DIRECTORY / RESOLVED_CONFIG_FILENAME
        )
        _write_yaml_atomically(resolved_config, resolved_path)

        self.data = resolved_config
        self.source_path = source_path.resolve()
        self.resolved_path = resolved_path.resolve()
        return self

    def as_dict(self) -> ConfigDict:
        """Return an independent copy of the resolved configuration."""
        return deepcopy(self.data)

    def print(self) -> None:
        """Print the resolved configuration as YAML."""
        if not self.data:
            raise RuntimeError("No configuration has been read.")
        print(_dump_yaml(self.data), end="")


def _read_packaged_defaults() -> ConfigDict:
    """Load the packaged TNT default configuration."""
    default_resource = files("tnt.defaults").joinpath("default_config.yaml")
    with default_resource.open("r", encoding="utf-8") as stream:
        loaded = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    return _require_mapping(loaded, "packaged default configuration")


def _read_yaml_mapping(path: Path, description: str) -> ConfigDict:
    """Read a YAML document whose root must be a mapping."""
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    return _require_mapping(loaded, description)


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


def _require_mapping(value: Any, description: str) -> ConfigDict:
    """Return a mapping or raise a configuration-focused error."""
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a YAML mapping.")
    return value


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Recursively merge mappings, with override values taking precedence."""
    result = deepcopy(base)
    for key, override_value in override.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            result[key] = _deep_merge(base_value, override_value)
        else:
            result[key] = deepcopy(override_value)
    return result


def _apply_schema_defaults(config: ConfigDict) -> ConfigDict:
    """Apply defaults to dynamically named objects and remove schema metadata."""
    resolved = deepcopy(config)
    dynamic_defaults = _require_mapping(
        resolved.pop("dynamic_object_defaults", {}),
        "dynamic_object_defaults",
    )
    kinematics_type_defaults = _require_mapping(
        resolved.pop("kinematics_type_defaults", {}),
        "kinematics_type_defaults",
    )

    component_defaults = _mapping_value(
        dynamic_defaults, "component", "dynamic_object_defaults"
    )
    parameter_defaults = _mapping_value(
        dynamic_defaults, "parameter", "dynamic_object_defaults"
    )
    kinematics_defaults = _mapping_value(
        dynamic_defaults, "kinematics", "dynamic_object_defaults"
    )

    components = _mapping_value(resolved, "system_components", "configuration")
    resolved["system_components"] = {
        name: _resolve_component(
            name,
            component,
            component_defaults,
            parameter_defaults,
            kinematics_defaults,
            kinematics_type_defaults,
        )
        for name, component in components.items()
    }

    system_parameters = _mapping_value(resolved, "system_parameters", "configuration")
    resolved["system_parameters"] = _resolve_parameters(
        system_parameters,
        parameter_defaults,
        "system_parameters",
    )
    return resolved


def _mapping_value(mapping: ConfigDict, key: str, parent: str) -> ConfigDict:
    """Get and type-check a nested mapping, defaulting an absent key to empty."""
    return _require_mapping(mapping.get(key, {}), f"{parent}.{key}")


def _resolve_component(
    name: str,
    component: Any,
    component_defaults: ConfigDict,
    parameter_defaults: ConfigDict,
    kinematics_defaults: ConfigDict,
    kinematics_type_defaults: ConfigDict,
) -> ConfigDict:
    """Resolve defaults for one dynamically named system component."""
    component_mapping = _require_mapping(
        component,
        f"system_components.{name}",
    )
    resolved = _deep_merge(component_defaults, component_mapping)

    parameters = _mapping_value(
        resolved,
        "parameters",
        f"system_components.{name}",
    )
    if parameters or "parameters" in resolved:
        resolved["parameters"] = _resolve_parameters(
            parameters,
            parameter_defaults,
            f"system_components.{name}.parameters",
        )

    kinematics = _mapping_value(
        resolved,
        "kinematics",
        f"system_components.{name}",
    )
    if kinematics or "kinematics" in resolved:
        resolved["kinematics"] = {
            kinematics_name: _resolve_kinematics(
                name,
                kinematics_name,
                settings,
                kinematics_defaults,
                kinematics_type_defaults,
            )
            for kinematics_name, settings in kinematics.items()
        }
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
            _require_mapping(settings, f"{path}.{name}"),
        )
        for name, settings in parameters.items()
    }


def _resolve_kinematics(
    component_name: str,
    name: str,
    settings: Any,
    kinematics_defaults: ConfigDict,
    kinematics_type_defaults: ConfigDict,
) -> ConfigDict:
    """Resolve common and type-specific defaults for one kinematics data set."""
    path = f"system_components.{component_name}.kinematics.{name}"
    settings_mapping = _require_mapping(settings, path)
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

    type_defaults = _require_mapping(
        kinematics_type_defaults[kinematics_type],
        f"kinematics_type_defaults.{kinematics_type}",
    )
    if _has_complete_histogram_metadata(settings_mapping):
        type_defaults = deepcopy(type_defaults)
        type_defaults.pop("histogram", None)

    resolved = _deep_merge(kinematics_defaults, type_defaults)
    return _deep_merge(resolved, settings_mapping)


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


def _resolve_io_directories(config: ConfigDict) -> Path:
    """Validate I/O paths and store their absolute forms in the snapshot."""
    io_settings = _mapping_value(config, "io_settings", "configuration")
    for key in ("input_directory", "output_directory"):
        value = io_settings.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"io_settings.{key} must be a non-empty string.")
        io_settings[key] = str(Path(value).expanduser().resolve())
    return Path(io_settings["output_directory"])


def _dump_yaml(config: ConfigDict) -> str:
    """Serialize a configuration deterministically as block-style YAML."""
    return yaml.safe_dump(
        config,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )


def _write_yaml_atomically(config: ConfigDict, destination: Path) -> None:
    """Write a resolved YAML configuration using an atomic replacement."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write("# Generated by TNT. Do not edit this resolved file.\n")
            stream.write(_dump_yaml(config))
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
