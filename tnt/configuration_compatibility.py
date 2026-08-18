"""Versioned compatibility policy for appending to one `AllModels` search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Self

import numpy as np
import yaml

from tnt.all_models import AllModels
from tnt.configuration import (
    ConfigDict,
    _read_yaml_bytes_mapping,
    _write_yaml_immutably,
)
from tnt.run_config_log import ConfigurationSnapshotReference

COMPATIBILITY_CONTRACT_VERSION = 1
COMPATIBILITY_SIGNATURE_FILENAME = "compatibility_signature.yaml"
_SEARCH_PARAMETER_KEYS = {
    "fixed",
    "generator_settings",
    "latex_label",
    "logarithmic",
    "value",
}


class ConfigurationCompatibilityError(ValueError):
    """A resumed search would mix scientifically incompatible models."""


@dataclass(frozen=True)
class ConfigurationCompatibilitySignature:
    """Canonical scientific identity used by the first compatibility contract."""

    contract_version: int
    critical_configuration: ConfigDict
    scientific_inputs: ConfigDict
    signature_sha256: str

    @classmethod
    def create(
        cls,
        critical_configuration: Mapping[str, Any],
        scientific_inputs: Mapping[str, Any],
    ) -> Self:
        """Create a signature from normalized compatibility inputs."""
        payload = {
            "contract_version": COMPATIBILITY_CONTRACT_VERSION,
            "critical_configuration": deepcopy(dict(critical_configuration)),
            "scientific_inputs": deepcopy(dict(scientific_inputs)),
        }
        return cls(
            contract_version=COMPATIBILITY_CONTRACT_VERSION,
            critical_configuration=payload["critical_configuration"],
            scientific_inputs=payload["scientific_inputs"],
            signature_sha256=_canonical_sha256(payload),
        )

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read and validate one persisted compatibility signature."""
        data = _read_yaml_bytes_mapping(
            path.read_bytes(),
            f"configuration compatibility signature {path}",
        )
        expected_keys = {
            "contract_version",
            "critical_configuration",
            "scientific_inputs",
            "signature_sha256",
        }
        if set(data) != expected_keys:
            raise ConfigurationCompatibilityError(
                f"Compatibility signature {path} must contain exactly "
                f"{sorted(expected_keys)!r}."
            )
        version = data["contract_version"]
        if version != COMPATIBILITY_CONTRACT_VERSION:
            raise ConfigurationCompatibilityError(
                f"Compatibility signature {path} uses contract version "
                f"{version!r}; TNT supports {COMPATIBILITY_CONTRACT_VERSION}."
            )
        critical = _require_mapping(data["critical_configuration"], path)
        inputs = _require_mapping(data["scientific_inputs"], path)
        expected_sha256 = _canonical_sha256(
            {
                "contract_version": version,
                "critical_configuration": critical,
                "scientific_inputs": inputs,
            }
        )
        if data["signature_sha256"] != expected_sha256:
            raise ConfigurationCompatibilityError(
                f"Compatibility signature {path} has an invalid SHA-256 digest."
            )
        return cls(
            contract_version=version,
            critical_configuration=critical,
            scientific_inputs=inputs,
            signature_sha256=expected_sha256,
        )

    def as_dict(self) -> ConfigDict:
        """Return the complete serializable signature document."""
        return {
            "contract_version": self.contract_version,
            "critical_configuration": deepcopy(self.critical_configuration),
            "scientific_inputs": deepcopy(self.scientific_inputs),
            "signature_sha256": self.signature_sha256,
        }

    def differences(self, other: Self) -> list[str]:
        """Return field paths whose compatibility-critical values differ."""
        if self.contract_version != other.contract_version:
            return ["contract_version"]
        return _different_paths(
            {
                "critical_configuration": self.critical_configuration,
                "scientific_inputs": self.scientific_inputs,
            },
            {
                "critical_configuration": other.critical_configuration,
                "scientific_inputs": other.scientific_inputs,
            },
        )


def build_and_preserve_compatibility_signature(
    config: Mapping[str, Any],
    snapshot: ConfigurationSnapshotReference,
) -> ConfigurationCompatibilitySignature:
    """Build the current signature and immutably preserve it with its snapshot."""
    signature = ConfigurationCompatibilitySignature.create(
        _critical_configuration(config),
        _scientific_input_hashes(config),
    )
    path = compatibility_signature_path(snapshot)
    if path.exists():
        persisted = ConfigurationCompatibilitySignature.read(path)
        differences = persisted.differences(signature)
        if differences:
            raise ConfigurationCompatibilityError(
                "The immutable resolved configuration snapshot now resolves to "
                "different scientific inputs: " + ", ".join(differences)
            )
        return persisted
    try:
        _write_yaml_immutably(
            signature.as_dict(),
            path,
            "configuration compatibility signature",
        )
    except FileExistsError:
        persisted = ConfigurationCompatibilitySignature.read(path)
        differences = persisted.differences(signature)
        if differences:
            raise ConfigurationCompatibilityError(
                "A concurrently created compatibility signature differs at: "
                + ", ".join(differences)
            )
        return persisted
    return signature


def ensure_resume_compatible(
    current: ConfigurationCompatibilitySignature,
    current_snapshot: ConfigurationSnapshotReference,
    historical_snapshots: Sequence[ConfigurationSnapshotReference],
    all_models: AllModels,
    which_chi2: str,
) -> None:
    """Reject incompatible historical snapshots or an unusable chi2 selection."""
    checked: set[str] = set()
    for snapshot in historical_snapshots:
        if snapshot.semantic_sha256 in checked:
            continue
        checked.add(snapshot.semantic_sha256)
        if snapshot.semantic_sha256 == current_snapshot.semantic_sha256:
            historical = current
        else:
            path = compatibility_signature_path(snapshot)
            if not path.is_file():
                raise ConfigurationCompatibilityError(
                    f"Cannot establish compatibility: historical snapshot "
                    f"{snapshot.snapshot_id} has no {COMPATIBILITY_SIGNATURE_FILENAME}."
                )
            historical = ConfigurationCompatibilitySignature.read(path)
        differences = historical.differences(current)
        if differences:
            detail = ", ".join(differences[:20])
            if len(differences) > 20:
                detail += f", and {len(differences) - 20} more"
            raise ConfigurationCompatibilityError(
                f"Configuration snapshot {snapshot.snapshot_id} is incompatible "
                f"with the current snapshot; changed fields: {detail}."
            )
    _validate_selected_chi2(all_models, which_chi2)
    _validate_parameter_columns(all_models, current.critical_configuration)


def compatibility_signature_path(snapshot: ConfigurationSnapshotReference) -> Path:
    """Return the companion signature path for one resolved snapshot."""
    snapshot_directory = snapshot.absolute_resolved_config_path.parent
    return snapshot_directory / COMPATIBILITY_SIGNATURE_FILENAME


def _critical_configuration(config: Mapping[str, Any]) -> ConfigDict:
    """Project a resolved runtime configuration onto scientifically fixed fields."""
    system_attributes = {
        key: deepcopy(value)
        for key, value in _mapping(config, "system_attributes").items()
        if key != "name"
    }
    return {
        "units": {"internal": deepcopy(_mapping(config, "units")["internal"])},
        "cosmological_parameters": deepcopy(
            _mapping(config, "cosmological_parameters")
        ),
        "system_attributes": system_attributes,
        "mge_settings": deepcopy(_mapping(config, "mge_settings")),
        # Contract version 1 deliberately rejects every numerics change.
        "numerics_settings": deepcopy(_mapping(config, "numerics_settings")),
        "MGEs": sorted(_mapping(config, "MGEs")),
        "spatial_binnings": _settings_without_file(
            _mapping(config, "spatial_binnings"), "bins_file"
        ),
        "potential": _potential_schema(_mapping(config, "potential")),
        "kinematic_data": _settings_without_file(
            _mapping(config, "kinematic_data"), "data_file"
        ),
        "population_data": _settings_without_file(
            _mapping(config, "population_data"), "data_file"
        ),
        "orbit_library_settings": deepcopy(_mapping(config, "orbit_library_settings")),
        "weight_solver_settings": deepcopy(_mapping(config, "weight_solver_settings")),
    }


def _potential_schema(potential: Mapping[str, Any]) -> ConfigDict:
    """Keep component/parameter schema while excluding future search controls."""
    result: ConfigDict = {}
    for component_name, raw_component in potential.items():
        component = _require_mapping(raw_component, Path(f"potential.{component_name}"))
        schema = {
            key: deepcopy(value)
            for key, value in component.items()
            if key != "parameters"
        }
        parameters = _require_mapping(
            component.get("parameters", {}),
            Path(f"potential.{component_name}.parameters"),
        )
        schema["parameters"] = {
            parameter_name: {
                key: deepcopy(value)
                for key, value in _require_mapping(
                    raw_parameter,
                    Path(f"potential.{component_name}.parameters.{parameter_name}"),
                ).items()
                if key not in _SEARCH_PARAMETER_KEYS
            }
            for parameter_name, raw_parameter in parameters.items()
        }
        result[component_name] = schema
    return result


def _settings_without_file(settings: Mapping[str, Any], file_key: str) -> ConfigDict:
    """Copy named scientific settings without their relocatable file paths."""
    return {
        name: {
            key: deepcopy(value)
            for key, value in _require_mapping(raw, Path(name)).items()
            if key != file_key
        }
        for name, raw in settings.items()
    }


def _scientific_input_hashes(config: Mapping[str, Any]) -> ConfigDict:
    """Hash every raw scientific input addressed by the resolved configuration."""
    input_directory = Path(_mapping(config, "io_settings")["input_directory"])
    return {
        "MGEs": {
            name: _file_sha256(_input_path(input_directory, filename))
            for name, filename in _mapping(config, "MGEs").items()
        },
        "spatial_binnings": _named_file_hashes(
            _mapping(config, "spatial_binnings"),
            "bins_file",
            input_directory,
        ),
        "kinematic_data": _named_file_hashes(
            _mapping(config, "kinematic_data"),
            "data_file",
            input_directory,
        ),
        "population_data": _named_file_hashes(
            _mapping(config, "population_data"),
            "data_file",
            input_directory,
        ),
    }


def _named_file_hashes(
    settings: Mapping[str, Any],
    file_key: str,
    input_directory: Path,
) -> ConfigDict:
    """Hash one required file field in every named settings entry."""
    return {
        name: _file_sha256(
            _input_path(
                input_directory,
                _require_mapping(raw, Path(name))[file_key],
            )
        )
        for name, raw in settings.items()
    }


def _input_path(input_directory: Path, filename: Any) -> Path:
    """Resolve and validate one configured scientific input filename."""
    if not isinstance(filename, str) or not filename:
        raise ConfigurationCompatibilityError(
            f"Scientific input filename must be a non-empty string, got {filename!r}."
        )
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = input_directory / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Scientific input file does not exist: {path}")
    return path


def _file_sha256(path: Path) -> str:
    """Hash one potentially large scientific input without loading it all at once."""
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_selected_chi2(all_models: AllModels, which_chi2: str) -> None:
    """Require the current search metric for every successful historical model."""
    if not all_models.has_successful_model():
        return
    if which_chi2 not in all_models.table.colnames:
        raise ConfigurationCompatibilityError(
            f"Cannot resume with which_chi2={which_chi2!r}: the historical "
            "AllModels table has no such column."
        )
    successful = np.asarray(all_models.table["weights_done"], dtype=bool)
    values = np.asarray(all_models.table[which_chi2][successful], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ConfigurationCompatibilityError(
            f"Cannot resume with which_chi2={which_chi2!r}: it is not finite "
            "for every successful historical model."
        )


def _validate_parameter_columns(
    all_models: AllModels,
    critical_configuration: Mapping[str, Any],
) -> None:
    """Require every included potential parameter in a nonempty model table."""
    if not len(all_models):
        return
    potential = _mapping(critical_configuration, "potential")
    expected = {
        f"{component_name}.{parameter_name}"
        for component_name, raw_component in potential.items()
        for component in [_require_mapping(raw_component, Path(component_name))]
        if component.get("include", True)
        for parameter_name in _require_mapping(
            component.get("parameters", {}), Path(f"{component_name}.parameters")
        )
    }
    missing = sorted(expected.difference(all_models.table.colnames))
    if missing:
        raise ConfigurationCompatibilityError(
            "AllModels is missing potential-parameter columns required by the "
            f"current configuration: {missing!r}."
        )


def _different_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    """Recursively report differing mapping/list/scalar paths."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(_different_paths(left[key], right[key], path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        differences = []
        pairs = zip(left, right, strict=True)
        for index, (left_value, right_value) in enumerate(pairs):
            differences.extend(
                _different_paths(left_value, right_value, f"{prefix}[{index}]")
            )
        return differences
    return [] if left == right else [prefix]


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    """Hash a canonical YAML representation with mapping order removed."""
    content = yaml.safe_dump(
        dict(value),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    ).encode("utf-8")
    return sha256(content).hexdigest()


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return one required mapping from a resolved configuration."""
    if key not in config:
        raise ConfigurationCompatibilityError(f"Missing configuration section {key!r}.")
    return _require_mapping(config[key], Path(key))


def _require_mapping(value: Any, path: Path) -> ConfigDict:
    """Return a plain mapping or raise a compatibility-specific error."""
    if not isinstance(value, Mapping):
        raise ConfigurationCompatibilityError(f"{path} must be a mapping.")
    return dict(value)
