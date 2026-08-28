"""Process-wide numerical policies for TNT's JAX runtime."""

from __future__ import annotations

from threading import Lock

import jax

DEFAULT_JAX_ENABLE_X64 = True

_configuration_lock = Lock()
_configured_jax_enable_x64: bool | None = None


def _initialize_default_jax_precision() -> None:
    """Establish TNT's default before importing JAX-backed TNT modules."""
    jax.config.update("jax_enable_x64", DEFAULT_JAX_ENABLE_X64)


def configure_jax_precision(enable_x64: bool) -> None:
    """Apply one resolved configuration's process-wide JAX precision policy."""
    global _configured_jax_enable_x64

    if not isinstance(enable_x64, bool):
        raise TypeError("enable_x64 must be a boolean.")

    with _configuration_lock:
        if (
            _configured_jax_enable_x64 is not None
            and _configured_jax_enable_x64 != enable_x64
        ):
            raise RuntimeError(
                "JAX x64 is already configured as "
                f"{_configured_jax_enable_x64} by a resolved TNT configuration; "
                f"it cannot be changed to {enable_x64} in the same process. "
                "Start a new Python process for a different JAX precision policy."
            )
        jax.config.update("jax_enable_x64", enable_x64)
        _configured_jax_enable_x64 = enable_x64


_initialize_default_jax_precision()
