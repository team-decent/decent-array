"""
Module-level interoperability functions.

Each function delegates to the active backend cached in this module's ``_BACKEND_INSTANCE``
slot. The slot is rebound by :func:`decent_array.interoperability.set_backend`.
Calling any of these before ``set_backend`` raises
:class:`RuntimeError`.

When this module and ``Backend`` are mypyc-compiled in the same group,
``_BACKEND_INSTANCE.add(...)`` dispatches as a native compiled-to-compiled call — no Python
attribute lookup, no bound-method allocation per call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from decent_array.interoperability._backend_manager import register_backend_listener

if TYPE_CHECKING:
    from decent_array import Array
    from decent_array.interoperability._abstracts import Backend
    from decent_array.types import ArrayKey, SupportedDevices

_BACKEND_INSTANCE: Backend | None = None
_error = RuntimeError("No backend active: call 'set_backend' with a supported framework to activate one.")


def _update_backend(backend: Backend | None) -> None:
    global _BACKEND_INSTANCE  # noqa: PLW0603
    _BACKEND_INSTANCE = backend


register_backend_listener(_update_backend)


def device_to_native(device: SupportedDevices) -> Any:  # noqa: ANN401
    """Convert :class:`~decent_array.types.SupportedDevices` to the active backend's native device."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.device_to_native(device)


def device_of(x: Array) -> SupportedDevices:
    """Return the :class:`~decent_array.types.SupportedDevices` of ``x``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.device_of(x)


def set_item(x: Array, key: ArrayKey, value: bool | int | float | complex | Array) -> None:
    """Set ``x[key] = value`` in place."""
    if _BACKEND_INSTANCE is None:
        raise _error
    _BACKEND_INSTANCE.set_item(x, key, value)


def get_item(x: Array, key: ArrayKey) -> Array:
    """Return ``x[key]``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.get_item(x, key)
