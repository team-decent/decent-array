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

from typing import TYPE_CHECKING

from decent_array.interoperability._backend_manager import register_backend_listener

if TYPE_CHECKING:
    from decent_array import Array
    from decent_array.interoperability._abstracts import Backend

_BACKEND_INSTANCE: Backend | None = None
_error = RuntimeError("No backend active: call 'set_backend' with a supported framework to activate one.")


def _update_backend(backend: Backend | None) -> None:
    global _BACKEND_INSTANCE  # noqa: PLW0603
    _BACKEND_INSTANCE = backend


register_backend_listener(_update_backend)


def sign(x: Array) -> Array:
    """Element-wise sign."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.sign(x)


def maximum(x1: Array | float, x2: Array | float) -> Array:
    """Element-wise maximum."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.maximum(x1, x2)


def argmax(x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
    """Index of maximum value along ``axis``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.argmax(x, axis, keepdims)


def argmin(x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
    """Index of minimum value along ``axis``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.argmin(x, axis, keepdims)
