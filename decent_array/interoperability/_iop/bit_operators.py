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

from decent_array._errors import no_backend_error
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


def bitwise_and(x1: bool | int | Array, x2: bool | int | Array) -> Array:
    """Element-wise bitwise/logical AND."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.bitwise_and(x1, x2)


def bitwise_invert(x: Array) -> Array:
    """Element-wise bitwise/logical NOT."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.bitwise_invert(x)


def bitwise_or(x1: bool | int | Array, x2: bool | int | Array) -> Array:
    """Element-wise bitwise/logical OR."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.bitwise_or(x1, x2)


def bitwise_xor(x1: bool | int | Array, x2: bool | int | Array) -> Array:
    """Element-wise bitwise/logical XOR."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.bitwise_xor(x1, x2)


def bitwise_left_shift(x1: int | Array, x2: int | Array) -> Array:
    """Element-wise bitwise left shift."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.bitwise_left_shift(x1, x2)


def bitwise_right_shift(x1: int | Array, x2: int | Array) -> Array:
    """Element-wise bitwise right shift."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.bitwise_right_shift(x1, x2)
