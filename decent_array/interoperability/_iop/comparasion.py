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


def equal(x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
    """Element-wise equality. Returns an :class:`~decent_array.Array` of bools."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.equal(x1, x2)


def not_equal(x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
    """Element-wise inequality. Returns an :class:`~decent_array.Array` of bools."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.not_equal(x1, x2)


def less(x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
    """Element-wise less-than. Returns an :class:`~decent_array.Array` of bools."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.less(x1, x2)


def less_equal(x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
    """Element-wise less-than-or-equal. Returns an :class:`~decent_array.Array` of bools."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.less_equal(x1, x2)


def greater(x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
    """Element-wise greater-than. Returns an :class:`~decent_array.Array` of bools."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.greater(x1, x2)


def greater_equal(x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
    """Element-wise greater-than-or-equal. Returns an :class:`~decent_array.Array` of bools."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.greater_equal(x1, x2)
