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


def add(x1: Array | float, x2: Array | float) -> Array:
    """Element-wise addition."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.add(x1, x2)


def iadd[T: Array](x1: T, x2: Array | float) -> T:
    """In-place element-wise addition."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.iadd(x1, x2)


def subtract(x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
    """Element-wise subtraction."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.subtract(x1, x2)


def isubtract[T: Array](x1: T, x2: int | float | complex | Array) -> T:
    """In-place element-wise subtraction."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.isubtract(x1, x2)


def multiply(x1: Array | float, x2: int | float | complex | Array) -> Array:
    """Element-wise multiplication."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.multiply(x1, x2)


def imultiply[T: Array](x1: T, x2: int | float | complex | Array) -> T:
    """In-place element-wise multiplication."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.imultiply(x1, x2)


def divide(x1: Array | float, x2: int | float | complex | Array) -> Array:
    """Element-wise division."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.divide(x1, x2)


def idivide[T: Array](x1: T, x2: int | float | complex | Array) -> T:
    """In-place element-wise division."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.idivide(x1, x2)


def pow(x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:  # noqa: A001
    """Raise ``x`` to power ``p``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.pow(x1, x2)


def negative(x: Array) -> Array:
    """Element-wise negation."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.negative(x)


def absolute(x: Array) -> Array:
    """Element-wise absolute value."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.absolute(x)


def abs(x: Array) -> Array:  # noqa: A001
    """
    Element-wise absolute value.

    Alias for :func:`absolute`.
    """
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.absolute(x)


def sqrt(x: Array) -> Array:
    """Element-wise square root."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.sqrt(x)
