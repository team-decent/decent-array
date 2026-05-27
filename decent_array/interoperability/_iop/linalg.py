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


def vecdot(x1: Array, x2: Array) -> Array:
    """Vector dot product of two arrays."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.vecdot(x1, x2)


def dot(x1: Array, x2: Array) -> Array:
    """
    Vector dot product of two arrays.

    Alias for :func:`vecdot`.
    """
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.vecdot(x1, x2)


def matmul(x1: Array, x2: Array) -> Array:
    """Matrix multiplication of two arrays."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.matmul(x1, x2)


def vector_norm(
    x: Array,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: int | float = 2,  # noqa: A002
) -> Array:
    """Vector norm of ``x``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.vector_norm(x, axis, keepdims, ord)


def norm(
    x: Array,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: int | float = 2,  # noqa: A002
) -> Array:
    """
    Vector norm of ``x``.

    Alias for :func:`vector_norm`.
    """
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.vector_norm(x, axis, keepdims, ord)
