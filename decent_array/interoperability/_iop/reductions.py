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


def sum(  # noqa: A001
    x: Array,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Sum elements of ``x`` along ``axis``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.sum(x, axis, keepdims)


def mean(
    x: Array,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Mean of ``x`` along ``axis``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.mean(x, axis, keepdims)


def min(  # noqa: A001
    x: Array,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Minimum of ``x`` along ``axis``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.min(x, axis, keepdims)


def max(  # noqa: A001
    x: Array,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Maximum of ``x`` along ``axis``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.max(x, axis, keepdims)


def any(x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:  # noqa: A001
    """Return True if any element of ``x`` is truthy."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.any(x, axis, keepdims)


def all(x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:  # noqa: A001
    """Return True if all elements of ``x`` are truthy."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.all(x, axis, keepdims)
