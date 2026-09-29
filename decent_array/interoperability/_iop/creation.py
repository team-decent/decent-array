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


def _update_backend(backend: Backend | None) -> None:
    global _BACKEND_INSTANCE  # noqa: PLW0603
    _BACKEND_INSTANCE = backend


register_backend_listener(_update_backend)


def zeros(shape: int | tuple[int, ...]) -> Array:
    """Create an array of zeros with the given shape."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.zeros(shape)


def zeros_like(x: Array) -> Array:
    """Create an array of zeros matching the shape and type of ``x``."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.zeros_like(x)


def ones(shape: int | tuple[int, ...]) -> Array:
    """Create an array of ones with the given shape."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.ones(shape)


def ones_like(x: Array) -> Array:
    """Create an array of ones matching the shape and type of ``x``."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.ones_like(x)


def empty(shape: int | tuple[int, ...]) -> Array:
    """Create an uninitialized array with the given shape."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.empty(shape)


def empty_like(x: Array, /) -> Array:
    """Create an uninitialized array with the same shape as ``x``."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.empty_like(x)


def eye(n_rows: int, n_cols: int | None = None, /, *, k: int = 0) -> Array:
    """Create a two-dimensional array with ones on the diagonal at offset ``k``."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.eye(n_rows, n_cols, k=k)


def arange(start: int | float, /, stop: int | float | None = None, step: int | float = 1) -> Array:
    """Return evenly spaced values within a half-open interval."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.arange(start, stop, step)


def linspace(
    start: int | float | complex,
    stop: int | float | complex,
    /,
    num: int,
    *,
    endpoint: bool = True,
) -> Array:
    """Return evenly spaced numbers over a specified interval."""
    if _BACKEND_INSTANCE is None:
        raise no_backend_error
    return _BACKEND_INSTANCE.linspace(start, stop, num, endpoint=endpoint)
