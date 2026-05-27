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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from decent_array.interoperability._backend_manager import register_backend_listener

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from decent_array import Array
    from decent_array.interoperability._abstracts import Backend
    from decent_array.types import DTypes, SupportedArrayTypes

_BACKEND_INSTANCE: Backend | None = None
_error = RuntimeError("No backend active: call 'set_backend' with a supported framework to activate one.")


def _update_backend(backend: Backend | None) -> None:
    global _BACKEND_INSTANCE  # noqa: PLW0603
    _BACKEND_INSTANCE = backend


register_backend_listener(_update_backend)


def copy(x: Array) -> Array:
    """Return a copy of ``x``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.copy(x)


def to_numpy(x: SupportedArrayTypes | Array) -> NDArray[Any]:
    """Convert ``x`` to a NumPy array on CPU."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.to_numpy(x)


def from_numpy(x: NDArray[Any]) -> Array:
    """Convert a NumPy array on CPU to an :class:`~decent_array.Array` on the active backend."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.from_numpy(x)


def from_numpy_like(x: NDArray[Any], like: Array) -> Array:
    """Convert a NumPy array to an :class:`~decent_array.Array` matching ``like``'s dtype and device."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.from_numpy_like(x, like)


def asarray(x: float | bool) -> Array:
    """Convert a Python scalar to an :class:`~decent_array.Array` on the active backend."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.asarray(x)


def stack(arrays: Sequence[Array], axis: int = 0) -> Array:
    """Stack a sequence of arrays along a new dimension."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.stack(arrays, axis)


def reshape(x: Array, shape: tuple[int, ...]) -> Array:
    """Reshape ``x`` to ``shape``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.reshape(x, shape)


def transpose(x: Array, axis: tuple[int, ...] | None = None) -> Array:
    """Transpose ``x``; ``None`` reverses dimensions."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.transpose(x, axis)


def shape(x: Array) -> tuple[int, ...]:
    """Return the shape of ``x``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.shape(x)


def size(x: Array) -> int:
    """Return the total number of elements in ``x``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.size(x)


def ndim(x: Array) -> int:
    """Return the number of dimensions of ``x``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.ndim(x)


def squeeze(x: Array, axis: int | tuple[int, ...] | None = None) -> Array:
    """Remove single-dimensional entries from ``x``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.squeeze(x, axis)


def expand_dims(x: Array, axis: int) -> Array:
    """Insert a singleton dimension at ``axis``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.unsqueeze(x, axis)


def unsqueeze(x: Array, axis: int) -> Array:
    """Insert a singleton dimension at ``axis``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.unsqueeze(x, axis)


def diag(x: Array) -> Array:
    """Build a diagonal matrix from a 1-D vector."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.diag(x)


def diagonal(x: Array, offset: int = 0) -> Array:
    """Extract the diagonal entries from a 2-D matrix at the given ``offset``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.diagonal(x, offset)


def astype(x: Array, dtype: DTypes) -> Array:
    """Cast ``x`` to a different dtype."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.astype(x, dtype)
