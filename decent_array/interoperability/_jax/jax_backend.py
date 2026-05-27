"""
JAX backend.

Importing this module registers the backend via :func:`register_backend`, so the
package can be auto-loaded on the first ``set_backend("jax")`` call.

JAX arrays are immutable, so :meth:`set_item` rebinds the wrapper's underlying value
rather than mutating it.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import time_ns
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from decent_array import Array
from decent_array.interoperability._abstracts import Backend
from decent_array.interoperability._backend_manager import register_backend
from decent_array.types import ArrayKey, DTypes, SupportedArrayTypes, SupportedDevices, SupportedFrameworks


def _unwrap(x: Any) -> Any:  # noqa: ANN401
    """Return the underlying value of an :class:`Array`, or pass ``x`` through."""
    return x.value if type(x) is Array else x


_DTYPE_MAP = {
    DTypes.BOOL: jnp.bool_,
    DTypes.UINT8: jnp.uint8,
    DTypes.UINT16: jnp.uint16,
    DTypes.UINT32: jnp.uint32,
    DTypes.UINT64: jnp.uint64,
    DTypes.INT8: jnp.int8,
    DTypes.INT16: jnp.int16,
    DTypes.INT32: jnp.int32,
    DTypes.INT64: jnp.int64,
    DTypes.FLOAT32: jnp.float32,
    DTypes.FLOAT64: jnp.float64,
    DTypes.COMPLEX64: jnp.complex64,
    DTypes.COMPLEX128: jnp.complex128,
}


class JaxBackend(Backend):  # noqa: PLR0904
    """JAX implementation of :class:`Backend`."""

    def __init__(self, device: SupportedDevices = SupportedDevices.CPU) -> None:
        super().__init__(device)
        self._native_device: jax.Device = self.device_to_native(device)
        self._key: jax.Array = jax.random.key(time_ns())

    # Array creation

    def zeros(self, shape: int | tuple[int, ...]) -> Array:
        return Array(jnp.zeros(shape, device=self._native_device))

    def zeros_like(self, x: Array) -> Array:
        return Array(jnp.zeros_like(x.value))

    def ones(self, shape: int | tuple[int, ...]) -> Array:
        return Array(jnp.ones(shape, device=self._native_device))

    def ones_like(self, x: Array) -> Array:
        return Array(jnp.ones_like(x.value))

    def eye(self, n: int) -> Array:
        return Array(jnp.eye(n, device=self._native_device))

    def device_to_native(self, device: SupportedDevices) -> jax.Device:
        if device == SupportedDevices.CPU:
            return jax.devices("cpu")[0]
        if device == SupportedDevices.GPU:
            return jax.devices("gpu")[0]
        raise ValueError(f"Unsupported device for JAX: {device}")

    def device_of(self, x: Array) -> SupportedDevices:
        platform = x.value.device.platform
        if platform == "gpu":
            return SupportedDevices.GPU
        if platform == "cpu":
            return SupportedDevices.CPU
        raise TypeError(f"Unsupported JAX platform: {platform}")

    # Array manipulation

    def copy(self, x: Array) -> Array:
        return Array(jnp.array(x.value, copy=True))

    def to_numpy(self, x: SupportedArrayTypes | Array) -> NDArray[Any]:
        return np.array(x.value if type(x) is Array else x)

    def from_numpy(self, x: NDArray[Any]) -> Array:
        return Array(jnp.array(x, device=self._native_device))

    def from_numpy_like(self, x: NDArray[Any], like: Array) -> Array:
        v = like.value
        return Array(jnp.asarray(x, dtype=v.dtype, device=v.device))

    def asarray(self, x: bool | int | float | complex) -> Array:
        return Array(jnp.array(x, device=self._native_device))

    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        if len(arrays) == 0:
            raise ValueError("Cannot stack an empty sequence of arrays.")
        return Array(jnp.stack([a.value for a in arrays], axis=axis))

    def reshape(self, x: Array, shape: tuple[int, ...]) -> Array:
        return Array(jnp.reshape(x.value, shape))

    def transpose(self, x: Array, axis: tuple[int, ...] | None = None) -> Array:
        return Array(jnp.transpose(x.value, axes=axis))

    def shape(self, x: Array) -> tuple[int, ...]:
        return tuple(x.value.shape)

    def size(self, x: Array) -> int:
        return int(x.value.size)

    def ndim(self, x: Array) -> int:
        return int(x.value.ndim)

    def squeeze(self, x: Array, axis: int | tuple[int, ...] | None = None) -> Array:
        return Array(jnp.squeeze(x.value, axis=axis))

    def unsqueeze(self, x: Array, axis: int) -> Array:
        return Array(jnp.expand_dims(x.value, axis=axis))

    def diag(self, x: Array) -> Array:
        if x.value.ndim != 1:
            raise ValueError(f"diag requires a 1-D array, got {x.value.ndim}-D")
        return Array(jnp.diag(x.value))

    def diagonal(self, x: Array, offset: int = 0) -> Array:
        if x.value.ndim != 2:
            raise ValueError(f"diagonal requires a 2-D array, got {x.value.ndim}-D")
        return Array(jnp.diagonal(x.value, offset=offset))

    def astype(self, x: Array, dtype: DTypes) -> Array:
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"Unsupported dtype '{dtype.value}' for NumPy backend.")
        return Array(jnp.asarray(x.value, dtype=_DTYPE_MAP[dtype]))

    # Linalg

    def vecdot(self, x1: Array, x2: Array) -> Array:
        return Array(jnp.dot(x1.value, x2.value))

    def matmul(self, x1: Array, x2: Array) -> Array:
        return Array(x1.value @ x2.value)

    def vector_norm(
        self,
        x: Array,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        ord: int | float = 2,  # noqa: A002
    ) -> Array:
        return Array(jnp.linalg.norm(x.value, ord=ord, axis=axis, keepdims=keepdims))

    # Math reductions

    def sum(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(jnp.sum(x.value, axis=axis, keepdims=keepdims))

    def mean(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(jnp.mean(x.value, axis=axis, keepdims=keepdims))

    def min(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(jnp.min(x.value, axis=axis, keepdims=keepdims))

    def max(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(jnp.max(x.value, axis=axis, keepdims=keepdims))

    def any(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:
        return bool(jnp.any(x.value, axis=axis, keepdims=keepdims))

    def all(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:
        return bool(jnp.all(x.value, axis=axis, keepdims=keepdims))

    # Math elementwise — JAX arrays are immutable; "in-place" ops rebind the wrapper.
    # Operands may be Array or scalar (operator dunders pass either); ``Array | float``
    # covers both because PEP 484's numeric tower implicitly admits ``int``.

    def add(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(jnp.add(_unwrap(x1), _unwrap(x2)))

    def iadd[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value = jnp.add(x1.value, _unwrap(x2))
        return x1

    def subtract(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(jnp.subtract(_unwrap(x1), _unwrap(x2)))

    def isubtract[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value = jnp.subtract(x1.value, _unwrap(x2))
        return x1

    def multiply(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(jnp.multiply(_unwrap(x1), _unwrap(x2)))

    def imultiply[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value = jnp.multiply(x1.value, _unwrap(x2))
        return x1

    def divide(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(jnp.divide(_unwrap(x1), _unwrap(x2)))

    def idivide[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value = jnp.divide(x1.value, _unwrap(x2))
        return x1

    def pow(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(jnp.power(_unwrap(x1), _unwrap(x2)))

    def negative(self, x: Array) -> Array:
        return Array(jnp.negative(x.value))

    def absolute(self, x: Array) -> Array:
        return Array(jnp.abs(x.value))

    def sqrt(self, x: Array) -> Array:
        return Array(jnp.sqrt(x.value))

    # Comparisons

    def equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(jnp.equal(_unwrap(x1), _unwrap(x2)))

    def not_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(jnp.not_equal(_unwrap(x1), _unwrap(x2)))

    def less(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(jnp.less(_unwrap(x1), _unwrap(x2)))

    def less_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(jnp.less_equal(_unwrap(x1), _unwrap(x2)))

    def greater(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(jnp.greater(_unwrap(x1), _unwrap(x2)))

    def greater_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(jnp.greater_equal(_unwrap(x1), _unwrap(x2)))

    # Bitwise

    def bitwise_and(self, x1: int | Array, x2: int | Array) -> Array:
        return Array(jnp.bitwise_and(_unwrap(x1), _unwrap(x2)))

    # Operators

    def sign(self, x: Array) -> Array:
        return Array(jnp.sign(x.value))

    def maximum(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(jnp.maximum(_unwrap(x1), _unwrap(x2)))

    def argmax(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return Array(jnp.argmax(x.value, axis=axis, keepdims=keepdims))

    def argmin(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return Array(jnp.argmin(x.value, axis=axis, keepdims=keepdims))

    def set_item(self, x: Array, key: ArrayKey, value: bool | int | float | complex | Array) -> None:
        # JAX arrays are immutable; rebind the wrapper to a new array with `key` updated.
        x.value = x.value.at[key].set(_unwrap(value))

    def get_item(self, x: Array, key: ArrayKey) -> Array:
        return Array(x.value[key])

    # RNG

    def set_seed(self, seed: int) -> None:
        self._key = jax.random.key(seed)

    def get_rng_state(self) -> dict[str, Any]:
        return {"jax_key": jax.random.key_data(self._key)}

    def set_rng_state(self, state: dict[str, Any]) -> None:
        if "jax_key" in state:
            self._key = jax.random.wrap_key_data(state["jax_key"])

    def normal(self, mean: float = 0.0, std: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        sub = self._next_key()
        sample = jax.random.normal(sub, shape=shape).to_device(self._native_device)
        return Array(mean + std * sample)

    def uniform(self, low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        sub = self._next_key()
        return Array(jax.random.uniform(sub, shape=shape, minval=low, maxval=high).to_device(self._native_device))

    def normal_like(self, x: Array, mean: float = 0.0, std: float = 1.0) -> Array:
        v = x.value
        sub = self._next_key()
        sample = jax.random.normal(sub, shape=v.shape, dtype=v.dtype)
        return Array(mean + std * sample)

    def uniform_like(self, x: Array, low: float = 0.0, high: float = 1.0) -> Array:
        v = x.value
        sub = self._next_key()
        return Array(jax.random.uniform(sub, shape=v.shape, dtype=v.dtype, minval=low, maxval=high))

    def choice(self, x: Array, size: int, replace: bool = True) -> Array:
        v = x.value
        sub = self._next_key()
        indices = jax.random.choice(sub, a=v.shape[0], shape=(size,), replace=replace)
        return Array(v[indices])

    # internals

    def _next_key(self) -> jax.Array:
        """Split the stored key, advance state, return a sub-key for one draw."""
        self._key, sub = jax.random.split(self._key)
        return cast("jax.Array", sub)


register_backend(SupportedFrameworks.JAX, JaxBackend)
