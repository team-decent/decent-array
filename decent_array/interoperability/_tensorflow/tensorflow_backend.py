"""
TensorFlow backend.

Importing this module registers the backend via :func:`register_backend`, so the
package can be auto-loaded on the first ``set_backend("tensorflow")`` call.

TF eager Tensors are immutable, so :meth:`set_item` round-trips through numpy and the
in-place math operations rebind the wrapper's underlying value.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from decent_array import Array
from decent_array.interoperability._abstracts import Backend
from decent_array.interoperability._backend_manager import register_backend
from decent_array.types import ArrayKey, DTypes, SupportedArrayTypes, SupportedDevices, SupportedFrameworks


def _unwrap(x: Any) -> Any:  # noqa: ANN401
    """Return the underlying value of an :class:`Array`, or pass ``x`` through."""
    return x.value if type(x) is Array else x


_DTYPE_MAP = {
    DTypes.BOOL: tf.bool,
    DTypes.UINT8: tf.uint8,
    DTypes.INT8: tf.int8,
    DTypes.UINT16: tf.uint16,
    DTypes.INT16: tf.int16,
    DTypes.UINT32: tf.uint32,
    DTypes.INT32: tf.int32,
    DTypes.UINT64: tf.uint64,
    DTypes.INT64: tf.int64,
    DTypes.FLOAT16: tf.float16,
    DTypes.FLOAT32: tf.float32,
    DTypes.FLOAT64: tf.float64,
    DTypes.COMPLEX64: tf.complex64,
    DTypes.COMPLEX128: tf.complex128,
}


class TensorflowBackend(Backend):  # noqa: PLR0904
    """TensorFlow implementation of :class:`Backend`."""

    def __init__(self, device: SupportedDevices = SupportedDevices.CPU) -> None:
        super().__init__(device)
        self._native_device: str = self.device_to_native(device)
        self._generator: tf.random.Generator = tf.random.Generator.from_non_deterministic_state(alg="philox")

    # Array creation

    def zeros(self, shape: int | tuple[int, ...]) -> Array:
        with tf.device(self._native_device):
            return Array(tf.zeros(shape))

    def zeros_like(self, x: Array) -> Array:
        return Array(tf.zeros_like(x.value))

    def ones(self, shape: int | tuple[int, ...]) -> Array:
        with tf.device(self._native_device):
            return Array(tf.ones(shape))

    def ones_like(self, x: Array) -> Array:
        return Array(tf.ones_like(x.value))

    def eye(self, n: int) -> Array:
        with tf.device(self._native_device):
            return Array(tf.eye(n))

    def device_to_native(self, device: SupportedDevices) -> str:
        if device in {SupportedDevices.CPU, SupportedDevices.GPU}:
            return f"/{device.value}:0"
        raise ValueError(f"Unsupported device for TensorFlow: {device}")

    def device_of(self, x: Array) -> SupportedDevices:
        device_str = x.value.device.lower()
        if "gpu" in device_str or "cuda" in device_str:
            return SupportedDevices.GPU
        return SupportedDevices.CPU

    # Array manipulation

    def copy(self, x: Array) -> Array:
        return Array(tf.identity(x.value))

    def to_numpy(self, x: SupportedArrayTypes | Array) -> NDArray[Any]:
        """Return the value of an :class:`Array` as a NumPy array."""
        v = x.value if type(x) is Array else x
        if isinstance(v, tf.Tensor):
            ret: NDArray[Any] = v.numpy()
        else:
            ret = np.asarray(v)
        return ret

    def from_numpy(self, x: NDArray[Any]) -> Array:
        """Create an :class:`Array` from a NumPy array."""
        with tf.device(self._native_device):
            return Array(tf.convert_to_tensor(x))

    def from_numpy_like(self, x: NDArray[Any], like: Array) -> Array:
        """Create an :class:`Array` from a NumPy array, on ``like``'s device with ``like``'s dtype."""
        v = like.value
        with tf.device(v.device):
            return Array(tf.convert_to_tensor(x, dtype=v.dtype))

    def asarray(self, x: bool | int | float | complex) -> Array:
        """Convert a Python scalar to an :class:`Array` on this backend."""
        with tf.device(self._native_device):
            # Its not a tf tensor but mypyc doesn't import tf so it complains about unsude type-ignores
            return Array(tf.convert_to_tensor(cast("tf.Tensor", x)))

    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        if len(arrays) == 0:
            raise ValueError("Cannot stack an empty sequence of arrays.")
        return Array(tf.stack([a.value for a in arrays], axis=axis))

    def reshape(self, x: Array, shape: tuple[int, ...]) -> Array:
        return Array(tf.reshape(x.value, shape))

    def transpose(self, x: Array, axis: tuple[int, ...] | None = None) -> Array:
        return Array(tf.transpose(x.value, perm=axis))

    def shape(self, x: Array) -> tuple[int, ...]:
        return cast("tuple[int, ...]", tuple(x.value.shape))

    def size(self, x: Array) -> int:
        return int(tf.size(x.value).numpy())

    def ndim(self, x: Array) -> int:
        return len(x.value.shape)

    def squeeze(self, x: Array, axis: int | tuple[int, ...] | None = None) -> Array:
        return Array(tf.squeeze(x.value, axis=axis))

    def unsqueeze(self, x: Array, axis: int) -> Array:
        return Array(tf.expand_dims(x.value, axis=axis))

    def diag(self, x: Array) -> Array:
        v = x.value
        rank = v.shape.ndims
        if rank != 1:
            raise ValueError(f"diag requires a 1-D tensor, got rank {rank}")
        return Array(tf.linalg.diag(v))

    def diagonal(self, x: Array, offset: int = 0) -> Array:
        v = x.value
        rank = v.shape.ndims
        if rank != 2:
            raise ValueError(f"diagonal requires a 2-D tensor, got rank {rank}")
        return Array(tf.linalg.diag_part(v, k=offset))

    def astype(self, x: Array, dtype: DTypes) -> Array:
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"Unsupported dtype '{dtype.value}' for TensorFlow backend.")
        return Array(tf.cast(x.value, dtype=_DTYPE_MAP[dtype]))

    # Linalg

    def vecdot(self, x1: Array, x2: Array) -> Array:
        return Array(tf.tensordot(x1.value, x2.value, axes=1))

    def matmul(self, x1: Array, x2: Array) -> Array:
        # tf.matmul requires both operands to have ndim >= 2; fall back to tensordot
        # for the vector cases so semantics match numpy / torch / jax matmul.
        a, b = x1.value, x2.value
        if a.shape.ndims is None or b.shape.ndims is None or a.shape.ndims < 2 or b.shape.ndims < 2:
            return Array(tf.tensordot(a, b, axes=1))
        return Array(a @ b)

    def vector_norm(
        self,
        x: Array,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        ord: int | float = 2,  # noqa: A002
    ) -> Array:
        v = x.value
        # tf.norm defaults differ from np.linalg.norm on 2-D inputs (operator vs.
        # Frobenius); match numpy's flat default by reducing over both trailing axes.
        axis = axis if axis is not None else (-2, -1) if v.ndim == 2 else None
        return Array(tf.norm(v, ord=ord, axis=axis, keepdims=keepdims))

    # Math reductions

    def sum(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(tf.reduce_sum(x.value, axis=axis, keepdims=keepdims))

    def mean(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(tf.reduce_mean(x.value, axis=axis, keepdims=keepdims))

    def min(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(tf.reduce_min(x.value, axis=axis, keepdims=keepdims))

    def max(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(tf.reduce_max(x.value, axis=axis, keepdims=keepdims))

    def any(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:
        return bool(tf.reduce_any(tf.cast(x.value, tf.bool), axis=axis, keepdims=keepdims).numpy())

    def all(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:
        return bool(tf.reduce_all(tf.cast(x.value, tf.bool), axis=axis, keepdims=keepdims).numpy())

    # Math elementwise — TF Tensors are immutable; "in-place" ops rebind the wrapper.
    # Operands may be Array or scalar (operator dunders pass either); ``Array | float``
    # covers both because PEP 484's numeric tower implicitly admits ``int``.

    def add(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(tf.add(_unwrap(x1), _unwrap(x2)))

    def iadd[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value = tf.add(x1.value, _unwrap(x2))
        return x1

    def subtract(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(tf.subtract(_unwrap(x1), _unwrap(x2)))

    def isubtract[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value = tf.subtract(x1.value, _unwrap(x2))
        return x1

    def multiply(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(tf.multiply(_unwrap(x1), _unwrap(x2)))

    def imultiply[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value = tf.multiply(x1.value, _unwrap(x2))
        return x1

    def divide(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(tf.divide(_unwrap(x1), _unwrap(x2)))

    def idivide[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value = tf.divide(x1.value, _unwrap(x2))
        return x1

    def pow(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(tf.pow(_unwrap(x1), _unwrap(x2)))

    def negative(self, x: Array) -> Array:
        return Array(tf.negative(x.value))

    def absolute(self, x: Array) -> Array:
        return Array(tf.abs(x.value))

    def sqrt(self, x: Array) -> Array:
        return Array(tf.sqrt(x.value))

    # Comparisons

    def equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(tf.equal(_unwrap(x1), _unwrap(x2)))

    def not_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(tf.not_equal(_unwrap(x1), _unwrap(x2)))

    def less(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(tf.less(_unwrap(x1), _unwrap(x2)))

    def less_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(tf.less_equal(_unwrap(x1), _unwrap(x2)))

    def greater(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(tf.greater(_unwrap(x1), _unwrap(x2)))

    def greater_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(tf.greater_equal(_unwrap(x1), _unwrap(x2)))

    # Bitwise — TF's native ``&`` dispatches to ``tf.math.logical_and`` for bool
    # tensors and ``tf.bitwise.bitwise_and`` for int tensors, matching numpy/torch/jax
    # operator semantics. Calling either named function directly here would constrain
    # us to one dtype family.

    def bitwise_and(self, x1: int | Array, x2: int | Array) -> Array:
        return Array(_unwrap(x1) & _unwrap(x2))

    # Operators

    def sign(self, x: Array) -> Array:
        return Array(tf.sign(x.value))

    def maximum(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(tf.maximum(_unwrap(x1), _unwrap(x2)))

    def argmax(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        v = x.value
        if axis is None:
            flat = tf.argmax(tf.reshape(v, [-1]), axis=0)
            if keepdims:
                ndim = v.shape.ndims or 0
                return Array(tf.reshape(flat, [1] * ndim))
            return Array(flat)
        out = tf.argmax(v, axis=axis)
        if keepdims:
            out = tf.expand_dims(out, axis=axis)
        return Array(out)

    def argmin(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        v = x.value
        if axis is None:
            flat = tf.argmin(tf.reshape(v, [-1]), axis=0)
            if keepdims:
                ndim = v.shape.ndims or 0
                return Array(tf.reshape(flat, [1] * ndim))
            return Array(flat)
        out = tf.argmin(v, axis=axis)
        if keepdims:
            out = tf.expand_dims(out, axis=axis)
        return Array(out)

    def set_item(self, x: Array, key: ArrayKey, value: bool | int | float | complex | Array) -> None:
        # TF eager tensors are immutable; round-trip through numpy so arbitrary indexing
        # patterns (slices, fancy indexing) Just Work. The wrapper is rebound to a fresh
        # tensor on the configured device. This is correct but allocates — algorithms
        # that hammer set_item in tight loops should consider numpy or pytorch.
        original = x.value
        np_array = original.numpy().copy()
        np_array[key] = np.asarray(_unwrap(value))
        with tf.device(self._native_device):
            x.value = tf.convert_to_tensor(np_array, dtype=original.dtype)

    def get_item(self, x: Array, key: ArrayKey) -> Array:
        return Array(x.value[key])

    # RNG

    def set_seed(self, seed: int) -> None:
        tf.random.set_seed(seed)
        self._generator = tf.random.Generator.from_seed(seed, alg="philox")

    def get_rng_state(self) -> dict[str, Any]:
        return {"tf_generator_state": self._generator.state.numpy()}

    def set_rng_state(self, state: dict[str, Any]) -> None:
        if "tf_generator_state" in state:
            self._generator = tf.random.Generator.from_state(state["tf_generator_state"], alg="philox")

    def normal(self, mean: float = 0.0, std: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        with tf.device(self._native_device):
            return Array(self._generator.normal(shape=shape, mean=mean, stddev=std))

    def uniform(self, low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        with tf.device(self._native_device):
            return Array(self._generator.uniform(shape=shape, minval=low, maxval=high))

    def normal_like(self, x: Array, mean: float = 0.0, std: float = 1.0) -> Array:
        v = x.value
        return Array(self._generator.normal(shape=tf.shape(v), mean=mean, stddev=std, dtype=v.dtype))

    def uniform_like(self, x: Array, low: float = 0.0, high: float = 1.0) -> Array:
        v = x.value
        return Array(self._generator.uniform(shape=tf.shape(v), minval=low, maxval=high, dtype=v.dtype))

    def choice(self, x: Array, size: int, replace: bool = True) -> Array:
        v = x.value
        n = v.shape[0]
        if replace:
            indices = self._generator.uniform(shape=(size,), minval=0, maxval=n, dtype=tf.int32)
        else:
            scores = self._generator.uniform(shape=(n,), dtype=tf.float32)
            indices = tf.cast(tf.math.top_k(scores, k=size).indices, tf.int32)
        return Array(tf.gather(v, indices))


register_backend(SupportedFrameworks.TENSORFLOW, TensorflowBackend)
