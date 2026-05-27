"""
NumPy backend.

Importing this module registers the backend via :func:`register_backend`, so the
package can be auto-loaded on the first ``set_backend("numpy")`` call.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray

from decent_array import Array
from decent_array.interoperability._abstracts import Backend
from decent_array.interoperability._backend_manager import register_backend
from decent_array.types import ArrayKey, DTypes, SupportedArrayTypes, SupportedDevices, SupportedFrameworks


def _unwrap(x: Any) -> Any:  # noqa: ANN401
    """
    Return the underlying value of an :class:`Array`, or pass ``x`` through.

    Typed as ``Any`` because operator dunders may pass either an :class:`Array` or a
    Python scalar; the strict abstract signature would force a ``cast`` at every call
    site without runtime benefit.
    """
    return x.value if type(x) is Array else x


_DTYPE_MAP = {
    DTypes.BOOL: np.bool_,
    DTypes.UINT8: np.uint8,
    DTypes.UINT16: np.uint16,
    DTypes.UINT32: np.uint32,
    DTypes.UINT64: np.uint64,
    DTypes.INT8: np.int8,
    DTypes.INT16: np.int16,
    DTypes.INT32: np.int32,
    DTypes.INT64: np.int64,
    DTypes.FLOAT32: np.float32,
    DTypes.FLOAT64: np.float64,
    DTypes.COMPLEX64: np.complex64,
    DTypes.COMPLEX128: np.complex128,
}


class NumpyBackend(Backend):  # noqa: PLR0904
    """NumPy implementation of :class:`Backend`."""

    def __init__(self, device: SupportedDevices = SupportedDevices.CPU) -> None:
        if device != SupportedDevices.CPU:
            raise ValueError(f"NumPy backend only supports CPU, got '{device.value}'.")
        super().__init__(device)
        self._rng: np.random.Generator = np.random.default_rng()

    # Array creation

    def zeros(self, shape: int | tuple[int, ...]) -> Array:
        return Array(np.zeros(shape))

    def zeros_like(self, x: Array) -> Array:
        return Array(np.zeros_like(x.value))

    def ones(self, shape: int | tuple[int, ...]) -> Array:
        return Array(np.ones(shape))

    def ones_like(self, x: Array) -> Array:
        return Array(np.ones_like(x.value))

    def eye(self, n: int) -> Array:
        return Array(np.eye(n))

    def device_to_native(self, device: SupportedDevices) -> Any:  # noqa: ANN401
        # NumPy has no explicit device management; surface the request unchanged.
        return device

    def device_of(self, x: Array) -> SupportedDevices:  # noqa: ARG002
        return SupportedDevices.CPU

    # Array manipulation

    def copy(self, x: Array) -> Array:
        v = x.value
        if isinstance(v, np.ndarray | np.generic):
            return Array(np.copy(v))
        return Array(deepcopy(v))

    def to_numpy(self, x: SupportedArrayTypes | Array) -> NDArray[Any]:
        """Return the value of an :class:`Array` as a NumPy array."""
        v = x.value if type(x) is Array else x
        if isinstance(v, np.ndarray):
            return v
        return np.asarray(v)

    def from_numpy(self, x: NDArray[Any]) -> Array:
        return Array(x)

    def from_numpy_like(self, x: NDArray[Any], like: Array) -> Array:
        # NumPy has no device dimension, so only the dtype of ``like`` matters.
        return Array(np.asarray(x, dtype=like.value.dtype))

    def asarray(self, x: bool | int | float | complex) -> Array:
        return Array(np.array(x))

    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        if len(arrays) == 0:
            raise ValueError("Cannot stack an empty sequence of arrays.")
        return Array(np.stack([a.value for a in arrays], axis=axis))

    def reshape(self, x: Array, shape: tuple[int, ...]) -> Array:
        return Array(np.reshape(x.value, shape))

    def transpose(self, x: Array, axis: tuple[int, ...] | None = None) -> Array:
        return Array(np.transpose(x.value, axes=axis))

    def shape(self, x: Array) -> tuple[int, ...]:
        return tuple(x.value.shape)

    def size(self, x: Array) -> int:
        return int(x.value.size)

    def ndim(self, x: Array) -> int:
        return int(x.value.ndim)

    def squeeze(self, x: Array, axis: int | tuple[int, ...] | None = None) -> Array:
        return Array(np.squeeze(x.value, axis=axis))

    def unsqueeze(self, x: Array, axis: int) -> Array:
        return Array(np.expand_dims(x.value, axis=axis))

    def diag(self, x: Array) -> Array:
        if x.value.ndim != 1:
            raise ValueError(f"diag requires a 1-D array, got {x.value.ndim}-D")
        return Array(np.diag(x.value))

    def diagonal(self, x: Array, offset: int = 0) -> Array:
        if x.value.ndim != 2:
            raise ValueError(f"diagonal requires a 2-D array, got {x.value.ndim}-D")
        return Array(np.diagonal(x.value, offset=offset))

    def astype(self, x: Array, dtype: DTypes) -> Array:
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"Unsupported dtype '{dtype.value}' for NumPy backend.")
        return Array(np.asarray(x.value, dtype=_DTYPE_MAP[dtype]))

    # Linalg

    def vecdot(self, x1: Array, x2: Array) -> Array:
        return Array(np.dot(x1.value, x2.value))

    def matmul(self, x1: Array, x2: Array) -> Array:
        return Array(x1.value @ x2.value)

    def vector_norm(
        self,
        x: Array,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        ord: int | float = 2,  # noqa: A002
    ) -> Array:
        return Array(np.linalg.norm(x.value, ord=ord, axis=axis, keepdims=keepdims))

    # Math reductions

    def sum(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(np.sum(x.value, axis=axis, keepdims=keepdims))

    def mean(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(np.mean(x.value, axis=axis, keepdims=keepdims))

    def min(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(np.min(x.value, axis=axis, keepdims=keepdims))

    def max(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(np.max(x.value, axis=axis, keepdims=keepdims))

    def any(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:
        return bool(np.any(x.value, axis=axis, keepdims=keepdims))

    def all(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:
        return bool(np.all(x.value, axis=axis, keepdims=keepdims))

    # Math elementwise — operands may be Array or scalar (operator dunders pass either).
    # ``Array | float`` covers both: PEP 484's numeric tower implicitly admits ``int``.

    def add(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.add(_unwrap(x1), _unwrap(x2)))

    def iadd[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value += _unwrap(x2)
        return x1

    def subtract(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.subtract(_unwrap(x1), _unwrap(x2)))

    def isubtract[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value -= _unwrap(x2)
        return x1

    def multiply(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.multiply(_unwrap(x1), _unwrap(x2)))

    def imultiply[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value *= _unwrap(x2)
        return x1

    def divide(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.divide(_unwrap(x1), _unwrap(x2)))

    def idivide[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value /= _unwrap(x2)
        return x1

    def pow(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.power(_unwrap(x1), _unwrap(x2)))

    def negative(self, x: Array) -> Array:
        return Array(np.negative(x.value))

    def absolute(self, x: Array) -> Array:
        return Array(np.abs(x.value))

    def sqrt(self, x: Array) -> Array:
        return Array(np.sqrt(x.value))

    # Comparisons

    def equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.equal(_unwrap(x1), _unwrap(x2)))

    def not_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.not_equal(_unwrap(x1), _unwrap(x2)))

    def less(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.less(_unwrap(x1), _unwrap(x2)))

    def less_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.less_equal(_unwrap(x1), _unwrap(x2)))

    def greater(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.greater(_unwrap(x1), _unwrap(x2)))

    def greater_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.greater_equal(_unwrap(x1), _unwrap(x2)))

    # Bitwise

    def bitwise_and(self, x1: int | Array, x2: int | Array) -> Array:
        return Array(np.bitwise_and(_unwrap(x1), _unwrap(x2)))

    # Operators

    def sign(self, x: Array) -> Array:
        return Array(np.sign(x.value))

    def maximum(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.maximum(_unwrap(x1), _unwrap(x2)))

    def argmax(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return Array(np.argmax(x.value, axis=axis, keepdims=keepdims))

    def argmin(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return Array(np.argmin(x.value, axis=axis, keepdims=keepdims))

    def set_item(self, x: Array, key: ArrayKey, value: bool | int | float | complex | Array) -> None:
        x.value[key] = _unwrap(value)

    def get_item(self, x: Array, key: ArrayKey) -> Array:
        return Array(x.value[key])

    # RNG

    def set_seed(self, seed: int) -> None:
        # Seed both the legacy global state and our owned Generator. The legacy state is
        # important because some downstream libraries (sklearn, pandas) consult it.
        np.random.seed(seed)  # noqa: NPY002
        self._rng = np.random.default_rng(seed)

    def get_rng_state(self) -> dict[str, Any]:
        # ``np.random.get_state()`` returns a tuple by default; ``legacy=False`` returns
        # the equivalent dict form, which both matches the surrounding ``dict[str, Any]``
        # value type (so mypyc's strict union narrowing is satisfied) and round-trips
        # cleanly through ``np.random.set_state``.
        return {
            "numpy_bit_generator_state": deepcopy(self._rng.bit_generator.state),
            "numpy_legacy_state": np.random.get_state(legacy=False),  # noqa: NPY002
        }

    def set_rng_state(self, state: dict[str, Any]) -> None:
        if "numpy_bit_generator_state" in state:
            self._rng = np.random.default_rng()
            self._rng.bit_generator.state = state["numpy_bit_generator_state"]
        if "numpy_legacy_state" in state:
            np.random.set_state(state["numpy_legacy_state"])  # noqa: NPY002

    def normal(self, mean: float = 0.0, std: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        return Array(self._rng.normal(loc=mean, scale=std, size=shape))

    def uniform(self, low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        return Array(self._rng.uniform(low=low, high=high, size=shape))

    def normal_like(self, x: Array, mean: float = 0.0, std: float = 1.0) -> Array:
        return Array(self._rng.normal(loc=mean, scale=std, size=x.value.shape).astype(x.value.dtype))

    def uniform_like(self, x: Array, low: float = 0.0, high: float = 1.0) -> Array:
        return Array(self._rng.uniform(low=low, high=high, size=x.value.shape).astype(x.value.dtype))

    def choice(self, x: Array, size: int, replace: bool = True) -> Array:
        return Array(self._rng.choice(x.value, size=size, replace=replace))


register_backend(SupportedFrameworks.NUMPY, NumpyBackend)
