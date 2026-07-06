"""
CuPy backend.

Importing this module registers the backend via :func:`register_backend`, so the
package can be auto-loaded on the first ``set_backend("cupy")`` call.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, cast

import numpy as np
import cupy as cp
from numpy.typing import NDArray

from decent_array import Array
from decent_array._errors import (
    MatrixTransposeError,
    NDimError,
    UnsupportedDeviceError,
    UnsupportedDTypeCreationError,
    stack_empty_error,
)
from decent_array._utils import is_scalar, unwrap
from decent_array.interoperability._abstracts import Backend
from decent_array.interoperability._backend_manager import register_backend
from decent_array.types import ArrayKey, ArrayTypes, Devices, Frameworks
from decent_array.types._dtypes import dtype


class CupyBackend(Backend):
    """CuPy implementation of :class:`Backend`."""

    def __init__(self, device: Devices = Devices.GPU) -> None:
        super().__init__(device, name=Frameworks.CUPY.value)
        if device != Devices.GPU:
            UnsupportedDeviceError(self.name, device.value)
        self._rng: cp.random.Generator = cp.random.default_rng()

    # Array creation

    def zeros(self, shape: int | tuple[int, ...]) -> Array:
        return Array(cp.zeros(shape))

    def zeros_like(self, x: Array) -> Array:
        return Array(cp.zeros_like(x.value))

    def ones(self, shape: int | tuple[int, ...]) -> Array:
        return Array(cp.ones(shape))

    def ones_like(self, x: Array) -> Array:
        return Array(cp.ones_like(x.value))

    def eye(self, n: int) -> Array:
        return Array(cp.eye(n))

    def device_to_native(self, device: Devices) -> Any:  # noqa: ANN401
        # NumPy has no explicit device management; surface the request unchanged.
        return device

    def device_of(self, x: Array) -> Devices:  # noqa: ARG002
        return Devices.CPU

    # Array manipulation

    def copy(self, x: Array) -> Array:
        v = x.value
        if isinstance(v, cp.ndarray | cp.generic):
            return Array(cp.copy(v))
        return Array(deepcopy(v))

    def to_numpy(self, x: ArrayTypes | Array) -> NDArray[Any]:
        """Return the value of an :class:`Array` as a NumPy array."""
        v = x.value if type(x) is Array else x
        if isinstance(v, cp.ndarray):
            return cp.asnumpy(v)
        return np.asarray(v)

    def from_numpy(self, x: NDArray[Any]) -> Array:
        return Array(cp.asarray(x))

    def from_numpy_like(self, x: NDArray[Any], like: Array) -> Array:
        return Array(cp.asarray(x, dtype=like.value.dtype))

    def asarray(self, x: bool | int | float | complex) -> Array:
        return Array(cp.array(x))

    def to_scalar(self, x: Array) -> Any:  # noqa: ANN401
        """
        Convert a 0-dim array to a scalar.

        Raises:
            TypeError: if ``x`` is not 0-dimensional.

        """
        if not is_scalar(x):
            raise TypeError("Only 0-dim arrays can be converted to Python scalars.")
        return x.value.item()

    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        if len(arrays) == 0:
            raise stack_empty_error
        return Array(np.stack([a.value for a in arrays], axis=axis))

    def reshape(self, x: Array, shape: tuple[int, ...]) -> Array:
        return Array(np.reshape(x.value, shape))

    def transpose(self, x: Array, axis: tuple[int, ...] | None = None) -> Array:
        return Array(np.transpose(x.value, axes=axis))

    def matrix_transpose(self, x: Array) -> Array:
        if x.ndim < 2:
            raise MatrixTransposeError(x.ndim)
        return Array(np.swapaxes(x.value, -1, -2))

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
        if x.ndim != 1:
            raise NDimError(1, x.ndim)
        return Array(np.diag(x.value))

    def diagonal(self, x: Array, offset: int = 0) -> Array:
        if x.ndim != 2:
            raise NDimError(2, x.ndim)
        return Array(np.diagonal(x.value, offset=offset))

    def astype(self, x: Array, dtype: dtype) -> Array:
        if not dtype.available:
            raise UnsupportedDTypeCreationError(dtype, self.name, self.device.value)
        return Array(np.asarray(x.value, dtype=dtype.backend_dtype))

    # Linalg

    def vecdot(self, x1: Array, x2: Array) -> Array:
        return Array(np.dot(x1.value, x2.value))

    def matmul(self, x1: Array, x2: Array) -> Array:
        return Array(x1.value @ x2.value)

    def imatmul[T: Array](self, x1: T, x2: Array) -> T:
        x1.value @= x2.value
        return x1

    def vector_norm(
        self,
        x: Array,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        ord: int | float = 2,  # noqa: A002
    ) -> Array:
        # axis in np.linalg.vector_norm seems to allow for int | tuple[int, ...] | None at runtime,
        # but is still typed as int | tuple[int, int] | None, hence the ignore
        return Array(np.linalg.vector_norm(x.value, ord=ord, axis=axis, keepdims=keepdims))  # type: ignore[arg-type]

    # Math reductions

    def sum(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        v = cast("np.ndarray[Any, Any]", x.value)
        if keepdims:
            return Array(np.sum(v, axis=axis, keepdims=True))
        return Array(np.sum(v, axis=axis))

    def mean(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        v = cast("np.ndarray[Any, Any]", x.value)
        if keepdims:
            return Array(np.mean(v, axis=axis, keepdims=True))
        return Array(np.mean(v, axis=axis))

    def min(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        v = cast("np.ndarray[Any, Any]", x.value)
        if keepdims:
            return Array(np.min(v, axis=axis, keepdims=True))
        return Array(np.min(v, axis=axis))

    def max(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        v = cast("np.ndarray[Any, Any]", x.value)
        if keepdims:
            return Array(np.max(v, axis=axis, keepdims=True))
        return Array(np.max(v, axis=axis))

    def any(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:
        v = cast("np.ndarray[Any, Any]", x.value)
        if keepdims:
            return bool(np.any(v, axis=axis, keepdims=True))
        return bool(np.any(v, axis=axis))

    def all(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:
        v = cast("np.ndarray[Any, Any]", x.value)
        if keepdims:
            return bool(np.all(v, axis=axis, keepdims=True))
        return bool(np.all(v, axis=axis))

    # Math elementwise — operands may be Array or scalar (operator dunders pass either).
    # ``Array | float`` covers both: PEP 484's numeric tower implicitly admits ``int``.

    def add(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.add(unwrap(x1), unwrap(x2)))

    def iadd[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value += unwrap(x2)
        return x1

    def subtract(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.subtract(unwrap(x1), unwrap(x2)))

    def isubtract[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value -= unwrap(x2)
        return x1

    def multiply(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.multiply(unwrap(x1), unwrap(x2)))

    def imultiply[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value *= unwrap(x2)
        return x1

    def divide(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.divide(unwrap(x1), unwrap(x2)))

    def idivide[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value /= unwrap(x2)
        return x1

    def floor_divide(self, x1: int | float | Array, x2: int | float | Array) -> Array:
        return Array(np.floor_divide(unwrap(x1), unwrap(x2)))

    def ifloordiv[T: Array](self, x1: T, x2: int | float | Array) -> T:
        x1.value //= unwrap(x2)
        return x1

    def remainder(self, x1: int | float | Array, x2: int | float | Array) -> Array:
        return Array(np.remainder(unwrap(x1), unwrap(x2)))

    def imod[T: Array](self, x1: T, x2: int | float | Array) -> T:
        x1.value %= unwrap(x2)
        return x1

    def pow(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.power(unwrap(x1), unwrap(x2)))

    def ipow[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value **= unwrap(x2)
        return x1

    def negative(self, x: Array) -> Array:
        return Array(np.negative(x.value))

    def absolute(self, x: Array) -> Array:
        return Array(np.abs(x.value))

    def sqrt(self, x: Array) -> Array:
        return Array(np.sqrt(x.value))

    # Comparisons

    def equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.equal(unwrap(x1), unwrap(x2)))

    def not_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.not_equal(unwrap(x1), unwrap(x2)))

    def less(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.less(unwrap(x1), unwrap(x2)))

    def less_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.less_equal(unwrap(x1), unwrap(x2)))

    def greater(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.greater(unwrap(x1), unwrap(x2)))

    def greater_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.greater_equal(unwrap(x1), unwrap(x2)))

    # Bitwise

    def bitwise_and(self, x1: bool | int | Array, x2: bool | int | Array) -> Array:
        return Array(np.bitwise_and(unwrap(x1), unwrap(x2)))

    def iand[T: Array](self, x1: T, x2: bool | int | Array) -> T:
        x1.value &= unwrap(x2)
        return x1

    def bitwise_invert(self, x: Array) -> Array:
        return Array(np.bitwise_not(x.value))

    def bitwise_or(self, x1: bool | int | Array, x2: bool | int | Array) -> Array:
        return Array(np.bitwise_or(unwrap(x1), unwrap(x2)))

    def ior[T: Array](self, x1: T, x2: bool | int | Array) -> T:
        x1.value |= unwrap(x2)
        return x1

    def bitwise_xor(self, x1: bool | int | Array, x2: bool | int | Array) -> Array:
        return Array(np.bitwise_xor(unwrap(x1), unwrap(x2)))

    def ixor[T: Array](self, x1: T, x2: bool | int | Array) -> T:
        x1.value ^= unwrap(x2)
        return x1

    def bitwise_left_shift(self, x1: int | Array, x2: int | Array) -> Array:
        return Array(np.left_shift(unwrap(x1), unwrap(x2)))

    def ilshift[T: Array](self, x1: T, x2: int | Array) -> T:
        x1.value <<= unwrap(x2)
        return x1

    def bitwise_right_shift(self, x1: int | Array, x2: int | Array) -> Array:
        return Array(np.right_shift(unwrap(x1), unwrap(x2)))

    def irshift[T: Array](self, x1: T, x2: int | Array) -> T:
        x1.value >>= unwrap(x2)
        return x1

    # Operators

    def sign(self, x: Array) -> Array:
        return Array(np.sign(x.value))

    def maximum(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(np.maximum(unwrap(x1), unwrap(x2)))

    def argmax(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        v = cast("np.ndarray[Any, Any]", x.value)
        if keepdims:
            return Array(np.argmax(v, axis=axis, keepdims=True))
        return Array(np.argmax(v, axis=axis))

    def argmin(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        v = cast("np.ndarray[Any, Any]", x.value)
        if keepdims:
            return Array(np.argmin(v, axis=axis, keepdims=True))
        return Array(np.argmin(v, axis=axis))

    def set_item(self, x: Array, key: ArrayKey, value: bool | int | float | complex | Array) -> None:
        x.value[key] = unwrap(value)

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

    # Dtypes

    @property
    def bool_(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp.bool_)

    @property
    def uint8(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp.uint8)

    @property
    def uint16(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp.uint16)

    @property
    def uint32(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp.uint32)

    @property
    def uint64(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp)

    @property
    def int8(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp.int8)

    @property
    def int16(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp.int16)

    @property
    def int32(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp.int32)

    @property
    def int64(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp.int64)

    @property
    def float16(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp.float16)

    @property
    def bfloat16(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp.bfloat16)

    @property
    def float32(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp.float32)

    @property
    def float64(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp.float64)

    @property
    def complex64(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp.complex64)

    @property
    def complex128(self) -> Any:  # noqa: ANN401
        return cp.dtype(cp.complex128)

    # Constants

    @property
    def e(self) -> Any:  # noqa: ANN401
        """e = 2.71828..."""  # noqa: D403
        return cp.e

    @property
    def inf(self) -> Any:  # noqa: ANN401
        """Infinity."""
        return cp.inf

    @property
    def nan(self) -> Any:  # noqa: ANN401
        """Not-a-number."""
        return cp.nan

    @property
    def pi(self) -> Any:  # noqa: ANN401
        """pi = 3.14159..."""  # noqa: D403
        return cp.pi


register_backend(Frameworks.CUPY, CupyBackend)
