"""
CuPy backend.

Importing this module registers the backend via :func:`register_backend`, so the
package can be auto-loaded on the first ``set_backend("cupy")`` call.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from decent_array import Array
from decent_array._errors import (
    MatrixTransposeError,
    NDimError,
    NotScalarError,
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
        self._native_device: cp.cuda.Device = self.device_to_native(device)
        self._seed: int = 0
        self._rng: cp.random.Generator = cp.random.default_rng(seed=self._seed)

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
        if device == Devices.GPU:
            # currently decent-array only supports specifying "gpu", not the device index;
            # so this returns the current device as selected by cupy
            return cp.cuda.Device()
        raise UnsupportedDeviceError(self.name, device.value)

    def device_of(self, x: Array) -> Devices:  # noqa: ARG002
        return Devices.GPU

    # Array manipulation

    def copy(self, x: Array) -> Array:
        v = x.value
        if isinstance(v, cp.ndarray | cp.generic):
            return Array(cp.copy(v))
        return Array(deepcopy(v))

    def to_numpy(self, x: ArrayTypes | Array) -> cp.ndarray[Any]:
        """Return the value of an :class:`Array` as a NumPy array."""
        v = x.value if type(x) is Array else x
        if isinstance(v, cp.ndarray):
            return cp.asnumpy(v)
        return cp.asarray(v)

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
            raise NotScalarError(x.ndim)
        return x.value.item()

    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        if len(arrays) == 0:
            raise stack_empty_error
        return Array(cp.stack([a.value for a in arrays], axis=axis))

    def reshape(self, x: Array, shape: tuple[int, ...]) -> Array:
        return Array(cp.reshape(x.value, shape))

    def transpose(self, x: Array, axis: tuple[int, ...] | None = None) -> Array:
        return Array(cp.transpose(x.value, axes=axis))

    def matrix_transpose(self, x: Array) -> Array:
        if x.ndim < 2:
            raise MatrixTransposeError(x.ndim)
        return Array(cp.matrix_transpose(x.value))

    def shape(self, x: Array) -> tuple[int, ...]:
        return tuple(x.value.shape)

    def size(self, x: Array) -> int:
        return int(x.value.size)

    def ndim(self, x: Array) -> int:
        return int(x.value.ndim)

    def squeeze(self, x: Array, axis: int | tuple[int, ...] | None = None) -> Array:
        return Array(cp.squeeze(x.value, axis=axis))

    def unsqueeze(self, x: Array, axis: int) -> Array:
        return Array(cp.expand_dims(x.value, axis=axis))

    def diag(self, x: Array) -> Array:
        if x.ndim != 1:
            raise NDimError(1, x.ndim)
        return Array(cp.diag(x.value))

    def diagonal(self, x: Array, offset: int = 0) -> Array:
        if x.ndim != 2:
            raise NDimError(2, x.ndim)
        return Array(cp.diagonal(x.value, offset=offset))

    def astype(self, x: Array, dtype: dtype) -> Array:
        if not dtype.available:
            raise UnsupportedDTypeCreationError(dtype, self.name, self.device.value)
        return Array(cp.astype(x.value, dtype=dtype.backend_dtype))

    # Linalg

    def vecdot(self, x1: Array, x2: Array) -> Array:
        return Array(cp.dot(x1.value, x2.value))

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
        if isinstance(axis, tuple) and len(axis) > 2:
            raise ValueError(f"'axis' of length {len(axis)} is currently unsupported by {self.name}")
        return Array(cp.linalg.norm(x.value, ord=ord, axis=axis, keepdims=keepdims))

    # Math reductions

    def sum(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(cp.sum(x.value, axis=axis, keepdims=keepdims))

    def mean(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(cp.mean(x.value, axis=axis, keepdims=keepdims))

    def min(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(cp.min(x.value, axis=axis, keepdims=keepdims))

    def max(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(cp.max(x.value, axis=axis, keepdims=keepdims))

    def any(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:
        return bool(cp.any(x.value, axis=axis, keepdims=keepdims))

    def all(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:
        return bool(cp.all(x.value, axis=axis, keepdims=keepdims))

    # Math elementwise — operands may be Array or scalar (operator dunders pass either).
    # ``Array | float`` covers both: PEP 484's numeric tower implicitly admits ``int``.

    def add(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(cp.add(unwrap(x1), unwrap(x2)))

    def iadd[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value += unwrap(x2)
        return x1

    def subtract(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(cp.subtract(unwrap(x1), unwrap(x2)))

    def isubtract[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value -= unwrap(x2)
        return x1

    def multiply(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(cp.multiply(unwrap(x1), unwrap(x2)))

    def imultiply[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value *= unwrap(x2)
        return x1

    def divide(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(cp.divide(unwrap(x1), unwrap(x2)))

    def idivide[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value /= unwrap(x2)
        return x1

    def floor_divide(self, x1: int | float | Array, x2: int | float | Array) -> Array:
        return Array(cp.floor_divide(unwrap(x1), unwrap(x2)))

    def ifloordiv[T: Array](self, x1: T, x2: int | float | Array) -> T:
        x1.value //= unwrap(x2)
        return x1

    def remainder(self, x1: int | float | Array, x2: int | float | Array) -> Array:
        return Array(cp.remainder(unwrap(x1), unwrap(x2)))

    def imod[T: Array](self, x1: T, x2: int | float | Array) -> T:
        x1.value %= unwrap(x2)
        return x1

    def pow(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(cp.power(unwrap(x1), unwrap(x2)))

    def ipow[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value **= unwrap(x2)
        return x1

    def negative(self, x: Array) -> Array:
        return Array(cp.negative(x.value))

    def absolute(self, x: Array) -> Array:
        return Array(cp.absolute(x.value))

    def sqrt(self, x: Array) -> Array:
        return Array(cp.sqrt(x.value))

    # Comparisons

    def equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(cp.equal(unwrap(x1), unwrap(x2)))

    def not_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(cp.not_equal(unwrap(x1), unwrap(x2)))

    def less(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(cp.less(unwrap(x1), unwrap(x2)))

    def less_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(cp.less_equal(unwrap(x1), unwrap(x2)))

    def greater(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(cp.greater(unwrap(x1), unwrap(x2)))

    def greater_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(cp.greater_equal(unwrap(x1), unwrap(x2)))

    # Bitwise

    def bitwise_and(self, x1: bool | int | Array, x2: bool | int | Array) -> Array:
        return Array(cp.bitwise_and(unwrap(x1), unwrap(x2)))

    def iand[T: Array](self, x1: T, x2: bool | int | Array) -> T:
        x1.value &= unwrap(x2)
        return x1

    def bitwise_invert(self, x: Array) -> Array:
        return Array(cp.bitwise_invert(x.value))

    def bitwise_or(self, x1: bool | int | Array, x2: bool | int | Array) -> Array:
        return Array(cp.bitwise_or(unwrap(x1), unwrap(x2)))

    def ior[T: Array](self, x1: T, x2: bool | int | Array) -> T:
        x1.value |= unwrap(x2)
        return x1

    def bitwise_xor(self, x1: bool | int | Array, x2: bool | int | Array) -> Array:
        return Array(cp.bitwise_xor(unwrap(x1), unwrap(x2)))

    def ixor[T: Array](self, x1: T, x2: bool | int | Array) -> T:
        x1.value ^= unwrap(x2)
        return x1

    def bitwise_left_shift(self, x1: int | Array, x2: int | Array) -> Array:
        return Array(cp.bitwise_left_shift(unwrap(x1), unwrap(x2)))

    def ilshift[T: Array](self, x1: T, x2: int | Array) -> T:
        x1.value <<= unwrap(x2)
        return x1

    def bitwise_right_shift(self, x1: int | Array, x2: int | Array) -> Array:
        return Array(cp.bitwise_right_shift(unwrap(x1), unwrap(x2)))

    def irshift[T: Array](self, x1: T, x2: int | Array) -> T:
        x1.value >>= unwrap(x2)
        return x1

    # Operators

    def sign(self, x: Array) -> Array:
        return Array(cp.sign(x.value))

    def maximum(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(cp.maximum(unwrap(x1), unwrap(x2)))

    def argmax(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return Array(cp.argmax(x.value, axis=axis, keepdims=keepdims))

    def argmin(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return Array(cp.argmin(x.value, axis=axis, keepdims=keepdims))

    def set_item(self, x: Array, key: ArrayKey, value: bool | int | float | complex | Array) -> None:
        x.value[key] = unwrap(value)

    def get_item(self, x: Array, key: ArrayKey) -> Array:
        return Array(x.value[key])

    # RNG
    # CuPy is still limited in how random states are handled; cp.random.Generator does not expose a serializable state;
    # also, the function cupy.random.get_random_state() returns a cp.random.RandomState for the current device, but this
    # object is not serializable. To work around this limitation, CupyBackend has a _seed attribute that stores the seed
    # at any time, and this seed is returned and set by get_rng_state and set_rng_state, respectively.
    # This might result in some loss of reproducibility, but it allows overall compatibility with the codebase.

    def set_seed(self, seed: int) -> None:
        self._seed = seed
        self._rng = cp.random.default_rng(self._seed)

    def get_rng_state(self) -> dict[str, Any]:
        return {"cupy_generator_seed": self._seed}

    def set_rng_state(self, state: dict[str, Any]) -> None:
        if "cupy_generator_seed" in state:
            self._seed = state["cupy_generator_seed"]
            self._rng = cp.random.default_rng(self._seed)

    def normal(self, mean: float = 0.0, std: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        return Array(self._rng.normal(loc=mean, scale=std, size=shape))

    def uniform(self, low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        return Array(self._rng.uniform(low=low, high=high, size=shape))

    def normal_like(self, x: Array, mean: float = 0.0, std: float = 1.0) -> Array:
        return Array(self._rng.normal(loc=mean, scale=std, size=x.value.shape, dtype=x.dtype.backend_dtype))

    def uniform_like(self, x: Array, low: float = 0.0, high: float = 1.0) -> Array:
        return Array(self._rng.uniform(low=low, high=high, size=x.value.shape, dtype=x.dtype.backend_dtype))

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
        return np.e

    @property
    def inf(self) -> Any:  # noqa: ANN401
        """Infinity."""
        return np.inf

    @property
    def nan(self) -> Any:  # noqa: ANN401
        """Not-a-number."""
        return np.nan

    @property
    def pi(self) -> Any:  # noqa: ANN401
        """pi = 3.14159..."""  # noqa: D403
        return np.pi


register_backend(Frameworks.CUPY, CupyBackend)
