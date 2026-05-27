"""
PyTorch backend.

Importing this module registers the backend via :func:`register_backend`, so the
package can be auto-loaded on the first ``set_backend("pytorch")`` call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from decent_array import Array
from decent_array.interoperability._abstracts import Backend
from decent_array.interoperability._backend_manager import register_backend
from decent_array.types import ArrayKey, DTypes, SupportedArrayTypes, SupportedDevices, SupportedFrameworks


def _unwrap(x: Any) -> Any:  # noqa: ANN401
    """Return the underlying value of an :class:`Array`, or pass ``x`` through."""
    return x.value if type(x) is Array else x


_DTYPE_MAP = {
    DTypes.BOOL: torch.bool,
    DTypes.UINT8: torch.uint8,
    DTypes.UINT16: torch.uint16,
    DTypes.UINT32: torch.uint32,
    DTypes.UINT64: torch.uint64,
    DTypes.INT8: torch.int8,
    DTypes.INT16: torch.int16,
    DTypes.INT32: torch.int32,
    DTypes.INT64: torch.int64,
    DTypes.FLOAT32: torch.float32,
    DTypes.FLOAT64: torch.float64,
    DTypes.COMPLEX64: torch.complex64,
    DTypes.COMPLEX128: torch.complex128,
}


class PyTorchBackend(Backend):  # noqa: PLR0904
    """PyTorch implementation of :class:`Backend`."""

    def __init__(self, device: SupportedDevices = SupportedDevices.CPU) -> None:
        super().__init__(device)
        self._native_device: str = self.device_to_native(device)
        self._generator: torch.Generator = torch.Generator(device=self._native_device)

    # Array creation

    def zeros(self, shape: int | tuple[int, ...]) -> Array:
        return Array(torch.zeros(shape, device=self._native_device))

    def zeros_like(self, x: Array) -> Array:
        return Array(torch.zeros_like(x.value))

    def ones(self, shape: int | tuple[int, ...]) -> Array:
        return Array(torch.ones(shape, device=self._native_device))

    def ones_like(self, x: Array) -> Array:
        return Array(torch.ones_like(x.value))

    def eye(self, n: int) -> Array:
        return Array(torch.eye(n, device=self._native_device))

    def device_to_native(self, device: SupportedDevices) -> str:
        if device == SupportedDevices.CPU:
            return "cpu"
        if device == SupportedDevices.GPU:
            return "cuda"
        if device == SupportedDevices.MPS:
            return "mps"
        raise ValueError(f"Unsupported device: {device}")

    def device_of(self, x: Array) -> SupportedDevices:
        kind = x.value.device.type
        if kind == "cpu":
            return SupportedDevices.CPU
        if kind == "cuda":
            return SupportedDevices.GPU
        if kind == "mps":
            return SupportedDevices.MPS
        raise TypeError(f"Unsupported PyTorch device type: {kind}")

    # Array manipulation

    def copy(self, x: Array) -> Array:
        return Array(x.value.detach().clone())

    def to_numpy(self, x: SupportedArrayTypes | Array) -> NDArray[Any]:
        """Return the value of an :class:`Array` as a NumPy array."""
        v = x.value if type(x) is Array else x
        if isinstance(v, torch.Tensor):
            ret: NDArray[Any] = v.cpu().numpy()
        else:
            ret = np.asarray(v)
        return ret

    def from_numpy(self, x: NDArray[Any]) -> Array:
        return Array(torch.from_numpy(x).to(device=self._native_device))

    def from_numpy_like(self, x: NDArray[Any], like: Array) -> Array:
        v = like.value
        return Array(torch.from_numpy(x).to(dtype=v.dtype, device=v.device))

    def asarray(self, x: bool | int | float | complex) -> Array:
        return Array(torch.tensor(x, device=self._native_device))

    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        if len(arrays) == 0:
            raise ValueError("Cannot stack an empty sequence of arrays.")
        return Array(torch.stack([a.value for a in arrays], dim=axis))

    def reshape(self, x: Array, shape: tuple[int, ...]) -> Array:
        return Array(torch.reshape(x.value, shape))

    def transpose(self, x: Array, axis: tuple[int, ...] | None = None) -> Array:
        v = x.value
        dims = axis if axis is not None else tuple(reversed(range(v.ndim)))
        return Array(torch.permute(v, dims=dims))

    def shape(self, x: Array) -> tuple[int, ...]:
        return tuple(x.value.shape)

    def size(self, x: Array) -> int:
        return int(x.value.numel())

    def ndim(self, x: Array) -> int:
        return int(x.value.ndim)

    def squeeze(self, x: Array, axis: int | tuple[int, ...] | None = None) -> Array:
        v = x.value
        if axis is None:
            return Array(torch.squeeze(v))
        return Array(torch.squeeze(v, dim=axis))

    def unsqueeze(self, x: Array, axis: int) -> Array:
        return Array(torch.unsqueeze(x.value, dim=axis))

    def diag(self, x: Array) -> Array:
        if x.value.ndim != 1:
            raise ValueError(f"diag requires a 1-D array, got {x.value.ndim}-D")
        return Array(torch.diag(x.value))

    def diagonal(self, x: Array, offset: int = 0) -> Array:
        if x.value.ndim != 2:
            raise ValueError(f"diagonal requires a 2-D array, got {x.value.ndim}-D")
        return Array(torch.diagonal(x.value, offset=offset))

    def astype(self, x: Array, dtype: DTypes) -> Array:
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"Unsupported dtype '{dtype.value}' for PyTorch backend.")
        return Array(x.value.to(dtype=_DTYPE_MAP[dtype]))

    # Linalg

    def vecdot(self, x1: Array, x2: Array) -> Array:
        return Array(torch.dot(x1.value, x2.value))

    def matmul(self, x1: Array, x2: Array) -> Array:
        return Array(x1.value @ x2.value)

    def vector_norm(
        self,
        x: Array,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        ord: int | float = 2,  # noqa: A002
    ) -> Array:
        return Array(torch.linalg.norm(x.value, ord=ord, axis=axis, keepdim=keepdims))

    # Math reductions

    def sum(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        v = x.value
        if axis is None:
            return Array(torch.sum(v))
        return Array(torch.sum(v, dim=axis, keepdim=keepdims))

    def mean(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        v = x.value
        if axis is None:
            return Array(torch.mean(v))
        return Array(torch.mean(v, dim=axis, keepdim=keepdims))

    def min(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        v = x.value
        if axis is None:
            return Array(torch.min(v))
        return Array(torch.amin(v, dim=axis, keepdim=keepdims))

    def max(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        v = x.value
        if axis is None:
            return Array(torch.max(v))
        return Array(torch.amax(v, dim=axis, keepdim=keepdims))

    def any(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:
        return bool(torch.any(x.value, dim=axis, keepdim=keepdims).item())

    def all(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:
        return bool(torch.all(x.value, dim=axis, keepdim=keepdims).item())

    # Math elementwise — operands may be Array or scalar (operator dunders pass either).
    # ``Array | float`` covers both: PEP 484's numeric tower implicitly admits ``int``.

    def add(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(torch.add(_unwrap(x1), _unwrap(x2)))

    def iadd[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value.add_(_unwrap(x2))
        return x1

    def subtract(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(torch.sub(_unwrap(x1), _unwrap(x2)))

    def isubtract[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value.sub_(_unwrap(x2))
        return x1

    def multiply(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(torch.mul(_unwrap(x1), _unwrap(x2)))

    def imultiply[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value.mul_(_unwrap(x2))
        return x1

    def divide(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(torch.div(_unwrap(x1), _unwrap(x2)))

    def idivide[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        x1.value.div_(_unwrap(x2))
        return x1

    def pow(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(torch.pow(_unwrap(x1), _unwrap(x2)))

    def negative(self, x: Array) -> Array:
        return Array(torch.neg(x.value))

    def absolute(self, x: Array) -> Array:
        return Array(torch.abs(x.value))

    def sqrt(self, x: Array) -> Array:
        return Array(torch.sqrt(x.value))

    # Comparisons

    def equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(torch.eq(_unwrap(x1), _unwrap(x2)))

    def not_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(torch.ne(_unwrap(x1), _unwrap(x2)))

    def less(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(torch.lt(_unwrap(x1), _unwrap(x2)))

    def less_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(torch.le(_unwrap(x1), _unwrap(x2)))

    def greater(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(torch.gt(_unwrap(x1), _unwrap(x2)))

    def greater_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        return Array(torch.ge(_unwrap(x1), _unwrap(x2)))

    # Bitwise

    def bitwise_and(self, x1: int | Array, x2: int | Array) -> Array:
        return Array(torch.bitwise_and(_unwrap(x1), _unwrap(x2)))

    # Operators

    def sign(self, x: Array) -> Array:
        return Array(torch.sign(x.value))

    def maximum(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        a, b = _unwrap(x1), _unwrap(x2)
        # torch.maximum requires both operands to be Tensors; lift Python scalars to
        # match the dtype/device of the tensor operand so the contract matches numpy.
        if not isinstance(a, torch.Tensor):
            ref = b if isinstance(b, torch.Tensor) else None
            a = torch.tensor(a, dtype=ref.dtype if ref is not None else None, device=self._native_device)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=a.dtype, device=a.device)
        return Array(torch.maximum(a, b))

    def argmax(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return Array(torch.argmax(x.value, dim=axis, keepdim=keepdims))

    def argmin(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return Array(torch.argmin(x.value, dim=axis, keepdim=keepdims))

    def set_item(self, x: Array, key: ArrayKey, value: bool | int | float | complex | Array) -> None:
        x.value[key] = _unwrap(value)

    def get_item(self, x: Array, key: ArrayKey) -> Array:
        return Array(x.value[key])

    # RNG

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._generator.manual_seed(seed)

    def get_rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "torch_cpu_state": torch.random.get_rng_state(),
            "torch_generator_state": self._generator.get_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda_states"] = torch.cuda.get_rng_state_all()
        return state

    def set_rng_state(self, state: dict[str, Any]) -> None:
        if "torch_cpu_state" in state:
            torch.random.set_rng_state(state["torch_cpu_state"])
        if "torch_cuda_states" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda_states"])
        if "torch_generator_state" in state:
            self._generator.set_state(state["torch_generator_state"])

    def normal(self, mean: float = 0.0, std: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        return Array(
            torch.normal(mean=mean, std=std, size=shape, device=self._native_device, generator=self._generator)
        )

    def uniform(self, low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        rand = torch.rand(size=shape, device=self._native_device, generator=self._generator)
        return Array((high - low) * rand + low)

    def normal_like(self, x: Array, mean: float = 0.0, std: float = 1.0) -> Array:
        v = x.value
        return Array(
            torch.normal(
                mean=mean,
                std=std,
                size=tuple(v.shape),
                dtype=v.dtype,
                device=v.device,
                generator=self._generator,
            )
        )

    def uniform_like(self, x: Array, low: float = 0.0, high: float = 1.0) -> Array:
        v = x.value
        rand = torch.rand(size=tuple(v.shape), dtype=v.dtype, device=v.device, generator=self._generator)
        return Array((high - low) * rand + low)

    def choice(self, x: Array, size: int, replace: bool = True) -> Array:
        v = x.value
        weights = torch.ones(v.shape[0], device=v.device)
        indices = weights.multinomial(num_samples=size, replacement=replace, generator=self._generator)
        return Array(v[indices])


register_backend(SupportedFrameworks.PYTORCH, PyTorchBackend)
