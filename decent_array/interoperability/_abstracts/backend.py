"""
Abstract :class:`Backend` contract.

All abstract methods live in this single class. The flat layout is mypyc-compatible:
when this module is included in the same compilation group as the concrete backends,
``_BACKEND.add(self, other)`` becomes a native compiled-to-compiled call (no Python
attribute lookup, no bound-method allocation), which removes the need for a ``raw_ops``
escape hatch on the hot path.

Section dividers in this file group related operations the way the legacy split files
did (creation, manipulation, linalg, math, operators, RNG); the only thing that
changed is that they all live on one class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from decent_array.types import Devices

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from decent_array import Array
    from decent_array.types import ArrayKey, DTypes, ArrayTypes


class Backend(ABC):  # noqa: PLR0904
    """
    Abstract base class for a backend.

    Concrete backends are bound to a single :class:`Devices` at construction
    time; that device is the default for all new arrays produced by this backend.
    """

    def __init__(self, device: Devices = Devices.CPU) -> None:
        self.device: Devices = device

    # Array creation ------------------------------------------------------

    @abstractmethod
    def zeros(self, shape: int | tuple[int, ...]) -> Array:
        """Create an array of zeros with the given shape."""

    @abstractmethod
    def zeros_like(self, x: Array) -> Array:
        """Create an array of zeros matching the shape and type of ``x``."""

    @abstractmethod
    def ones(self, shape: int | tuple[int, ...]) -> Array:
        """Create an array of ones with the given shape."""

    @abstractmethod
    def ones_like(self, x: Array) -> Array:
        """Create an array of ones matching the shape and type of ``x``."""

    @abstractmethod
    def eye(self, n: int) -> Array:
        """Create an ``n x n`` identity matrix."""

    @abstractmethod
    def device_to_native(self, device: Devices) -> Any:  # noqa: ANN401
        """Convert :class:`Devices` to the backend's native device representation."""

    @abstractmethod
    def device_of(self, x: Array) -> Devices:
        """Return the :class:`Devices` of the given array."""

    # Array manipulation --------------------------------------------------

    @abstractmethod
    def copy(self, x: Array) -> Array:
        """Return a copy of ``x``."""

    @abstractmethod
    def to_numpy(self, x: ArrayTypes | Array) -> NDArray[Any]:
        """Convert ``x`` to a NumPy array on the CPU."""

    @abstractmethod
    def from_numpy(self, x: NDArray[Any]) -> Array:
        """Convert a NumPy array on the CPU to an :class:`Array` on this backend."""

    @abstractmethod
    def from_numpy_like(self, x: NDArray[Any], like: Array) -> Array:
        """Convert a Numpy array to an :class:`Array` on this backend, matching shape and type of ``like``."""

    @abstractmethod
    def asarray(self, x: bool | int | float | complex) -> Array:
        """Convert a Python scalar to an :class:`Array` on this backend."""

    @abstractmethod
    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        """Stack a sequence of arrays along a new dimension."""

    @abstractmethod
    def reshape(self, x: Array, shape: tuple[int, ...]) -> Array:
        """Reshape ``x`` to ``shape``."""

    @abstractmethod
    def transpose(self, x: Array, axis: tuple[int, ...] | None = None) -> Array:
        """Transpose ``x``; ``None`` reverses the dimensions."""

    @abstractmethod
    def matrix_transpose(self, x: Array) -> Array:
        """Transpose the innermost two dimensions of ``x``."""

    @abstractmethod
    def shape(self, x: Array) -> tuple[int, ...]:
        """Return the shape of ``x``."""

    @abstractmethod
    def size(self, x: Array) -> int:
        """Return the total number of elements in ``x``."""

    @abstractmethod
    def ndim(self, x: Array) -> int:
        """Return the number of dimensions of ``x``."""

    @abstractmethod
    def squeeze(self, x: Array, axis: int | tuple[int, ...] | None = None) -> Array:
        """Remove single-dimensional entries from ``x``."""

    @abstractmethod
    def unsqueeze(self, x: Array, axis: int) -> Array:
        """Insert a singleton dimension at ``axis``."""

    @abstractmethod
    def diag(self, x: Array) -> Array:
        """Build a diagonal matrix from a 1-D vector."""

    @abstractmethod
    def diagonal(self, x: Array, offset: int = 0) -> Array:
        """Extract the diagonal entries from a 2-D matrix at the given ``offset``."""

    @abstractmethod
    def astype(self, x: Array, dtype: DTypes) -> Array:
        """Cast ``x`` to a different dtype."""

    # Linalg --------------------------------------------------------------

    @abstractmethod
    def vecdot(self, x1: Array, x2: Array) -> Array:
        """Dot product of two arrays."""

    @abstractmethod
    def matmul(self, x1: Array, x2: Array) -> Array:
        """Matrix multiplication of two arrays."""

    @abstractmethod
    def imatmul[T: Array](self, x1: T, x2: Array) -> T:
        """In-place matrix multiplication."""

    @abstractmethod
    def vector_norm(
        self,
        x: Array,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        ord: int | float = 2,  # noqa: A002
    ) -> Array:
        """Compute the norm of ``x``."""

    # Math reductions -----------------------------------------------------

    @abstractmethod
    def sum(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        """Sum elements of ``x`` along ``axis``."""

    @abstractmethod
    def mean(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        """Mean of ``x`` along ``axis``."""

    @abstractmethod
    def min(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        """Minimum of ``x`` along ``axis``."""

    @abstractmethod
    def max(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        """Maximum of ``x`` along ``axis``."""

    @abstractmethod
    def any(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:
        """Return True if any element of ``x`` is truthy."""

    @abstractmethod
    def all(self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> bool:
        """Return True if all elements of ``x`` are truthy."""

    # Math elementwise — both operands may be Array or scalar (operator dunders pass
    # either). ``Array | float`` covers both because PEP 484's numeric tower implicitly
    # admits ``int``.

    @abstractmethod
    def add(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        """Element-wise addition."""

    @abstractmethod
    def iadd[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        """In-place element-wise addition."""

    @abstractmethod
    def subtract(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        """Element-wise subtraction."""

    @abstractmethod
    def isubtract[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        """In-place element-wise subtraction."""

    @abstractmethod
    def multiply(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        """Element-wise multiplication."""

    @abstractmethod
    def imultiply[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        """In-place element-wise multiplication."""

    @abstractmethod
    def divide(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        """Element-wise division."""

    @abstractmethod
    def idivide[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        """In-place element-wise division."""

    @abstractmethod
    def floor_divide(self, x1: int | float | Array, x2: int | float | Array) -> Array:
        """Element-wise floor division."""

    @abstractmethod
    def ifloordiv[T: Array](self, x1: T, x2: int | float | Array) -> T:
        """In-place floor division."""

    @abstractmethod
    def remainder(self, x1: int | float | Array, x2: int | float | Array) -> Array:
        """Element-wise remainder after floor division."""

    @abstractmethod
    def imod[T: Array](self, x1: T, x2: int | float | Array) -> T:
        """In-place remainder after floor division."""

    @abstractmethod
    def pow(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        """Raise ``x1`` to power ``x2``."""

    @abstractmethod
    def ipow[T: Array](self, x1: T, x2: int | float | complex | Array) -> T:
        """In-place raise ``x1`` to power ``x2``."""

    @abstractmethod
    def negative(self, x: Array) -> Array:
        """Element-wise negation."""

    @abstractmethod
    def absolute(self, x: Array) -> Array:
        """Element-wise absolute value."""

    @abstractmethod
    def sqrt(self, x: Array) -> Array:
        """Element-wise square root."""

    # Comparisons — both operands may be Array or scalar.

    @abstractmethod
    def equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        """Element-wise equality. Returns an :class:`Array` of bools."""

    @abstractmethod
    def not_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        """Element-wise inequality. Returns an :class:`Array` of bools."""

    @abstractmethod
    def less(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        """Element-wise less-than. Returns an :class:`Array` of bools."""

    @abstractmethod
    def less_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        """Element-wise less-than-or-equal. Returns an :class:`Array` of bools."""

    @abstractmethod
    def greater(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        """Element-wise greater-than. Returns an :class:`Array` of bools."""

    @abstractmethod
    def greater_equal(self, x1: int | float | complex | Array, x2: int | float | complex | Array) -> Array:
        """Element-wise greater-than-or-equal. Returns an :class:`Array` of bools."""

    # Bitwise — operands may be int/bool arrays or scalars. Mirrors Python's ``&``,
    # which dispatches to ``logical_and`` on bool tensors and ``bitwise_and`` on int
    # tensors in every supported framework.

    @abstractmethod
    def bitwise_and(self, x1: bool | int | Array, x2: bool | int | Array) -> Array:
        """Element-wise bitwise/logical AND."""

    @abstractmethod
    def iand[T: Array](self, x1: T, x2: bool | int | Array) -> T:
        """In-place bitwise/logical AND."""

    @abstractmethod
    def bitwise_invert(self, x: Array) -> Array:
        """Element-wise bitwise/logical NOT."""

    @abstractmethod
    def bitwise_or(self, x1: bool | int | Array, x2: bool | int | Array) -> Array:
        """Element-wise bitwise/logical OR."""

    @abstractmethod
    def ior[T: Array](self, x1: T, x2: bool | int | Array) -> T:
        """In-place bitwise/logical OR."""

    @abstractmethod
    def bitwise_xor(self, x1: bool | int | Array, x2: bool | int | Array) -> Array:
        """Element-wise bitwise/logical XOR."""

    @abstractmethod
    def ixor[T: Array](self, x1: T, x2: bool | int | Array) -> T:
        """In-place bitwise/logical XOR."""

    @abstractmethod
    def bitwise_left_shift(self, x1: int | Array, x2: int | Array) -> Array:
        """Element-wise bitwise left shift."""

    @abstractmethod
    def ilshift[T: Array](self, x1: T, x2: int | Array) -> T:
        """In-place bitwise left shift."""

    @abstractmethod
    def bitwise_right_shift(self, x1: int | Array, x2: int | Array) -> Array:
        """Element-wise bitwise right shift."""

    @abstractmethod
    def irshift[T: Array](self, x1: T, x2: int | Array) -> T:
        """In-place bitwise right shift."""

    # Operators -----------------------------------------------------------

    @abstractmethod
    def sign(self, x: Array) -> Array:
        """Element-wise sign."""

    @abstractmethod
    def maximum(self, x1: int | float | Array, x2: int | float | Array) -> Array:
        """Element-wise maximum."""

    @abstractmethod
    def argmax(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        """Index of maximum value along ``axis``."""

    @abstractmethod
    def argmin(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        """Index of minimum value along ``axis``."""

    @abstractmethod
    def set_item(self, x: Array, key: ArrayKey, value: bool | int | float | complex | Array) -> None:
        """Set ``x[key] = value``."""

    @abstractmethod
    def get_item(self, x: Array, key: ArrayKey) -> Array:
        """Return ``x[key]``."""

    # RNG -----------------------------------------------------------------

    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """Seed the backend's RNG with ``seed``."""

    @abstractmethod
    def get_rng_state(self) -> dict[str, Any]:
        """Return a snapshot of the backend's RNG state."""

    @abstractmethod
    def set_rng_state(self, state: dict[str, Any]) -> None:
        """Restore an RNG snapshot produced by :meth:`get_rng_state`."""

    @abstractmethod
    def normal(self, mean: float = 0.0, std: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        """Draw normally distributed samples."""

    @abstractmethod
    def uniform(self, low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        """Draw uniformly distributed samples from ``[low, high)``."""

    @abstractmethod
    def normal_like(self, x: Array, mean: float = 0.0, std: float = 1.0) -> Array:
        """Draw normally distributed samples shaped like ``x`` with same dtype."""

    @abstractmethod
    def uniform_like(self, x: Array, low: float = 0.0, high: float = 1.0) -> Array:
        """Draw uniformly distributed samples shaped like ``x`` with same dtype."""

    @abstractmethod
    def choice(self, x: Array, size: int, replace: bool = True) -> Array:
        """Sample ``size`` elements from ``x``."""

    # DTYPES --------------------------------------------------------------

    @property
    @abstractmethod
    def bool(self) -> Any:  # noqa: ANN401
        """Bool dtype."""

    @property
    @abstractmethod
    def uint8(self) -> Any:  # noqa: ANN401
        """Unsigned 8-bit integer dtype."""

    @property
    def uint16(self) -> Any | None:  # noqa: ANN401
        """Unsigned 16-bit integer dtype (optional)."""
        return None

    @property
    def uint32(self) -> Any | None:  # noqa: ANN401
        """Unsigned 32-bit integer dtype (optional)."""
        return None

    @property
    def uint64(self) -> Any | None:  # noqa: ANN401
        """Unsigned 64-bit integer dtype (optional)."""
        return None

    @property
    @abstractmethod
    def int8(self) -> Any:  # noqa: ANN401
        """Signed 8-bit integer dtype."""

    @property
    @abstractmethod
    def int16(self) -> Any:  # noqa: ANN401
        """Signed 16-bit integer dtype."""

    @property
    @abstractmethod
    def int32(self) -> Any:  # noqa: ANN401
        """Signed 32-bit integer dtype."""

    @property
    @abstractmethod
    def int64(self) -> Any:  # noqa: ANN401
        """Signed 64-bit integer dtype."""

    @property
    @abstractmethod
    def float16(self) -> Any:  # noqa: ANN401
        """16-bit floating-point dtype."""

    @property
    @abstractmethod
    def float32(self) -> Any:  # noqa: ANN401
        """32-bit floating-point dtype."""

    @property
    @abstractmethod
    def float64(self) -> Any:  # noqa: ANN401
        """64-bit floating-point dtype."""

    @property
    @abstractmethod
    def complex64(self) -> Any:  # noqa: ANN401
        """64-bit complex dtype."""

    @property
    @abstractmethod
    def complex128(self) -> Any:  # noqa: ANN401
        """128-bit complex dtype."""

    @property
    def float128(self) -> Any | None:  # noqa: ANN401
        """128-bit floating-point dtype (optional)."""
        return None

    @property
    def complex256(self) -> Any | None:  # noqa: ANN401
        """256-bit complex dtype (optional)."""
        return None

    @property
    def qint8(self) -> Any | None:  # noqa: ANN401
        """Quantized 8-bit integer dtype (optional)."""
        return None

    @property
    def qint16(self) -> Any | None:  # noqa: ANN401
        """Quantized 16-bit integer dtype (optional)."""
        return None

    @property
    def qint32(self) -> Any | None:  # noqa: ANN401
        """Quantized 32-bit integer dtype (optional)."""
        return None

    @property
    def quint8(self) -> Any | None:  # noqa: ANN401
        """Quantized unsigned 8-bit integer dtype (optional)."""
        return None

    @property
    def quint16(self) -> Any | None:  # noqa: ANN401
        """Quantized unsigned 16-bit integer dtype (optional)."""
        return None

    @property
    def bfloat16(self) -> Any | None:  # noqa: ANN401
        """Brain floating point 16-bit dtype (optional)."""
        return None

    @property
    def unicode(self) -> Any | None:  # noqa: ANN401
        """Unicode string dtype (optional)."""
        return None

    @property
    def bytes(self) -> Any | None:  # noqa: ANN401
        """Byte string dtype (optional)."""
        return None

    @property
    def object(self) -> Any | None:  # noqa: ANN401
        """Python object dtype (optional)."""
        return None

    @property
    def void(self) -> Any | None:  # noqa: ANN401
        """Raw/void dtype (optional)."""
        return None
