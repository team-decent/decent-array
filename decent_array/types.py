"""Type definitions for optimization variables."""

from __future__ import annotations

import sys
from enum import Enum
from typing import TYPE_CHECKING, SupportsIndex, TypeAlias, Union

from decent_array.interoperability._backend_manager import register_backend_listener

if TYPE_CHECKING:
    import jax
    import numpy
    import tensorflow as tf
    import torch

    from decent_array._array import Array
    from decent_array.interoperability._abstracts import Backend


_BACKEND_INSTANCE: Backend | None = None
_error = RuntimeError("No backend active: call 'set_backend' with a supported framework to activate one.")


def _update_backend(backend: Backend | None) -> None:
    global _BACKEND_INSTANCE  # noqa: PLW0603
    _BACKEND_INSTANCE = backend

    _reset_dtype_registry()

    if backend is not None:
        _initialize_dtype_registry(backend)


def _reset_dtype_registry() -> None:
    """Remove dtype module attributes and clear the native reverse map."""
    module = sys.modules[__name__]

    _NATIVE_TO_IOP.clear()
    for name in _SUPPORTED_DTYPES + _OPTIONAL_DTYPES:
        if hasattr(module, name):
            delattr(module, name)


register_backend_listener(_update_backend)


ArrayLike: TypeAlias = Union["numpy.ndarray", "torch.Tensor", "tf.Tensor", "jax.Array"]  # noqa: UP040
"""
Type alias for array-like types supported in decent-array, including NumPy arrays,
PyTorch tensors, TensorFlow tensors, and JAX arrays.
"""

SupportedArrayTypes: TypeAlias = bool | int | float | complex | ArrayLike  # noqa: UP040
"""
Type alias for supported types for optimization variables in decent-array,
including array-like types and scalars.
"""

ArrayKey: TypeAlias = (  # noqa: UP040
    "int | SupportsIndex | slice | Array | tuple[int | SupportsIndex | slice | Array | None, ...] | None"
)
"""
Type alias for valid keys used to index into supported array types.
Includes single indices, tuples of indices, slices, and tuples of slices.
"""


# Its important that the enum values correspond to the folder names of the backends,
# since those are used for dynamic imports in _backend_manager.py
class SupportedFrameworks(Enum):
    """Enum for supported frameworks in decent-array."""

    NUMPY = "numpy"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"


class SupportedDevices(Enum):
    """Enum for supported devices in decent-array."""

    CPU = "cpu"
    GPU = "gpu"
    MPS = "mps"


class DTypes(Enum):
    """Enum for supported dtypes in decent-array."""

    BOOL = "bool"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"


_STRING_TO_DTYPE = {dt.value: dt for dt in DTypes}


class dtype:  # noqa: N801
    """Base class for dtypes."""

    def __init__(self, name: str):
        if _BACKEND_INSTANCE is None:
            raise _error

        # name doesn't map to any supported/optional dtype
        if name not in (_SUPPORTED_DTYPES + _OPTIONAL_DTYPES):
            raise ValueError(f"dtype {name} is not supported.")

        # native_dtype is None if the dtype is not supported
        native_dtype = getattr(_BACKEND_INSTANCE, name, None)
        if native_dtype is None:
            raise ValueError(f"dtype {name} is not supported.")  # TODO add device and backend name to contextualize

        self._name = name
        self._native_dtype = native_dtype  # TODO better name?

    @property
    def name(self) -> str:
        """Name of the dtype."""
        return self._name

    def __str__(self) -> str:
        """Name of the dtype."""
        return self.name

    def __eq__(self, other: object) -> bool:
        """Check equivalence by ``name`` attributes."""
        if not isinstance(other, dtype):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        """Hash of the dtype."""
        return hash(self.name)


_SUPPORTED_DTYPES = [
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]

_OPTIONAL_DTYPES = [
    "uint16",
    "uint32",
    "uint64",
    "float128",
    "complex256",
    "qint8",
    "qint16",
    "qint32",
    "quint8",
    "quint16",
    "bfloat16",
    "unicode",
    "bytes",
    "object",
    "void",
]


# TODO jax needs jax.config.x64_enabled=True to use 64-bit; if not, it will silently downcast to 32-bit
# right now I return None if jax.config.x64_enabled=False, because it is effectively unavailable. do we want to do this
# or let it silently downcast?

# TODO do we want to have an optional dependence on ml_dtypes to support bfloat16 for numpy?

# ✓: available, ~: limited support, ✗: not available
# dtype                      | NumPy | JAX       | PyTorch    | TensorFlow | Notes  # noqa: ERA001
# ---------------------------|-------|-----------|------------|------------|------------------------------------------
# uint16                     | ✓     | ✓         | ~          | ✓          | https://github.com/pytorch/pytorch/issues/58734
# uint32                     | ✓     | ✓         | ~          | ✓          | https://github.com/pytorch/pytorch/issues/58734
# uint64                     | ✓     | ✓         | ~          | ✓          | https://github.com/pytorch/pytorch/issues/58734

# float128                   | ✓     | ✗         | ✗          | ✗          | platform-dependent
# complex256                 | ✓     | ✗         | ✗          | ✗          | platform-dependent

# qint8                      | ✗     | ✗         | ✓          | ✓          |
# qint16                     | ✗     | ✗         | ✗          | ✓          |
# qint32                     | ✗     | ✗         | ✓          | ✓          |

# quint8                     | ✗     | ✗         | ✓          | ✓          |
# quint16                    | ✗     | ✗         | ✗          | ✓          |

# bfloat16                   | ✗*    | ✓         | ✓          | ✓          | * available with ml_dtypes dependency

# unicode_                   | ✓     | ✗         | ✗          | ✗          | np.str_
# bytes_                     | ✓     | ✗         | ✗          | ✓          | np.bytes_, tf.string
# object                     | ✓     | ✗         | ✗          | ✗          |
# void                       | ✓     | ✗         | ✗          | ✗          |


# dict for reverse search native dtype -> iop dtype
_NATIVE_TO_IOP: dict[object, dtype] = {}


def _initialize_dtype_registry(backend: Backend) -> None:
    """Initialize dtype objects and the reverse native-to-iop map for ``backend``."""
    global _NATIVE_TO_IOP  # noqa: PLW0603
    module = sys.modules[__name__]
    _NATIVE_TO_IOP = {}

    for name in (_SUPPORTED_DTYPES + _OPTIONAL_DTYPES):
        _register_dtype(module, backend, name)


def _register_dtype(module: object, backend: Backend, name: str) -> None:
    """Create and publish a dtype (if available)."""
    native_dtype = getattr(backend, name, None)
    if native_dtype is None:
        return

    iop_dtype = dtype(name)
    setattr(module, name, iop_dtype)
    _NATIVE_TO_IOP[native_dtype] = iop_dtype
