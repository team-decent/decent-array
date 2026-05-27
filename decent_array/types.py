"""Type definitions for optimization variables."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, SupportsIndex, TypeAlias, Union

if TYPE_CHECKING:
    import jax
    import numpy
    import tensorflow as tf
    import torch

    from decent_array._array import Array

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
