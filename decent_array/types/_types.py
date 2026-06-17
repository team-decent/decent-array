"""Type definitions."""

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

ArrayTypes: TypeAlias = bool | int | float | complex | ArrayLike  # noqa: UP040
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
class Frameworks(Enum):
    """Enum for supported frameworks in decent-array."""

    NUMPY = "numpy"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"


class Devices(Enum):
    """Enum for supported devices in decent-array."""

    CPU = "cpu"
    GPU = "gpu"
    MPS = "mps"
