"""
Interoperability layer.

Typical usage::

    import decent_array.interoperability as iop

    iop.set_backend("numpy")
    a = iop.zeros((3, 3))
    iop.set_seed(42)  # Optional: set RNG seed for reproducibility
    s = iop.normal(shape=(2,))

"""

from ._backend_manager import default_device, set_backend
from ._iop.bit_operators import bitwise_and
from ._iop.comparasion import equal, greater, greater_equal, less, less_equal, not_equal
from ._iop.creation import eye, ones, ones_like, zeros, zeros_like
from ._iop.linalg import dot, matmul, norm, vecdot, vector_norm
from ._iop.manipulations import (
    asarray,
    astype,
    copy,
    diag,
    diagonal,
    expand_dims,
    from_numpy,
    from_numpy_like,
    ndim,
    reshape,
    shape,
    size,
    squeeze,
    stack,
    to_numpy,
    transpose,
    unsqueeze,
)
from ._iop.math import abs, absolute, add, divide, multiply, negative, pow, sqrt, subtract  # noqa: A004
from ._iop.operators import argmax, argmin, maximum, sign
from ._iop.reductions import all, any, max, mean, min, sum  # noqa: A004
from ._iop.rng import (
    choice,
    derive_seed,
    get_numpy_rng,
    get_rng_state,
    get_seed,
    normal,
    normal_like,
    set_rng_state,
    set_seed,
    uniform,
    uniform_like,
)
from ._iop.utils import device_of

__all__ = [
    "abs",
    "absolute",
    "add",
    "all",
    "any",
    "argmax",
    "argmin",
    "asarray",
    "astype",
    "bitwise_and",
    "choice",
    "copy",
    "default_device",
    "derive_seed",
    "device_of",
    "diag",
    "diagonal",
    "divide",
    "dot",
    "equal",
    "expand_dims",
    "eye",
    "from_numpy",
    "from_numpy_like",
    "get_numpy_rng",
    "get_rng_state",
    "get_seed",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "matmul",
    "max",
    "maximum",
    "mean",
    "min",
    "multiply",
    "ndim",
    "negative",
    "norm",
    "normal",
    "normal_like",
    "not_equal",
    "ones",
    "ones_like",
    "pow",
    "reshape",
    "set_backend",
    "set_rng_state",
    "set_seed",
    "shape",
    "sign",
    "size",
    "sqrt",
    "squeeze",
    "stack",
    "subtract",
    "sum",
    "to_numpy",
    "transpose",
    "uniform",
    "uniform_like",
    "unsqueeze",
    "vecdot",
    "vector_norm",
    "zeros",
    "zeros_like",
]
