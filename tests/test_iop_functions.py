"""Tests for the module-level functions in :mod:`decent_array.interoperability.iop.functions`."""

from __future__ import annotations

import numpy as np
import pytest

import decent_array.interoperability as iop
from decent_array import Array
from decent_array.interoperability._backend_manager import reset_backends
from decent_array.interoperability._iop.math import iadd, idivide, imultiply, isubtract
from decent_array.interoperability._iop.utils import device_to_native, get_item, set_item
from decent_array.types import DTypes


def _np(arr: Array) -> np.ndarray:
    return iop.to_numpy(arr)


# Array creation ---------------------------------------------------------


def test_zeros(backend: tuple) -> None:
    arr = iop.zeros((2, 3))
    np.testing.assert_allclose(_np(arr), np.zeros((2, 3)))


def test_zeros_like(backend: tuple) -> None:
    src = iop.from_numpy(np.ones((2, 3), dtype=np.float32))
    arr = iop.zeros_like(src)
    np.testing.assert_allclose(_np(arr), np.zeros((2, 3)))


def test_ones(backend: tuple) -> None:
    arr = iop.ones((2, 3))
    np.testing.assert_allclose(_np(arr), np.ones((2, 3)))


def test_ones_like(backend: tuple) -> None:
    src = iop.from_numpy(np.zeros((2, 3), dtype=np.float32))
    arr = iop.ones_like(src)
    np.testing.assert_allclose(_np(arr), np.ones((2, 3)))


def test_eye(backend: tuple) -> None:
    arr = iop.eye(3)
    np.testing.assert_allclose(_np(arr), np.eye(3))


def test_device_to_native(backend: tuple) -> None:
    framework, device = backend
    native = device_to_native(device)
    # Just verify the call succeeds and returns a non-None value (varies per backend).
    assert native is not None


def test_device_of(backend: tuple) -> None:
    _framework, device = backend
    arr = iop.zeros((3,))
    assert iop.device_of(arr) == device


# Array manipulation -----------------------------------------------------


def test_copy_independent(backend: tuple) -> None:
    src = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    dst = iop.copy(src)
    np.testing.assert_allclose(_np(dst), [1.0, 2.0, 3.0])
    # Mutating the copy shouldn't affect the original.
    dst[0] = 99.0
    np.testing.assert_allclose(_np(src), [1.0, 2.0, 3.0])


def test_to_numpy(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    out = iop.to_numpy(arr)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_from_numpy_roundtrip(backend: tuple) -> None:
    raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr = iop.from_numpy(raw)
    np.testing.assert_allclose(_np(arr), raw)


def test_from_numpy_like_matches_dtype(backend: tuple) -> None:
    # ``like`` is int32 so the resulting Array should be int32 even though the
    # source numpy array is float32. (Avoids float64, which torch MPS rejects.)
    like = iop.from_numpy(np.array([0, 0], dtype=np.int32))
    raw = np.array([1.5, 2.5], dtype=np.float32)
    arr = iop.from_numpy_like(raw, like)
    out_np = _np(arr)
    assert out_np.dtype == np.int32
    np.testing.assert_array_equal(out_np, [1, 2])


def test_from_numpy_like_matches_device(backend: tuple) -> None:
    # ``like`` lives on whichever device the active backend uses; the result of
    # from_numpy_like must report the same device.
    like = iop.from_numpy(np.array([0.0, 0.0], dtype=np.float32))
    raw = np.array([3.0, 4.0], dtype=np.float32)
    arr = iop.from_numpy_like(raw, like)
    assert iop.device_of(arr) == iop.device_of(like)


def test_to_array_from_scalar(backend: tuple) -> None:
    arr = iop.asarray(2.5)
    np.testing.assert_allclose(_np(arr), 2.5)


def test_stack(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    b = iop.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
    arr = iop.stack([a, b])
    np.testing.assert_allclose(_np(arr), [[1.0, 2.0], [3.0, 4.0]])


def test_stack_dim(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    b = iop.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
    arr = iop.stack([a, b], axis=1)
    np.testing.assert_allclose(_np(arr), [[1.0, 3.0], [2.0, 4.0]])


def test_stack_empty_raises(backend: tuple) -> None:
    with pytest.raises(ValueError, match=r"empty sequence"):
        iop.stack([])


def test_reshape(backend: tuple) -> None:
    arr = iop.from_numpy(np.arange(6, dtype=np.float32))
    out = iop.reshape(arr, (2, 3))
    np.testing.assert_allclose(_np(out), np.arange(6, dtype=np.float32).reshape(2, 3))


def test_transpose_default(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.transpose(arr)), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])


def test_transpose_explicit_dim(backend: tuple) -> None:
    arr = iop.from_numpy(np.zeros((2, 3, 4), dtype=np.float32))
    out = iop.transpose(arr, axis=(1, 0, 2))
    assert iop.shape(out) == (3, 2, 4)


def test_shape_function(backend: tuple) -> None:
    arr = iop.from_numpy(np.zeros((2, 3, 4), dtype=np.float32))
    assert iop.shape(arr) == (2, 3, 4)


def test_size_function(backend: tuple) -> None:
    arr = iop.from_numpy(np.zeros((2, 3, 4), dtype=np.float32))
    assert iop.size(arr) == 24


def test_ndim_function(backend: tuple) -> None:
    arr = iop.from_numpy(np.zeros((2, 3, 4), dtype=np.float32))
    assert iop.ndim(arr) == 3


def test_squeeze_default(backend: tuple) -> None:
    arr = iop.from_numpy(np.zeros((1, 3, 1), dtype=np.float32))
    assert iop.shape(iop.squeeze(arr)) == (3,)


def test_squeeze_specific_dim(backend: tuple) -> None:
    arr = iop.from_numpy(np.zeros((1, 3, 1), dtype=np.float32))
    assert iop.shape(iop.squeeze(arr, axis=0)) == (3, 1)


def test_expand_dims(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    out = iop.expand_dims(arr, axis=0)
    assert iop.shape(out) == (1, 3)


def test_unsqueeze(backend: tuple) -> None:
    arr = iop.from_numpy(np.zeros((3,), dtype=np.float32))
    assert iop.shape(iop.unsqueeze(arr, axis=0)) == (1, 3)
    assert iop.shape(iop.unsqueeze(arr, axis=1)) == (3, 1)


def test_diag_from_vector(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.diag(arr)), np.diag([1.0, 2.0, 3.0]))


def test_diagonal_from_matrix(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.diagonal(arr)), [1.0, 4.0])


def test_diagonal_with_offset(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.diagonal(arr, offset=1)), [2.0])
    np.testing.assert_allclose(_np(iop.diagonal(arr, offset=-1)), [3.0])


def test_astype_to_float(backend: tuple) -> None:
    arr = iop.asarray(3.0)
    out = iop.astype(arr, DTypes.FLOAT32)
    np_out = _np(out)
    assert np_out.dtype == np.float32
    assert np_out == pytest.approx(3.0)


def test_astype_to_int(backend: tuple) -> None:
    arr = iop.asarray(3.0)
    out = iop.astype(arr, DTypes.INT32)
    np_out = _np(out)
    assert np_out.dtype == np.int32
    assert int(np_out) == 3


def test_astype_to_bool(backend: tuple) -> None:
    arr = iop.asarray(1.0)
    out = iop.astype(arr, DTypes.BOOL)
    np_out = _np(out)
    assert np_out.dtype == np.bool_
    assert bool(np_out) is True


# Linalg -----------------------------------------------------------------


def test_dot(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = iop.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.dot(a, b)), 32.0)


def test_vecdot(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = iop.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.vecdot(a, b)), 32.0)


def test_matmul(backend: tuple) -> None:
    a = iop.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    b = iop.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
    expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
    np.testing.assert_allclose(_np(iop.matmul(a, b)), expected)


def test_norm_default_l2(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.vector_norm(arr)), 5.0)


def test_norm_p1(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([3.0, -4.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.vector_norm(arr, ord=1)), 7.0)


def test_norm_dim_keepdims(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([[3.0, 4.0], [6.0, 8.0]], dtype=np.float32))
    out = iop.vector_norm(arr, ord=2, axis=1, keepdims=True)
    assert iop.shape(out) == (2, 1)
    np.testing.assert_allclose(_np(out).reshape(-1), [5.0, 10.0])


# Math reductions --------------------------------------------------------


def test_sum_all(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.sum(arr)), 10.0)


def test_sum_dim(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.sum(arr, axis=0)), [4.0, 6.0])


def test_sum_dim_keepdims(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    out = iop.sum(arr, axis=0, keepdims=True)
    assert iop.shape(out) == (1, 2)


def test_mean_all(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.mean(arr)), 2.5)


def test_mean_dim(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.mean(arr, axis=1)), [1.5, 3.5])


def test_min_all(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.min(arr)), 1.0)


def test_min_dim(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.min(arr, axis=0)), [1.0, 2.0])


def test_max_all(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.max(arr)), 5.0)


def test_max_dim(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.max(arr, axis=0)), [3.0, 5.0])


def test_any_true(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert iop.any(arr) is True


def test_any_false(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert iop.any(arr) is False


def test_all_true(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert iop.all(arr) is True


def test_all_false(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([1.0, 0.0, 3.0], dtype=np.float32))
    assert iop.all(arr) is False


# Math elementwise -------------------------------------------------------


def test_add_two_arrays(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    b = iop.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.add(a, b)), [4.0, 6.0])


def test_add_array_and_scalar(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.add(a, 10.0)), [11.0, 12.0])


def test_sub(backend: tuple) -> None:
    a = iop.from_numpy(np.array([5.0, 6.0], dtype=np.float32))
    b = iop.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.subtract(a, b)), [4.0, 4.0])


def test_mul(backend: tuple) -> None:
    a = iop.from_numpy(np.array([2.0, 3.0], dtype=np.float32))
    b = iop.from_numpy(np.array([4.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.multiply(a, b)), [8.0, 15.0])


def test_div(backend: tuple) -> None:
    a = iop.from_numpy(np.array([8.0, 10.0], dtype=np.float32))
    b = iop.from_numpy(np.array([2.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.divide(a, b)), [4.0, 2.0])


def test_iadd_func(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    out = iadd(a, 10.0)
    # Returned wrapper is the same instance.
    assert out is a
    np.testing.assert_allclose(_np(a), [11.0, 12.0])


def test_isub_func(backend: tuple) -> None:
    a = iop.from_numpy(np.array([5.0, 6.0], dtype=np.float32))
    out = isubtract(a, 1.0)
    assert out is a
    np.testing.assert_allclose(_np(a), [4.0, 5.0])


def test_imul_func(backend: tuple) -> None:
    a = iop.from_numpy(np.array([2.0, 3.0], dtype=np.float32))
    out = imultiply(a, 4.0)
    assert out is a
    np.testing.assert_allclose(_np(a), [8.0, 12.0])


def test_idiv_func(backend: tuple) -> None:
    a = iop.from_numpy(np.array([8.0, 12.0], dtype=np.float32))
    out = idivide(a, 4.0)
    assert out is a
    np.testing.assert_allclose(_np(a), [2.0, 3.0])


def test_pow_function(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([2.0, 3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.pow(arr, 2)), [4.0, 9.0, 16.0])
    arr2 = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.pow(arr, arr2)), [2.0, 9.0, 64.0])


def test_negative(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([1.0, -2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.negative(arr)), [-1.0, 2.0, -3.0])


def test_abs(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([-1.0, 2.0, -3.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.abs(arr)), [1.0, 2.0, 3.0])


def test_absolute(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([1.0, -2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.absolute(arr)), [1.0, 2.0, 3.0])


def test_sqrt(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([1.0, 4.0, 9.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.sqrt(arr)), [1.0, 2.0, 3.0])


# Operators --------------------------------------------------------------


def test_sign(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([-2.0, 0.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.sign(arr)), [-1.0, 0.0, 1.0])


def test_maximum_arrays(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 5.0, 3.0], dtype=np.float32))
    b = iop.from_numpy(np.array([4.0, 2.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.maximum(a, b)), [4.0, 5.0, 6.0])


def test_maximum_array_and_scalar(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 5.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.maximum(a, 4.0)), [4.0, 5.0, 4.0])


# Comparisons ------------------------------------------------------------


def test_eq_arrays(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = iop.from_numpy(np.array([1.0, 5.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(_np(iop.equal(a, b)), [True, False, True])


def test_eq_array_and_scalar(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(_np(iop.equal(a, 2.0)), [False, True, False])


def test_ne_arrays(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = iop.from_numpy(np.array([1.0, 5.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(_np(iop.not_equal(a, b)), [False, True, False])


def test_ne_array_and_scalar(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(_np(iop.not_equal(a, 2.0)), [True, False, True])


def test_lt_arrays(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = iop.from_numpy(np.array([2.0, 2.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(_np(iop.less(a, b)), [True, False, False])


def test_lt_array_and_scalar(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(_np(iop.less(a, 2.5)), [True, True, False])


def test_le_arrays(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = iop.from_numpy(np.array([2.0, 2.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(_np(iop.less_equal(a, b)), [True, True, False])


def test_le_array_and_scalar(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(_np(iop.less_equal(a, 2.0)), [True, True, False])


def test_gt_arrays(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = iop.from_numpy(np.array([2.0, 2.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(_np(iop.greater(a, b)), [False, False, True])


def test_gt_array_and_scalar(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(_np(iop.greater(a, 1.5)), [False, True, True])


def test_ge_arrays(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = iop.from_numpy(np.array([2.0, 2.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(_np(iop.greater_equal(a, b)), [False, True, True])


def test_ge_array_and_scalar(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(_np(iop.greater_equal(a, 2.0)), [False, True, True])


# Bitwise ----------------------------------------------------------------


def test_bitwise_and_bool_arrays(backend: tuple) -> None:
    a = iop.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    mask1 = iop.greater(a, 1.0)
    mask2 = iop.less(a, 4.0)
    np.testing.assert_array_equal(_np(iop.bitwise_and(mask1, mask2)), [False, True, True, False])


def test_bitwise_and_int_arrays(backend: tuple) -> None:
    # Bitwise on int dtypes is well-defined across all backends; bool tensors are
    # tested separately because TF dispatches them to ``logical_and``.
    a = iop.from_numpy(np.array([0b1100, 0b1010], dtype=np.int32))
    b = iop.from_numpy(np.array([0b1010, 0b0110], dtype=np.int32))
    np.testing.assert_array_equal(_np(iop.bitwise_and(a, b)), [0b1000, 0b0010])


def test_argmax_default(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.argmax(arr)), 5)


def test_argmax_dim(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([[1.0, 5.0, 2.0], [4.0, 0.0, 3.0]], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.argmax(arr, axis=1)), [1, 0])


def test_argmin_default(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0], dtype=np.float32))
    # First occurrence of minimum is index 1.
    np.testing.assert_allclose(_np(iop.argmin(arr)), 1)


def test_argmin_dim(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([[1.0, 5.0, 2.0], [4.0, 0.0, 3.0]], dtype=np.float32))
    np.testing.assert_allclose(_np(iop.argmin(arr, axis=1)), [0, 1])


def test_argmax_keepdims(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([[1.0, 5.0, 2.0], [4.0, 0.0, 3.0]], dtype=np.float32))
    out = iop.argmax(arr, axis=1, keepdims=True)
    assert iop.shape(out) == (2, 1)


def test_set_item_function(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    set_item(arr, 0, iop.from_numpy(np.array(99.0, dtype=np.float32)))
    np.testing.assert_allclose(_np(arr), [99.0, 2.0, 3.0])


def test_get_item_function(backend: tuple) -> None:
    arr = iop.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(_np(get_item(arr, 1)), 2.0)


# No-backend errors ------------------------------------------------------


def test_function_raises_when_no_backend() -> None:
    reset_backends()
    with pytest.raises(RuntimeError, match=r"No backend active"):
        iop.zeros((3,))


def test_to_array_round_trip_with_bool(backend: tuple) -> None:
    arr = iop.asarray(True)
    out = iop.astype(arr, DTypes.BOOL)
    np_out = _np(out)
    assert np_out.dtype == np.bool_
    assert bool(np_out) is True
