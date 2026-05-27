"""Tests for :class:`decent_array.Array` operators, properties, and dunders."""

from __future__ import annotations

import re

import numpy as np
import pytest

import decent_array.interoperability as iop
from decent_array import Array
from decent_array.interoperability._backend_manager import reset_backends


def _np(arr: Array) -> np.ndarray:
    """Convert a backend-native ``Array`` to a numpy array for assertions."""
    return iop.to_numpy(arr)


def _create_array(data: float | list[float] | list[list[float]]) -> Array:
    """Create an Array from a list of floats for testing."""
    return iop.from_numpy(np.array(data, dtype=np.float32))


# Construction ------------------------------------------------------------


def test_init_requires_active_backend() -> None:
    reset_backends()
    with pytest.raises(RuntimeError, match=re.compile(r"No backend registered", re.IGNORECASE)):
        Array(np.zeros(3))


def test_init_records_active_backend(backend: tuple) -> None:
    arr = iop.zeros((3,))
    assert arr._backend is not None
    assert isinstance(arr, Array)


# Binary arithmetic -------------------------------------------------------


def test_add_array(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    b = _create_array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(_np(a + b), [5.0, 7.0, 9.0])


def test_add_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(_np(a + 2), [3.0, 4.0, 5.0])


def test_radd_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(_np(2 + a), [3.0, 4.0, 5.0])


def test_sub_array(backend: tuple) -> None:
    a = _create_array([4.0, 5.0, 6.0])
    b = _create_array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(_np(a - b), [3.0, 3.0, 3.0])


def test_sub_scalar(backend: tuple) -> None:
    a = _create_array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(_np(a - 1), [3.0, 4.0, 5.0])


def test_rsub_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(_np(10 - a), [9.0, 8.0, 7.0])


def test_mul_array(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    b = _create_array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(_np(a * b), [2.0, 6.0, 12.0])


def test_mul_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(_np(a * 3), [3.0, 6.0, 9.0])


def test_rmul_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(_np(3 * a), [3.0, 6.0, 9.0])


def test_truediv_array(backend: tuple) -> None:
    a = _create_array([4.0, 9.0, 16.0])
    b = _create_array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(_np(a / b), [2.0, 3.0, 4.0])


def test_truediv_scalar(backend: tuple) -> None:
    a = _create_array([2.0, 4.0, 6.0])
    np.testing.assert_allclose(_np(a / 2), [1.0, 2.0, 3.0])


def test_rtruediv_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(_np(8 / a), [8.0, 4.0, 2.0])


def test_matmul_array(backend: tuple) -> None:
    a = _create_array([[1.0, 2.0], [3.0, 4.0]])
    b = _create_array([[5.0, 6.0], [7.0, 8.0]])
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    np.testing.assert_allclose(_np(a @ b), expected)


def test_rmatmul_explicit_call(backend: tuple) -> None:
    # Python dispatch never picks Array.__rmatmul__ when both operands are Array,
    # so call it directly to exercise the code path.
    a = _create_array([[1.0, 2.0], [3.0, 4.0]])
    b = _create_array([[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_allclose(_np(a.__rmatmul__(b)), _np(b @ a))


def test_pow_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(_np(a**2), [1.0, 4.0, 9.0])


def test_pow_array(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    b = _create_array([3.0, 2.0, 1.0])
    np.testing.assert_allclose(_np(a**b), [1.0, 4.0, 3.0])


# Comparisons ------------------------------------------------------------


def test_eq_array(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    b = _create_array([1.0, 5.0, 3.0])
    result = a == b
    assert isinstance(result, Array)
    np.testing.assert_array_equal(_np(result), [True, False, True])


def test_eq_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(_np(a == 2.0), [False, True, False])


def test_ne_array(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    b = _create_array([1.0, 5.0, 3.0])
    np.testing.assert_array_equal(_np(a != b), [False, True, False])


def test_ne_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(_np(a != 2.0), [True, False, True])


def test_lt_array(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    b = _create_array([1.0, 5.0, 0.0])
    np.testing.assert_array_equal(_np(a < b), [False, True, False])


def test_lt_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(_np(a < 2.5), [True, True, False])


def test_le_array(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    b = _create_array([1.0, 5.0, 0.0])
    np.testing.assert_array_equal(_np(a <= b), [True, True, False])


def test_le_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(_np(a <= 2.0), [True, True, False])


def test_gt_array(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    b = _create_array([1.0, 5.0, 0.0])
    np.testing.assert_array_equal(_np(a > b), [False, False, True])


def test_gt_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(_np(a > 1.5), [False, True, True])


def test_ge_array(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    b = _create_array([1.0, 5.0, 0.0])
    np.testing.assert_array_equal(_np(a >= b), [True, False, True])


def test_ge_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(_np(a >= 2.0), [False, True, True])


def test_array_is_unhashable(backend: tuple) -> None:  # noqa: ARG001
    """
    Overriding ``__eq__`` makes Array unhashable, matching numpy/torch/jax/tf.

    The check is on ``hash(arr)`` rather than ``Array.__hash__ is None`` because
    mypyc realizes unhashability via a type-error-raising slot descriptor while
    pure Python sets the attribute to ``None``; ``hash()`` raises ``TypeError``
    in either case.
    """
    arr = _create_array([1.0])
    with pytest.raises(TypeError, match=re.compile(r"unhashable", re.IGNORECASE)):
        hash(arr)


# Bitwise ----------------------------------------------------------------


def test_and_array(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0, 4.0])
    mask1 = a > 1.0
    mask2 = a < 4.0
    np.testing.assert_array_equal(_np(mask1 & mask2), [False, True, True, False])


def test_and_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    mask = a > 1.0
    np.testing.assert_array_equal(_np(mask & True), [False, True, True])
    np.testing.assert_array_equal(_np(mask & False), [False, False, False])


def test_rand_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    mask = a > 1.0
    # Python evaluates ``True & mask`` via ``mask.__rand__(True)`` since bool's
    # ``__and__`` doesn't accept Array.
    np.testing.assert_array_equal(_np(True & mask), [False, True, True])


# In-place arithmetic -----------------------------------------------------


def test_iadd_array(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    b = _create_array([4.0, 5.0, 6.0])
    a += b
    np.testing.assert_allclose(_np(a), [5.0, 7.0, 9.0])


def test_iadd_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    a += 10
    np.testing.assert_allclose(_np(a), [11.0, 12.0, 13.0])


def test_isub_array(backend: tuple) -> None:
    a = _create_array([5.0, 6.0, 7.0])
    b = _create_array([1.0, 2.0, 3.0])
    a -= b
    np.testing.assert_allclose(_np(a), [4.0, 4.0, 4.0])


def test_isub_scalar(backend: tuple) -> None:
    a = _create_array([5.0, 6.0, 7.0])
    a -= 1
    np.testing.assert_allclose(_np(a), [4.0, 5.0, 6.0])


def test_imul_array(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    b = _create_array([4.0, 5.0, 6.0])
    a *= b
    np.testing.assert_allclose(_np(a), [4.0, 10.0, 18.0])


def test_imul_scalar(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    a *= 2
    np.testing.assert_allclose(_np(a), [2.0, 4.0, 6.0])


def test_itruediv_array(backend: tuple) -> None:
    a = _create_array([2.0, 6.0, 12.0])
    b = _create_array([2.0, 3.0, 4.0])
    a /= b
    np.testing.assert_allclose(_np(a), [1.0, 2.0, 3.0])


def test_itruediv_scalar(backend: tuple) -> None:
    a = _create_array([2.0, 4.0, 6.0])
    a /= 2
    np.testing.assert_allclose(_np(a), [1.0, 2.0, 3.0])


# Unary ------------------------------------------------------------------


def test_neg(backend: tuple) -> None:
    a = _create_array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(_np(-a), [-1.0, 2.0, -3.0])


def test_abs(backend: tuple) -> None:
    a = _create_array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(_np(abs(a)), [1.0, 2.0, 3.0])


# Indexing ---------------------------------------------------------------


def test_getitem_int(backend: tuple) -> None:
    a = _create_array([10.0, 20.0, 30.0])
    np.testing.assert_allclose(_np(a[1]), 20.0)


def test_getitem_slice(backend: tuple) -> None:
    a = _create_array([10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(_np(a[1:3]), [20.0, 30.0])


def test_getitem_tuple(backend: tuple) -> None:
    a = _create_array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(_np(a[1, 0]), 3.0)


def test_setitem_with_array(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    a[1] = _create_array(99.0)
    np.testing.assert_allclose(_np(a), [1.0, 99.0, 3.0])


def test_setitem_with_scalar(backend: tuple) -> None:
    # __setitem__ wraps a non-Array value in Array internally.
    a = _create_array([1.0, 2.0, 3.0])
    a[2] = 99.0
    np.testing.assert_allclose(_np(a), [1.0, 2.0, 99.0])


# Container / coercion / repr -------------------------------------------


def test_len(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0, 4.0])
    assert len(a) == 4


def test_float_coercion(backend: tuple) -> None:
    a = _create_array([2.5])
    assert float(a) == pytest.approx(2.5)


def test_repr(backend: tuple) -> None:
    a = _create_array([1.0, 2.0])
    text = repr(a)
    assert text.startswith("Array(")
    assert text.endswith(")")


def test_str(backend: tuple) -> None:
    a = _create_array([1.0, 2.0])
    # ``str(arr)`` delegates to the wrapped value's stringifier; just check it succeeds
    # and is non-empty rather than pin per-backend formatting.
    assert isinstance(str(a), str)
    assert len(str(a)) > 0


# Properties -------------------------------------------------------------


def test_shape(backend: tuple) -> None:
    a = iop.from_numpy(np.zeros((2, 3, 4), dtype=np.float32))
    assert a.shape == (2, 3, 4)


def test_size(backend: tuple) -> None:
    a = iop.from_numpy(np.zeros((2, 3, 4), dtype=np.float32))
    assert a.size == 24


def test_ndim(backend: tuple) -> None:
    a = iop.from_numpy(np.zeros((2, 3, 4), dtype=np.float32))
    assert a.ndim == 3


def test_transpose_property(backend: tuple) -> None:
    a = _create_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(_np(a.transpose), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])


def test_T_alias(backend: tuple) -> None:
    a = _create_array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(_np(a.T), _np(a.transpose))


def test_any_true(backend: tuple) -> None:
    a = _create_array([0.0, 0.0, 1.0])
    assert a.any is True


def test_any_false(backend: tuple) -> None:
    a = _create_array([0.0, 0.0, 0.0])
    assert a.any is False


def test_all_true(backend: tuple) -> None:
    a = _create_array([1.0, 2.0, 3.0])
    assert a.all is True


def test_all_false(backend: tuple) -> None:
    a = _create_array([1.0, 0.0, 3.0])
    assert a.all is False


def test_device_property(backend: tuple) -> None:
    _framework, device = backend
    a = iop.zeros((3,))
    assert a.device == device
