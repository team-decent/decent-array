"""Test checking whether dtypes that decent-array claims are available are actually available."""

from __future__ import annotations

from typing import Any

import pytest

from decent_array.types._dtypes import _ALL_DTYPES, dtype
from decent_array.types import Frameworks


def _sample_values(dt: dtype) -> list[Any]:
    if dt.name == "bool_":
        return [True, False, True]
    if dt.name in {"complex64", "complex128", "complex256"}:
        return [1 + 2j, 3 + 4j, 5 + 6j]
    if dt.name == "bytes_":
        return [b"a", b"bb", b"ccc"]
    if dt.name == "unicode_":
        return ["a", "bb", "ccc"]
    if dt.name == "void":
        return [b"ab", b"cd", b"ef"]
    return [1, 2, 3]


def _exercise_available_dtype(
    backend: tuple,
    dt: dtype,
) -> None:
    framework, _ = backend
    values = _sample_values(dt)

    assert dt.available
    assert dt.backend_dtype is not None

    if framework is Frameworks.NUMPY:
        import numpy as np

        x = np.asarray(values, dtype=dt.backend_dtype)
        if dt.name == "bool_":
            _ = np.logical_not(x)
        elif dt.name in {"bytes_", "unicode_", "void"}:
            assert x.dtype is not None
            assert x.shape == (3,)
        else:
            _ = x + x
        return

    if framework is Frameworks.PYTORCH:
        import torch

        if dt.name in {"qint8", "qint16", "qint32", "quint8", "quint16"}:
            assert isinstance(dt.backend_dtype, torch.dtype)
            return

        x = torch.tensor(values, dtype=dt.backend_dtype)
        if dt.name == "bool_":
            _ = torch.logical_not(x)
        else:
            _ = x + x
        return

    if framework is Frameworks.JAX:
        import jax.numpy as jnp

        x = jnp.asarray(values, dtype=dt.backend_dtype)
        if dt.name == "bool_":
            _ = jnp.logical_not(x)
        else:
            _ = x + x
        return

    if framework is Frameworks.TENSORFLOW:
        import tensorflow as tf

        x = tf.constant(values, dtype=dt.backend_dtype)
        if dt.name == "bool_":
            _ = tf.logical_not(x)
        elif dt.name == "bytes_":
            assert x.shape.rank == 1
            assert x.shape[0] == 3
        elif dt.name in {"qint8", "qint16", "qint32", "quint8", "quint16"}:
            assert x.shape == (3,)
            assert dt.backend_dtype.is_quantized
        else:
            _ = x + x
        return


@pytest.fixture
def available_dtypes_for_backend(
    backend: tuple,
) -> tuple[dtype, ...]:
    # set_backend has already run; dtypes are already bound to backend dtypes and marked as available/unavailable
    _ = backend
    return tuple(dt for dt in _ALL_DTYPES if dt.available)


@pytest.fixture
def available_dtype(
    request: pytest.FixtureRequest,
    backend: tuple,
    available_dtypes_for_backend: tuple[dtype, ...],
) -> dtype:
    framework, device = backend
    dt = request.param
    assert isinstance(dt, dtype)
    if dt not in available_dtypes_for_backend:
        pytest.skip(f"\t\t{dt.name} is marked unavailable on {framework.value}/{device.value}")
    return dt


def test_available_dtypes_have_backend_dtype(
    available_dtypes_for_backend: tuple[dtype, ...],
) -> None:
    for dt in available_dtypes_for_backend:
        assert dt.backend_dtype is not None


@pytest.mark.parametrize("available_dtype", _ALL_DTYPES, indirect=True, ids=lambda dt: dt.name)
def test_available_dtypes_actually_work(
    backend: tuple[Frameworks, Any],
    available_dtype: dtype,
) -> None:
    _exercise_available_dtype(backend, available_dtype)