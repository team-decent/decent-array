"""Tests for dtype initialization and backend binding."""

from __future__ import annotations

import math
import pytest

from decent_array.interoperability import _backend_manager
from decent_array.interoperability._backend_manager import reset_backends
from decent_array.types import _dtypes
import decent_array._constants as constants


@pytest.fixture(autouse=True)
def reset_backend_state():
    """Reset backend state before and after each test."""
    reset_backends()
    yield
    reset_backends()


def test_all_dtypes_are_bound_to_backend(backend: tuple) -> None:
    """Every dtype is bound to the corresponding backend attribute."""
    backend_instance = _backend_manager._BACKEND_INSTANCE

    assert backend_instance is not None

    for name in _dtypes._SUPPORTED:
        dt = getattr(_dtypes, name)
        expected_backend_dtype = getattr(backend_instance, name, None)

        assert dt.backend_dtype is expected_backend_dtype
        assert dt.available is (expected_backend_dtype is not None)


def test_available_dtypes_cache_is_correct(backend: tuple) -> None:
    expected = {
        dt for dt in _dtypes._ALL_DTYPES if dt.available
    }

    assert _dtypes._AVAILABLE_DTYPES == expected


def test_dtypes_returns_only_available_dtypes(backend: tuple) -> None:
    available = _dtypes.dtypes()

    assert available
    assert all(dt.available for dt in available.values())
    assert set(available.values()) == _dtypes._AVAILABLE_DTYPES


def test_backend_dtype_mapping_is_correct(backend: tuple) -> None:
    expected = {
        dt.backend_dtype: dt
        for dt in _dtypes._AVAILABLE_DTYPES
    }

    assert _dtypes._BACKEND_DTYPE_TO_DTYPE == expected

    for dt in _dtypes._AVAILABLE_DTYPES:
        assert (
            _dtypes._BACKEND_DTYPE_TO_DTYPE[dt.backend_dtype]
            is dt
        )


def test_reset_clears_dtype_bindings(backend: tuple) -> None:
    """Resetting backends clears dtype bindings and caches."""
    assert _backend_manager._BACKEND_INSTANCE is not None
    assert _dtypes._AVAILABLE_DTYPES

    reset_backends()

    assert _backend_manager._BACKEND_INSTANCE is None
    assert not _dtypes._AVAILABLE_DTYPES
    assert not _dtypes._BACKEND_DTYPE_TO_DTYPE

    for dt in _dtypes._ALL_DTYPES:
        assert dt.available is False
        assert dt.backend_dtype is None


def test_reset_restores_default_constants(backend: tuple) -> None:
    """Resetting the backend restores Python math defaults."""
    assert _backend_manager._BACKEND_INSTANCE is not None

    reset_backends()

    assert constants.e == math.e
    assert constants.inf == math.inf
    assert math.isnan(constants.nan)
    assert constants.pi == math.pi
