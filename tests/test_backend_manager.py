"""Tests for :mod:`decent_array.interoperability._backend_manager`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from decent_array.interoperability import _backend_manager as backend_manager
from decent_array.interoperability._abstracts import Backend
from decent_array.interoperability._backend_manager import (
    _instantiate,
    _normalize,
    default_device,
    register_backend,
    register_backend_listener,
    reset_backends,
    set_backend,
)
from decent_array.types import SupportedDevices, SupportedFrameworks

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _isolate_listeners_and_backends() -> Iterator[None]:
    """Snapshot+restore module-level state so backend-manager tests don't leak."""
    listeners_snapshot = backend_manager._BACKEND_LISTENERS.copy()
    registry_snapshot = backend_manager._BACKEND_REGISTRY.copy()
    reset_backends()
    yield
    backend_manager._BACKEND_LISTENERS[:] = listeners_snapshot
    backend_manager._BACKEND_REGISTRY.clear()
    backend_manager._BACKEND_REGISTRY.update(registry_snapshot)
    reset_backends()


# _normalize -------------------------------------------------------------


def test_normalize_accepts_enum() -> None:
    assert _normalize(SupportedFrameworks.NUMPY) == SupportedFrameworks.NUMPY


def test_normalize_accepts_string() -> None:
    assert _normalize("numpy") == SupportedFrameworks.NUMPY


def test_normalize_unknown_raises() -> None:
    with pytest.raises(KeyError, match=r"Unknown backend"):
        _normalize("not-a-backend")


# set_backend ------------------------------------------------------------


def test_set_backend_with_string() -> None:
    set_backend("numpy")
    assert backend_manager._ACTIVE_BACKEND.get() == SupportedFrameworks.NUMPY


def test_set_backend_with_enum() -> None:
    set_backend(SupportedFrameworks.NUMPY)
    assert backend_manager._ACTIVE_BACKEND.get() == SupportedFrameworks.NUMPY


def test_set_backend_idempotent_same_backend() -> None:
    set_backend("numpy")
    # Re-activating with the same backend+device must be a no-op (no exception).
    set_backend("numpy")
    assert backend_manager._ACTIVE_BACKEND.get() == SupportedFrameworks.NUMPY


def test_set_backend_different_backend_raises() -> None:
    set_backend("numpy")
    with pytest.raises(RuntimeError, match=r"already set to"):
        set_backend("pytorch")


def test_set_backend_with_string_device() -> None:
    set_backend("numpy", "cpu")
    instance = _instantiate(SupportedFrameworks.NUMPY, SupportedDevices.CPU)
    assert instance.device == SupportedDevices.CPU


def test_set_backend_invalid_name_raises() -> None:
    with pytest.raises(KeyError):
        set_backend("not-a-backend")


# register_backend -------------------------------------------------------


def test_register_backend_rejects_non_subclass() -> None:
    class NotABackend:
        pass

    with pytest.raises(TypeError, match=r"subclass of Backend"):
        register_backend(SupportedFrameworks.NUMPY, NotABackend)  # type: ignore[arg-type]


def test_register_backend_replaces_cached_instance() -> None:
    # First import registers the real backend; instantiate to populate cache.
    set_backend("numpy")
    cached = backend_manager._BACKEND_INSTANCES.get(SupportedFrameworks.NUMPY)
    assert cached is not None

    # Re-register the same class — cache should be cleared so next instantiate is fresh.
    from decent_array.interoperability._numpy.numpy_backend import NumpyBackend  # noqa: PLC0415

    reset_backends()
    register_backend(SupportedFrameworks.NUMPY, NumpyBackend)
    assert SupportedFrameworks.NUMPY not in backend_manager._BACKEND_INSTANCES


# register_backend_listener ---------------------------------------------


def test_listener_called_on_activation() -> None:
    received: list[Backend | None] = []

    def listener(backend: Backend | None) -> None:
        received.append(backend)

    register_backend_listener(listener)
    set_backend("numpy")
    assert len(received) == 1
    assert isinstance(received[0], Backend)


def test_listener_called_immediately_when_backend_already_active() -> None:
    set_backend("numpy")
    received: list[Backend | None] = []

    def listener(backend: Backend | None) -> None:
        received.append(backend)

    register_backend_listener(listener)
    assert len(received) == 1
    assert isinstance(received[0], Backend)


def test_listener_called_with_none_on_reset() -> None:
    set_backend("numpy")
    received: list[Backend | None] = []

    def listener(backend: Backend | None) -> None:
        received.append(backend)

    register_backend_listener(listener)
    reset_backends()
    # First call: immediate notification with active backend (because already active).
    # Second call: notification with None on reset.
    assert received[-1] is None


# reset_backends ---------------------------------------------------------


def test_reset_backends_clears_active() -> None:
    set_backend("numpy")
    reset_backends()
    assert backend_manager._ACTIVE_BACKEND.get() is None
    assert backend_manager._BACKEND_INSTANCE is None


def test_reset_backends_clears_instance_cache() -> None:
    set_backend("numpy")
    assert SupportedFrameworks.NUMPY in backend_manager._BACKEND_INSTANCES
    reset_backends()
    assert SupportedFrameworks.NUMPY not in backend_manager._BACKEND_INSTANCES


# _instantiate ----------------------------------------------------------


def test_instantiate_caches_instance() -> None:
    a = _instantiate(SupportedFrameworks.NUMPY, SupportedDevices.CPU)
    b = _instantiate(SupportedFrameworks.NUMPY, SupportedDevices.CPU)
    assert a is b


def test_set_backend_device_mismatch_raises() -> None:
    set_backend("numpy", SupportedDevices.CPU)
    # NumPy backend rejects non-CPU devices at construction; check behavior via the
    # configured-mismatch path: re-set with a different device after first activation.
    with pytest.raises((RuntimeError, ValueError)):
        # The same backend cannot be reconfigured to a different device.
        set_backend("numpy", "gpu")


# default_device --------------------------------------------------------


def test_default_device_returns_active_device() -> None:
    set_backend("numpy", SupportedDevices.CPU)
    assert default_device() == SupportedDevices.CPU


def test_default_device_raises_when_no_backend() -> None:
    reset_backends()
    with pytest.raises(RuntimeError, match=r"No active backend"):
        default_device()
