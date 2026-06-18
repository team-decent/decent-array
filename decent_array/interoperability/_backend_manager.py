from __future__ import annotations

import importlib
from collections.abc import Callable
from contextvars import ContextVar

import decent_array._constants as constants
from decent_array import types
from decent_array.types._dtypes import _SUPPORTED
from decent_array.types._types import Devices, Frameworks

from ._abstracts import Backend

_BACKEND_REGISTRY: dict[Frameworks, type[Backend]] = {}
_BACKEND_INSTANCES: dict[Frameworks, Backend] = {}
_ACTIVE_BACKEND: ContextVar[Frameworks | None] = ContextVar(
    "decent_array.interoperability.active_backend", default=None
)
_BACKEND_LISTENERS: list[Callable[[Backend | None], None]] = []
_BACKEND_INSTANCE: Backend | None = None


def set_backend(
    backend: Frameworks | str,
    device: Devices | str = Devices.CPU,
) -> None:
    """
    Set the active backend (and target device) for the current execution context.

    The first call binds both the backend and the device; subsequent calls must use the
    same backend *and* the same device or a :class:`RuntimeError` is raised. This
    single-backend, single-device invariant lets the rest of the interoperability layer
    skip framework dispatch and isinstance checks, and lets backends construct array
    creation routines bound to a specific accelerator.

    Backend modules are auto-imported on demand.

    Args:
        backend: A :class:`~decent_array.types.Frameworks` value, its canonical string (e.g.
            ``"numpy"``, ``"pytorch"``).
        device: Target accelerator. Accepts a :class:`~decent_array.types.Devices` value or its
            string equivalent (``"cpu"``, ``"gpu"``, ``"mps"``). Defaults to CPU. The
            backend's array-creation methods produce arrays on this device by default.

    Raises:
        RuntimeError: If a different backend (or the same backend with a different device)
            is already active in this context.
        ImportError: If the backend module cannot be imported (e.g. due to a missing optional dependency).

    """  # noqa: DOC502
    requested = _normalize(backend)
    requested_device = device if isinstance(device, Devices) else Devices(device)

    current = _ACTIVE_BACKEND.get()
    if current is not None and current != requested:
        raise RuntimeError(
            f"Backend already set to '{current.value}', cannot set to '{requested.value}'. "
            "A single execution context may only use one backend."
        )

    cached = _instantiate(requested, requested_device)
    if cached.device != requested_device:
        raise RuntimeError(
            f"Backend '{requested.value}' already configured with device "
            f"'{cached.device.value}', cannot reconfigure to '{requested_device.value}'."
        )

    if current is None:
        _ACTIVE_BACKEND.set(requested)
        global _BACKEND_INSTANCE  # noqa: PLW0603
        _BACKEND_INSTANCE = cached
        for listener in _BACKEND_LISTENERS:
            listener(_BACKEND_INSTANCE)

    _bind_dtypes(_BACKEND_INSTANCE)
    _bind_constants(_BACKEND_INSTANCE)


def register_backend_listener(listener: Callable[[Backend | None], None]) -> None:
    """
    Register a callback to be invoked on backend activation.

    The callback receives the active backend instance as its only argument. If a backend
    is already active, the callback is invoked immediately with the current backend.

    Args:
        listener: A callable that accepts a single :class:`Backend` instance argument.

    """
    _BACKEND_LISTENERS.append(listener)
    if _BACKEND_INSTANCE is not None:
        listener(_BACKEND_INSTANCE)


def register_backend(
    backend: Frameworks,
    cls: type[Backend],
) -> None:
    """
    Register a backend class under a :class:`Frameworks` value.

    Called once per backend module *after* the class definition.
    Backends are instantiated lazily on first use. Re-registering replaces the
    previous class and discards any cached instance, but keeps existing aliases
    (which still point to the same canonical name).

    Args:
        backend: Canonical backend identifier.
        cls: A concrete subclass of :class:`Backend`.

    Raises:
        TypeError: If ``cls`` is not a subclass of :class:`Backend`.

    """
    if not issubclass(cls, Backend):
        raise TypeError(f"Registered backend must be a subclass of Backend, got {cls}")
    _BACKEND_REGISTRY[backend] = cls
    _BACKEND_INSTANCES.pop(backend, None)


def reset_backends() -> None:
    """
    Clear the active backend and all cached instances for the current context.

    Intended for tests or tightly scoped execution; not part of normal use. Registry
    entries (classes and aliases) are preserved.
    """
    global _BACKEND_INSTANCE  # noqa: PLW0603
    _ACTIVE_BACKEND.set(None)
    _BACKEND_INSTANCES.clear()
    _BACKEND_INSTANCE = None
    for listener in _BACKEND_LISTENERS:
        listener(None)


def default_device() -> Devices:
    """
    Return the default device for the active backend.

    Raises:
        RuntimeError: If no backend is currently active.

    """
    backend = _BACKEND_INSTANCE
    if backend is None:
        raise RuntimeError(
            "No active backend. Call set_backend() to initialize a backend before querying the default device."
        )
    return backend.device


def _normalize(backend: Frameworks | str) -> Frameworks:
    """
    Convert a backend identifier to its canonical :class:`Frameworks` value.

    Raises:
        KeyError: If the input is not a valid backend identifier.

    """
    if isinstance(backend, Frameworks):
        return backend
    try:
        return Frameworks(backend)
    except ValueError as exc:
        valid = ", ".join(f.value for f in Frameworks)
        raise KeyError(f"Unknown backend '{backend}'. Valid backends: {valid}.") from exc


def _instantiate(backend: Frameworks, device: Devices) -> Backend:
    """
    Get or create a backend instance for the given backend and device.

    Raises:
        KeyError: If the backend is not registered and cannot be auto-imported.

    """
    if backend in _BACKEND_INSTANCES:
        return _BACKEND_INSTANCES[backend]

    if backend not in _BACKEND_REGISTRY:
        _auto_import(backend)  # Attempt to load the backend module, which should register the backend as a side-effect.

    cls = _BACKEND_REGISTRY.get(backend)
    if cls is None:
        raise KeyError(
            f"Backend '{backend.value}' is not registered. Ensure the corresponding backend module is importable."
        )

    instance = cls(device=device)
    _BACKEND_INSTANCES[backend] = instance
    return instance


def _auto_import(backend: Frameworks) -> None:
    """
    Import the backend's package so its registration side-effect runs.

    Raises:
        ImportError: If the backend module cannot be imported.

    """
    current_module = __name__.rsplit(".", 1)[0]
    module_name = current_module + f"._{backend.value}"
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import the backend module for '{backend.value}'. Ensure the "
            "corresponding backend package is installed and importable."
        ) from exc


def _bind_dtypes(backend: Backend | None) -> None:
    """Bind dtype objects to the corresponding backend dtypes (if available)."""
    if backend is None:
        return
    for name in _SUPPORTED:
        dt = getattr(types, name)
        backend_dt = getattr(backend, name, None)
        dt._available = backend_dt is not None  # noqa: SLF001
        dt._backend_dtype = backend_dt  # noqa: SLF001


def _bind_constants(backend: Backend | None) -> None:
    """Bind constants to the corresponding backend constants."""
    if backend is None:
        return
    for name in constants._CONSTANTS:  # noqa: SLF001
        backend_c = getattr(backend, name, None)
        if backend_c is None:
            return
        setattr(constants, name, backend_c)
