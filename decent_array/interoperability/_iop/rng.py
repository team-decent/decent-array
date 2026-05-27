"""
Random-number coordination across backends.

The active backend handles its own RNG, but two extra concerns sit above it:

1. Python's :mod:`random` is often used incidentally and must also be seeded.
2. NumPy's RNG is frequently consulted by other frameworks (e.g. dataset shuffling
   helpers, scikit-learn pre-processing) regardless of the active backend, so its state
   must be tracked and restored alongside the active backend's state.

:class:`_RngCoordinator` owns both concerns. RNG functions exposed by ``_iop`` route
through a process-singleton coordinator.

When the active backend *is* numpy, the coordinator avoids double-seeding to keep the
RNG-state snapshot self-consistent.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, cast

from decent_array.interoperability._backend_manager import _instantiate, register_backend_listener
from decent_array.types import SupportedDevices, SupportedFrameworks

if TYPE_CHECKING:
    import numpy

    from decent_array import Array
    from decent_array.interoperability._abstracts import Backend
    from decent_array.interoperability._numpy import NumpyBackend


_NUMPY_STATE_KEY = "__numpy_rng_state__"
_PYTHON_RANDOM_KEY = "__python_random_state__"
_BACKEND_INSTANCE: Backend | None = None
_error = RuntimeError("No backend active: call 'set_backend' with a supported framework to activate one.")


def _update_backend(backend: Backend | None) -> None:
    global _BACKEND_INSTANCE  # noqa: PLW0603
    _BACKEND_INSTANCE = backend


register_backend_listener(_update_backend)


class _RngCoordinator:
    """Coordinate RNG seeding/state across the active backend, NumPy, and Python's random."""

    def __init__(self) -> None:
        self._global_seed: int | None = None

    def set_seed(self, seed: int, *, set_global_seed: bool = True) -> None:
        """
        Seed Python's ``random``, NumPy's RNG, and the active backend's RNG.

        Args:
            seed: Base seed.
            set_global_seed: If False, leaves :func:`get_seed` untouched. Use this for
                trial-local reseeding where the externally observable base seed must be
                preserved.

        """
        if _BACKEND_INSTANCE is None:
            raise _error

        random.seed(seed)
        active = _BACKEND_INSTANCE
        active.set_seed(seed)
        numpy_backend = self.numpy_backend()
        if numpy_backend is not active:
            numpy_backend.set_seed(seed)
        if set_global_seed:
            self._global_seed = seed

    def get_seed(self) -> int | None:
        """Return the seed last passed to :meth:`set_seed` (with ``set_global_seed=True``)."""
        return self._global_seed

    def get_rng_state(self) -> dict[str, Any]:
        """
        Snapshot the RNG state of the active backend, NumPy (if auxiliary), and Python's random.

        The active backend's state is returned as-is. If the active backend is not NumPy,
        NumPy's state is embedded under the reserved key ``"__numpy_rng_state__"``. The
        Python ``random`` state is always embedded under ``"__python_random_state__"`` so
        that incidental ``random.random()`` calls survive a snapshot/restore round-trip.

        """
        if _BACKEND_INSTANCE is None:
            raise _error

        active = _BACKEND_INSTANCE
        state = active.get_rng_state()
        state[_PYTHON_RANDOM_KEY] = random.getstate()
        numpy_backend = self.numpy_backend()
        if numpy_backend is not active:
            state[_NUMPY_STATE_KEY] = numpy_backend.get_rng_state()
        return state

    def set_rng_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`get_rng_state`."""
        if _BACKEND_INSTANCE is None:
            raise _error

        # Copy so we can mutate without surprising the caller.
        state = dict(state)
        python_state = state.pop(_PYTHON_RANDOM_KEY, None)
        if python_state is not None:
            random.setstate(python_state)
        active = _BACKEND_INSTANCE
        numpy_backend = self.numpy_backend()
        if numpy_backend is not active:
            numpy_state = state.pop(_NUMPY_STATE_KEY, None)
            if numpy_state is not None:
                numpy_backend.set_rng_state(numpy_state)
        active.set_rng_state(state)

    def numpy_backend(self) -> NumpyBackend:
        return cast("NumpyBackend", _instantiate(SupportedFrameworks.NUMPY, SupportedDevices.CPU))


_COORDINATOR = _RngCoordinator()


def set_seed(seed: int) -> None:
    """Seed Python ``random``, NumPy, and the active backend's RNG with ``seed``."""
    _COORDINATOR.set_seed(seed)


def _set_seed_without_global(seed: int) -> None:
    """
    Seed without changing the value returned by :func:`get_seed`.

    Used for trial-local reseeding where the externally observable base seed must be preserved.
    """
    _COORDINATOR.set_seed(seed, set_global_seed=False)


def _reset_rng() -> None:
    """Reset RNG state to a fresh state."""
    global _COORDINATOR  # noqa: PLW0603
    _COORDINATOR = _RngCoordinator()


def get_numpy_rng() -> numpy.random.Generator:
    """Return a NumPy Generator seeded from the current state."""
    np_backend = _COORDINATOR.numpy_backend()
    return np_backend._rng  # noqa: SLF001


def get_seed() -> int | None:
    """Return the most recently set global seed, or ``None`` if unset."""
    return _COORDINATOR.get_seed()


def get_rng_state() -> dict[str, Any]:
    """Return a snapshot of the active backend's RNG state."""
    return _COORDINATOR.get_rng_state()


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore an RNG snapshot produced by :func:`get_rng_state`."""
    _COORDINATOR.set_rng_state(state)


def derive_seed() -> int:
    """
    Derive a new seed from the current state.

    This is useful when you want to create a new generator that is independent but reproducible from the current one.
    For example, you might use this to seed a data loader's RNG based on the main RNG to ensure that data shuffling is
    reproducible across runs, but different from the main RNG used for model initialization.

    Returns:
        An integer seed derived from the current RNG state.

    """
    current_seed = get_seed()
    if current_seed is None:
        return random.randint(0, 2**32 - 1)
    # Derive a new seed by hashing the current seed with some random data.
    random_data = random.getrandbits(256)
    return (current_seed + random_data) % (2**32)


def normal(mean: float = 0.0, std: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
    """Draw normally distributed samples on the active backend."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.normal(mean, std, shape)


def uniform(low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
    """Draw uniformly distributed samples on the active backend."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.uniform(low, high, shape)


def normal_like(x: Array, mean: float = 0.0, std: float = 1.0) -> Array:
    """Draw normally distributed samples shaped like ``x`` with same dtype."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.normal_like(x, mean, std)


def uniform_like(x: Array, low: float = 0.0, high: float = 1.0) -> Array:
    """Draw uniformly distributed samples shaped like ``x`` with same dtype."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.uniform_like(x, low, high)


def choice(x: Array, size: int, replace: bool = True) -> Array:
    """Sample ``size`` elements from ``x``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.choice(x, size, replace)
