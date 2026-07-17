"""Shared fixtures: parametrize tests across every (framework, device) combination.

Each test using the ``backend`` fixture runs once per (framework, device) pair from
:class:`Frameworks` x :class:`Devices`. Combinations whose backend
package is missing or whose device is not present on the current host are marked
``skip`` so the test report stays interpretable on machines with partial accelerator
support.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from decent_array.interoperability._backend_manager import reset_backends
from decent_array.types import Devices, Frameworks

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


def _framework_importable(framework: Frameworks) -> bool:
    try:
        if framework == Frameworks.NUMPY:
            import numpy  # noqa: F401, PLC0415
        elif framework == Frameworks.PYTORCH:
            import torch  # noqa: F401, PLC0415
        elif framework == Frameworks.JAX:
            import jax  # noqa: F401, PLC0415
        elif framework == Frameworks.TENSORFLOW:
            import tensorflow  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _device_available(framework: Frameworks, device: Devices) -> bool:
    """Return True iff this (framework, device) pair can run on the current host."""
    if not _framework_importable(framework):
        return False
    if framework == Frameworks.NUMPY:
        return device == Devices.CPU
    if framework == Frameworks.PYTORCH:
        import torch  # noqa: PLC0415

        if device == Devices.CPU:
            return True
        if device == Devices.GPU:
            try:
                return bool(torch.cuda.is_available())
            except Exception:
                return False
        if device == Devices.MPS:
            try:
                return bool(torch.backends.mps.is_available())
            except Exception:
                return False
    if framework == Frameworks.JAX:
        if device == Devices.MPS:
            return False
        import jax  # noqa: PLC0415

        try:
            jax.devices(device.value)
        except Exception:
            return False
        return True
    if framework == Frameworks.TENSORFLOW:
        if device == Devices.MPS:
            return False
        import tensorflow as tf  # noqa: PLC0415

        if device == Devices.CPU:
            return True
        if device == Devices.GPU:
            try:
                return len(tf.config.list_physical_devices("GPU")) > 0
            except Exception:
                return False
    return False


def _backend_params() -> list[pytest.ParameterSet]:
    params: list[pytest.ParameterSet] = []
    for framework in Frameworks:
        for device in Devices:
            test_id = f"{framework.value}-{device.value}"
            if _device_available(framework, device):
                params.append(pytest.param((framework, device), id=test_id))
            else:
                params.append(
                    pytest.param(
                        (framework, device),
                        id=test_id,
                        marks=pytest.mark.skip(reason=f"{framework.value}/{device.value} unavailable"),
                    )
                )
    return params


BACKEND_PARAMS = _backend_params()


@pytest.fixture(params=BACKEND_PARAMS)
def backend(request: FixtureRequest) -> Iterator[tuple[Frameworks, Devices]]:
    """Activate the (framework, device) backend for this test, then reset on teardown."""
    from decent_array.interoperability import set_backend  # noqa: PLC0415

    framework, device = request.param
    set_backend(framework, device)
    yield framework, device
    reset_backends()


@pytest.fixture
def reset_after() -> Iterator[None]:
    """For tests that touch backend-manager state directly without the ``backend`` fixture."""
    yield
    reset_backends()
