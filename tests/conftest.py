"""Shared fixtures: parametrize tests across every (framework, device) combination.

Each test using the ``backend`` fixture runs once per (framework, device) pair from
:class:`SupportedFrameworks` x :class:`SupportedDevices`. Combinations whose backend
package is missing or whose device is not present on the current host are marked
``skip`` so the test report stays interpretable on machines with partial accelerator
support.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from decent_array.interoperability._backend_manager import reset_backends
from decent_array.types import SupportedDevices, SupportedFrameworks

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


def _framework_importable(framework: SupportedFrameworks) -> bool:
    try:
        if framework == SupportedFrameworks.NUMPY:
            import numpy  # noqa: F401, PLC0415
        elif framework == SupportedFrameworks.PYTORCH:
            import torch  # noqa: F401, PLC0415
        elif framework == SupportedFrameworks.JAX:
            import jax  # noqa: F401, PLC0415
        elif framework == SupportedFrameworks.TENSORFLOW:
            import tensorflow  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _device_available(framework: SupportedFrameworks, device: SupportedDevices) -> bool:
    """Return True iff this (framework, device) pair can run on the current host."""
    if not _framework_importable(framework):
        return False
    if framework == SupportedFrameworks.NUMPY:
        return device == SupportedDevices.CPU
    if framework == SupportedFrameworks.PYTORCH:
        import torch  # noqa: PLC0415

        if device == SupportedDevices.CPU:
            return True
        if device == SupportedDevices.GPU:
            try:
                return bool(torch.cuda.is_available())
            except Exception:
                return False
        if device == SupportedDevices.MPS:
            try:
                return bool(torch.backends.mps.is_available())
            except Exception:
                return False
    if framework == SupportedFrameworks.JAX:
        if device == SupportedDevices.MPS:
            return False
        import jax  # noqa: PLC0415

        try:
            jax.devices(device.value)
        except Exception:
            return False
        return True
    if framework == SupportedFrameworks.TENSORFLOW:
        if device == SupportedDevices.MPS:
            return False
        import tensorflow as tf  # noqa: PLC0415

        if device == SupportedDevices.CPU:
            return True
        if device == SupportedDevices.GPU:
            try:
                return len(tf.config.list_physical_devices("GPU")) > 0
            except Exception:
                return False
    return False


def _backend_params() -> list[pytest.ParameterSet]:
    params: list[pytest.ParameterSet] = []
    for framework in SupportedFrameworks:
        for device in SupportedDevices:
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
def backend(request: FixtureRequest) -> Iterator[tuple[SupportedFrameworks, SupportedDevices]]:
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
