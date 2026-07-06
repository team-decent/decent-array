"""Common errors used across backends."""

from typing import Any

from mypy_extensions import mypyc_attr


@mypyc_attr(native_class=False)
class NotScalarError(TypeError):
    def __init__(self, ndim: int):
        super().__init__(f"Only 0-dim arrays can be converted to Python scalars, got {ndim}-dim array.")


class NDimError(ValueError):
    def __init__(self, required_ndim: int, actual_ndim: int):
        super().__init__(f"A {required_ndim}-dim array is required, got {actual_ndim}-dim array.")


class MatrixTransposeError(ValueError):
    def __init__(self, ndim: int):
        super().__init__(f"An aray with at least 2 dimensions is required, got {ndim}-dim array.")


class UnsupportedDTypeCreationError(ValueError):
    def __init__(self, dtype: Any, backend_name: str, device_name: str):  # noqa: ANN401
        super().__init__(f"Unsupported dtype '{dtype}' for {backend_name} on {device_name}.")


class UnsupportedDeviceError(ValueError):
    def __init__(self, backend_name: str, device_name: str):
        super().__init__(f"{backend_name} does not support device '{device_name}'.")


stack_empty_error = ValueError("Cannot stack an empty sequence of arrays.")


no_backend_error = RuntimeError("No backend active: call 'set_backend' with a supported framework to activate one.")
