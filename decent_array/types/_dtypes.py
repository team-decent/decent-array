"""Data type definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from decent_array.interoperability._backend_manager import register_backend_listener

if TYPE_CHECKING:
    from decent_array.interoperability._abstracts import Backend


_BACKEND_INSTANCE: Backend | None = None
_error = RuntimeError("No backend active: call 'set_backend' with a supported framework to activate one.")


def _update_backend(backend: Backend | None) -> None:
    global _BACKEND_INSTANCE  # noqa: PLW0603
    _BACKEND_INSTANCE = backend


register_backend_listener(_update_backend)


class dtype:  # noqa: N801
    """Base class for dtypes."""

    def __init__(self, name: str):
        if _BACKEND_INSTANCE is None:
            raise _error

        # name doesn't map to any dtype
        if name not in _ALL_DTYPES:
            raise ValueError(f"dtype {name} is not supported.")

        # this is None if the dtype is not supported
        backend_dtype = getattr(_BACKEND_INSTANCE, name, None)

        self._name = name
        self._backend_dtype: Any = backend_dtype
        self._available = backend_dtype is not None

    @property
    def name(self) -> str:
        """Name of the dtype."""
        return self._name

    @property
    def available(self) -> bool:
        """Availability of the dtype (dependent on backend, device, backend settings, OS)."""
        return self._available

    def __str__(self) -> str:
        """Name of the dtype."""
        return self.name

    def __eq__(self, other: object) -> bool:
        """Check equivalence by ``name`` attributes."""
        if not isinstance(other, dtype):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        """Hash of the dtype."""
        return hash(self.name)


# TODO create global error message for this when an unavailable dtype is attampted in astype and other funcs
# raise ValueError(f"dtype {name} is not supported.")  # TODO add device and backend name to contextualize

# TODO do we want to have an optional dependence on ml_dtypes to support bfloat16 for numpy?


_BOOL_DTYPES = {
    "bool_": dtype("bool"),
}

_SIGNED_INT_DTYPES = {
    "int8": dtype("int8"),
    "int16": dtype("int16"),
    "int32": dtype("int32"),
    "int64": dtype("int64"),
}

_UNSIGNED_INT_DTYPES = {
    "uint8": dtype("uint8"),
    "uint16": dtype("uint16"),
    "uint32": dtype("uint32"),
    "uint64": dtype("uint64"),
}

_REAL_FLOATING_DTYPES = {
    "float16": dtype("float16"),
    "bfloat16": dtype("bfloat16"),
    "float32": dtype("float32"),
    "float64": dtype("float64"),
    "float128": dtype("float128"),
}

_COMPLEX_FLOATING_DTYPES = {
    "complex64": dtype("complex64"),
    "complex128": dtype("complex128"),
    "complex256": dtype("complex256"),
}

_QUANTIZED_SIGNED_INT_DTYPES = {
    "qint8": dtype("qint8"),
    "qint16": dtype("qint16"),
    "qint32": dtype("qint32"),
}

_QUANTIZED_UNSIGNED_INT_DTYPES = {
    "quint8": dtype("quint8"),
    "quint16": dtype("quint16"),
}

_MISCELLANEOUS_DTYPES = {
    "unicode_": dtype("unicode"),
    "bytes_": dtype("bytes"),
    "object": dtype("object"),
    "void": dtype("void"),
}

_INTEGRAL_DTYPES = (_SIGNED_INT_DTYPES |
                    _UNSIGNED_INT_DTYPES |
                    _QUANTIZED_SIGNED_INT_DTYPES |
                    _QUANTIZED_UNSIGNED_INT_DTYPES
                    )
_NUMERIC_DTYPES = _INTEGRAL_DTYPES | _REAL_FLOATING_DTYPES | _COMPLEX_FLOATING_DTYPES
_ALL_DTYPES = _BOOL_DTYPES | _NUMERIC_DTYPES | _MISCELLANEOUS_DTYPES


_BACKEND_DTYPE_TO_DTYPE: dict[Any, dtype] = {}
for dt in _ALL_DTYPES.values():
    if dt.available:
        _BACKEND_DTYPE_TO_DTYPE[dt._backend_dtype] = dt  # noqa: SLF001


def dtypes() -> dict[str, dtype]:
    """Return a dictionary of available dtypes."""
    return {name: dt for name, dt in _ALL_DTYPES.items() if dt.available}
