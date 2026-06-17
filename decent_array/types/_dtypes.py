"""Data type definitions."""

from __future__ import annotations

from typing import Any

_SUPPORTED = {
    "bfloat16",
    "bool_",
    "bytes_",
    "complex64",
    "complex128",
    "complex256",
    "float16",
    "float32",
    "float64",
    "float128",
    "int8",
    "int16",
    "int32",
    "int64",
    "object_",
    "qint8",
    "qint16",
    "qint32",
    "quint8",
    "quint16",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "unicode_",
    "void",
}


class dtype:  # noqa: N801
    """Base class for dtypes."""

    def __init__(self, name: str):
        # name doesn't map to any dtype
        if name not in _SUPPORTED:
            raise ValueError(f"dtype {name} is not supported. Supported dtypes: {', '.join(_SUPPORTED)}")

        # initialize with placeholder values
        self._name = name
        self._backend_dtype: Any = None
        self._available = False

    @property
    def name(self) -> str:
        """Name of the dtype."""
        return self._name

    @property
    def available(self) -> bool:
        """Availability of the dtype (dependent on backend, device, backend settings, OS)."""
        return self._available

    @property
    def backend_dtype(self) -> Any:  # noqa: ANN401
        """The corresponding backend dtype object."""
        return self._backend_dtype

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


# instantiate all the supported dtypes
bool_ = dtype("bool_")
_BOOL_DTYPES = {"bool_": bool_}

int8 = dtype("int8")
int16 = dtype("int16")
int32 = dtype("int32")
int64 = dtype("int64")
_SIGNED_INT_DTYPES = {"int8": int8, "int16": int16, "int32": int32, "int64": int64}

uint8 = dtype("uint8")
uint16 = dtype("uint16")
uint32 = dtype("uint32")
uint64 = dtype("uint64")
_UNSIGNED_INT_DTYPES = {"uint8": uint8, "uint16": uint16, "uint32": uint32, "uint64": uint64}

float16 = dtype("float16")
bfloat16 = dtype("bfloat16")
float32 = dtype("float32")
float64 = dtype("float64")
float128 = dtype("float128")
_REAL_FLOATING_DTYPES = {
    "float16": float16,
    "bfloat16": bfloat16,
    "float32": float32,
    "float64": float64,
    "float128": float128,
}

complex64 = dtype("complex64")
complex128 = dtype("complex128")
complex256 = dtype("complex256")
_COMPLEX_FLOATING_DTYPES = {"complex64": complex64, "complex128": complex128, "complex256": complex256}

qint8 = dtype("qint8")
qint16 = dtype("qint16")
qint32 = dtype("qint32")
_QUANTIZED_SIGNED_INT_DTYPES = {"qint8": qint8, "qint16": qint16, "qint32": qint32}

quint8 = dtype("quint8")
quint16 = dtype("quint16")
_QUANTIZED_UNSIGNED_INT_DTYPES = {"quint8": quint8, "quint16": quint16}

unicode_ = dtype("unicode_")
bytes_ = dtype("bytes_")
object_ = dtype("object_")
void = dtype("void")
_MISCELLANEOUS_DTYPES = {"unicode_": unicode_, "bytes_": bytes_, "object_": object_, "void": void}

_INTEGRAL_DTYPES = (
    _SIGNED_INT_DTYPES | _UNSIGNED_INT_DTYPES | _QUANTIZED_SIGNED_INT_DTYPES | _QUANTIZED_UNSIGNED_INT_DTYPES
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
