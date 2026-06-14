from typing import TYPE_CHECKING

from decent_array.types._dtypes import _ALL_DTYPES, dtype, dtypes
from decent_array.types._types import (
    ArrayKey as ArrayKey,
)
from decent_array.types._types import (
    ArrayLike as ArrayLike,
)
from decent_array.types._types import (
    DTypes as DTypes,
)
from decent_array.types._types import (
    SupportedArrayTypes as SupportedArrayTypes,
)
from decent_array.types._types import (
    SupportedDevices as SupportedDevices,
)
from decent_array.types._types import (
    SupportedFrameworks as SupportedFrameworks,
)

_static_exports = [  # noqa: RUF067
    "ArrayLike",
    "SupportedArrayTypes",
    "ArrayKey",
    "SupportedFrameworks",
    "SupportedDevices",
    "DTypes",
    "dtype",
    "dtypes",
]

_all_dtypes = list(_ALL_DTYPES.keys())  # noqa: RUF067
_available_dtypes = [name for name, dt in _ALL_DTYPES.items() if dt.available]  # noqa: RUF067

__all_docs__ = _static_exports + _all_dtypes

__all__ = [
    "ArrayKey",
    "ArrayLike",
    "DTypes",
    "SupportedArrayTypes",
    "SupportedDevices",
    "SupportedFrameworks",
    "dtype",
    "dtypes",
]

if not TYPE_CHECKING:  # noqa: RUF067
    __all__ += _available_dtypes  # noqa: PLE0605
