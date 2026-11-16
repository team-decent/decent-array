import operator
from typing import Any

from decent_array._array import Array


def unwrap(x: Any) -> Any:  # noqa: ANN401
    """
    Return the underlying value of an :class:`Array`, or pass ``x`` through.

    Typed as ``Any`` because operator dunders may pass either an :class:`Array` or a
    Python scalar; the strict abstract signature would force a ``cast`` at every call
    site without runtime benefit.
    """
    return x.value if type(x) is Array else x


def unwrap_key(key: Any) -> Any:  # noqa: ANN401
    """Normalize an indexing key by unwrapping :class:`Array` items and custom ``__index__`` objects."""
    if type(key) is Array:
        return key.value
    if isinstance(key, tuple):
        return tuple(unwrap_key(k) for k in key)
    if (
        not isinstance(key, int | slice | type(None) | type(Ellipsis))
        and not hasattr(key, "shape")
        and hasattr(key, "__index__")
    ):
        return operator.index(key)
    return key


def is_scalar(x: Array) -> bool:
    """Return True if ``x`` is a 0-dim Array."""
    return x.ndim == 0
