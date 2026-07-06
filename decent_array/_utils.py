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


def is_scalar(x: Array) -> bool:
    """Return True if ``x`` is a 0-dim Array."""
    return x.ndim == 0
