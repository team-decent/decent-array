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
    """
    Normalize an indexing key for the native framework.

    Unwraps :class:`Array` items and converts custom ``SupportsIndex`` objects (e.g. ``W[agent_i, agent_j]``)
    to ``int``, since TensorFlow and JAX reject them. Native arrays/scalars (which expose ``shape``) and
    ``bool`` (which numpy treats as a new axis) are passed through unchanged.
    """
    if type(key) is Array:
        return key.value
    if type(key) is tuple:
        return tuple(unwrap_key(k) for k in key)
    if not isinstance(key, int) and hasattr(key, "__index__") and not hasattr(key, "shape"):
        return operator.index(key)
    return key


def is_scalar(x: Array) -> bool:
    """Return True if ``x`` is a 0-dim Array."""
    return x.ndim == 0
