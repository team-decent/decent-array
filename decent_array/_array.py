"""
Lightweight wrapper around backend-native arrays.

The :class:`Array` class wraps a single value of the active backend's framework type.
Under the single-active-backend invariant maintained by the backend manager, every
:class:`Array` at runtime holds a value from the same framework, so operators dispatch
directly to the active backend without per-call isinstance dispatch.

Operator contract is *strict*: binary arithmetic and indexing accept either another
:class:`Array` or a Python scalar (``int``/``float``).

Hot-path notes:

* ``__add__``/``__sub__``/``__mul__``/``__truediv__``/``__matmul__``, the unary
  ``__neg__``/``__abs__``/``__pow__``, the comparisons ``__eq__``/``__ne__``/``__lt__``/
  ``__le__``/``__gt__``/``__ge__`` and the bitwise ``__and__``/``__rand__`` are
  inlined: every supported framework's tensor implements the equivalent operator
  natively with numpy-equivalent semantics, so routing through the interoperability layer
  introduces unnecessary overhead.
* Operators that *do* go through the backend (in-place math, indexing, properties
  like ``shape``/``transpose``) read the cached ``_backend`` slot.
* Overriding ``__eq__`` makes :class:`Array` unhashable (Python clears ``__hash__``
  automatically). This matches numpy/torch/jax/tf, where element-wise equality is
  more useful than identity-based hashing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from decent_array.interoperability._backend_manager import register_backend_listener

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from decent_array.interoperability._abstracts import Backend
    from decent_array.types import ArrayKey, SupportedArrayTypes, SupportedDevices


_BACKEND_INSTANCE: Backend | None = None


def _update_backend(backend: Backend | None) -> None:
    global _BACKEND_INSTANCE  # noqa: PLW0603
    _BACKEND_INSTANCE = backend


register_backend_listener(_update_backend)


class Array:  # noqa: PLR0904
    """
    Wrapper around a single backend-native array.

    Storage is two slots (``value``, ``_backend``) declared via ``__slots__``;
    instances have no ``__dict__``. Every operator that delegates to the backend
    reads the cached slot, so dispatch is one slot load plus the backend method call.
    """

    __slots__ = ("_backend", "value")

    def __init__(self, value: SupportedArrayTypes) -> None:
        """
        Wrap ``value`` in an :class:`Array`.

        Args:
            value: A backend-native array (or scalar) to wrap. The attribute is typed
                as :class:`typing.Any` because the wrapper is intentionally type-erased
                — backend code accesses :attr:`value` knowing the framework type, and
                typing it more strictly forces a ``cast`` at every call site without
                runtime benefit.

        Raises:
            RuntimeError: If no backend is registered yet. An :class:`Array` cannot be
                constructed until a backend is set; call :func:`set_backend` to initialize
                the interoperability layer.

        """
        if _BACKEND_INSTANCE is None:
            raise RuntimeError(
                "No backend registered yet. An Array cannot be constructed until a backend is set. "
                "Call set_backend() to initialize the interoperability layer."
            )

        self.value: Any = value
        self._backend: Backend = _BACKEND_INSTANCE

    # Binary arithmetic ----------------------------------------------------

    def __add__(self, other: int | float | complex | Array, /) -> Array:
        """Return the sum of the array and another array or a scalar."""
        return Array(self.value + (other.value if type(other) is Array else other))

    def __radd__(self, other: int | float | complex | Array, /) -> Array:
        """Return the sum of the array and a scalar."""
        return Array(other + self.value)

    def __sub__(self, other: int | float | complex | Array, /) -> Array:
        """Return the subtraction of another array or a scalar from the array."""
        return Array(self.value - (other.value if type(other) is Array else other))

    def __rsub__(self, other: int | float | complex | Array, /) -> Array:
        """Return the subtraction of the array from a scalar."""
        return Array(other - self.value)

    def __mul__(self, other: int | float | complex | Array, /) -> Array:
        """Return the product of the array and another array or a scalar."""
        return Array(self.value * (other.value if type(other) is Array else other))

    def __rmul__(self, other: int | float | complex | Array, /) -> Array:
        """Return the product of the array and a scalar."""
        return Array(other * self.value)

    def __truediv__(self, other: int | float | complex | Array, /) -> Array:
        """Return the true division of the array by ``other``."""
        return Array(self.value / (other.value if type(other) is Array else other))

    def __rtruediv__(self, other: int | float | complex | Array, /) -> Array:
        """Return the true division of ``other`` by the array."""
        return Array(other / self.value)

    def __matmul__(self, other: Array, /) -> Array:
        """Return the matrix multiplication of the array with ``other``."""
        return Array(self.value @ other.value)

    def __rmatmul__(self, other: Array, /) -> Array:
        """Return the matrix multiplication of ``other`` with the array."""
        return Array(other.value @ self.value)

    def __pow__(self, other: int | float | complex | Array, /) -> Array:
        """Exponentiate the array by a scalar power."""
        # numpy/torch/jax/tf all implement ``tensor ** p`` with semantics matching the
        # backend's ``pow``; routing through the backend would cost an extra method
        # call for no behavioral difference.
        return Array(self.value ** (other.value if type(other) is Array else other))

    # Comparisons ----------------------------------------------------------
    #
    # Element-wise comparisons return an :class:`Array` of bools. The ``__eq__`` and
    # ``__ne__`` parameters are typed ``object`` to match the LSP signature inherited
    # from :class:`object`; the body still enforces the strict ``Array | scalar``
    # contract via the underlying framework's operator (incompatible operands raise
    # from the backend's native comparison, matching ``__add__``).
    #
    # Overriding ``__eq__`` makes instances unhashable; ``__hash__ = None`` makes that
    # explicit (and silences the lint that flags the dropped ``__hash__``).

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: bool | int | float | complex | Array, /) -> Array:  # type: ignore[override]
        """Element-wise equality."""
        return Array(self.value == (other.value if type(other) is Array else other))

    def __ne__(self, other: bool | int | float | complex | Array, /) -> Array:  # type: ignore[override]
        """Element-wise inequality."""
        return Array(self.value != (other.value if type(other) is Array else other))

    def __lt__(self, other: int | float | Array, /) -> Array:
        """Element-wise less-than."""
        return Array(self.value < (other.value if type(other) is Array else other))

    def __le__(self, other: int | float | Array, /) -> Array:
        """Element-wise less-than-or-equal."""
        return Array(self.value <= (other.value if type(other) is Array else other))

    def __gt__(self, other: int | float | Array, /) -> Array:
        """Element-wise greater-than."""
        return Array(self.value > (other.value if type(other) is Array else other))

    def __ge__(self, other: int | float | Array, /) -> Array:
        """Element-wise greater-than-or-equal."""
        return Array(self.value >= (other.value if type(other) is Array else other))

    # Bitwise --------------------------------------------------------------
    #
    # Bitwise AND is only defined for integer/boolean dtypes. ``__and__``'s ``Array
    # | int`` is a Union (mypyc keeps Union operands boxed, so a ``bool`` operand
    # stays a ``bool`` and TF's strict dtype check accepts it). ``__rand__``'s
    # operand is typed ``Any`` for the same reason: a single-primitive annotation
    # like ``int`` causes mypyc to unbox a ``True`` to ``1`` before the call body,
    # which fails TF's ``1 & bool_tensor`` rejection. Native operator semantics on
    # the wrapped tensor enforce the actual dtype contract.

    def __and__(self, other: bool | int | Array, /) -> Array:
        """Element-wise bitwise/logical AND."""
        return Array(self.value & (other.value if type(other) is Array else other))

    def __rand__(self, other: bool | int | Array, /) -> Array:
        """Element-wise bitwise/logical AND with the array on the right."""
        return Array((other.value if type(other) is Array else other) & self.value)

    # In-place arithmetic --------------------------------------------------
    #
    # The backend handles the framework's mutability semantics: numpy/pytorch mutate
    # `value` in place, jax/tensorflow rebind it. In every case the returned object is
    # the same wrapper instance, so we just discard the return and yield ``self``.

    def __iadd__(self, other: int | float | complex | Array, /) -> Self:
        """In-place addition."""
        self._backend.iadd(self, other)
        return self

    def __isub__(self, other: int | float | complex | Array, /) -> Self:
        """In-place subtraction."""
        self._backend.isubtract(self, other)
        return self

    def __imul__(self, other: int | float | complex | Array, /) -> Self:
        """In-place multiplication."""
        self._backend.imultiply(self, other)
        return self

    def __itruediv__(self, other: int | float | complex | Array, /) -> Self:
        """In-place true division."""
        self._backend.idivide(self, other)
        return self

    # Unary ----------------------------------------------------------------

    def __neg__(self) -> Array:
        """Return the negation of the array."""
        # Native ``-tensor`` matches the backend's ``negative`` wrapper across all
        # supported frameworks, so the indirection is not needed.
        return Array(-self.value)

    def __abs__(self) -> Array:
        """Return the absolute value of the array."""
        # Same rationale as ``__neg__`` — native ``abs(tensor)`` matches each
        # backend's ``absolute`` implementation.
        return Array(abs(self.value))

    # Indexing -------------------------------------------------------------

    def __getitem__(self, key: ArrayKey, /) -> Array:
        """Return the item at ``key``."""
        return self._backend.get_item(self, key)

    def __setitem__(self, key: ArrayKey, value: bool | int | float | complex | Array, /) -> None:
        """Set the item at ``key`` to ``value``."""
        self._backend.set_item(self, key, value)

    # Containers / iteration ----------------------------------------------

    def __len__(self) -> int:
        """Return the size of the first dimension of the array."""
        return len(self.value)

    # Coercion -------------------------------------------------------------

    def __float__(self) -> float:
        """Coerce a scalar array to a Python float."""
        return float(self._backend.squeeze(self).value)

    # Repr -----------------------------------------------------------------

    def __repr__(self) -> str:
        """Show the wrapper and the wrapped value."""
        return f"Array({self.value!r})"

    def __str__(self) -> str:
        """Stringify the wrapped value, not the wrapper."""
        return str(self.value)

    # Properties -----------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the array."""
        return self._backend.shape(self)

    @property
    def size(self) -> int:
        """Return the total number of elements in the array."""
        return self._backend.size(self)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the array."""
        return self._backend.ndim(self)

    @property
    def transpose(self) -> Array:
        """Return a transposed view of the array."""
        return self._backend.transpose(self)

    @property
    def T(self) -> Array:  # noqa: N802
        """Return a transposed view of the array."""
        return self.transpose

    @property
    def any(self) -> bool:
        """Return True if any element of the array is truthy."""
        return self._backend.any(self)

    @property
    def all(self) -> bool:
        """Return True if all elements of the array are truthy."""
        return self._backend.all(self)

    @property
    def device(self) -> SupportedDevices:
        """Return the device of the array."""
        return self._backend.device_of(self)

    @property
    def numpy(self) -> NDArray[Any]:
        """Return a NumPy array view of the array's data."""
        return self._backend.to_numpy(self)
