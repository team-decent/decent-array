User Guide
==========

Installation
------------
Requires `Python 3.13+ <https://www.python.org/downloads/>`_

.. code-block:: bash

    pip install decent-array


dtypes
------

decent-array exposes a number of dtypes which are bound to the corresponding framework-native dtypes.
decent-array exposes dtypes as instances of :class:`~decent_array.types.dtype`, which can be accessed as
`from decent_array.types import float32` or just `types.float32`. Additionally, dtypes can be accessed by
`dtype("float32")`.

dtypes have attributes: :attr:`~decent_array.types.dtype.name`, :attr:`~decent_array.types.dtype.available` (see 
discussion below), :attr:`~decent_array.types.dtype.backend_dtype` (which binds the dtype to the corresponding
framework-native dtype).

The following lists the dtypes exposed by decent-array.

- Booleans
  - `bool_`

- Unsigned integers
  - `uint8`
  - `uint16`
  - `uint32`
  - `uint64`

- Signed integers
  - `int8`
  - `int16`
  - `int32`
  - `int64`

- Floating point
  - `float16`
  - `bfloat16`
  - `float32`
  - `float64`
  - `float128`

- Complex
  - `complex64`
  - `complex128`
  - `complex256`

- Quantized integers
  - `qint8`
  - `quint8`
  - `qint16`
  - `quint16`
  - `qint32`

- Miscellaneous
  - `unicode_`
  - `bytes_`
  - `object_`
  - `void`



Availability
~~~~~~~~~~~~

Not all dtypes are available in all configurations. The following factors affect dtype availability:
framework, device, OS, framework settings; additionally, some framework-native operations only support a subset of
dtypes.

There is no reliable way to check which dtypes are available in the current setting, and this is also subject to change
as frameworks develop. In decent-array, we make a best effort to determine whether a dtype is available and, if not,
set :attr:`~decent_array.types.dtype.available`. to `False`. Currently, we mark dtypes as un/available based on the
table below. Additionally, dtypes are marked as unavailable (and :attr:`~decent_array.types.dtype.backend_dtype` is
`None`) if the backend has not been initialized via `set_backend`.

However, :attr:`~decent_array.types.dtype.available` is generally not a reliable indicator of availability. The most
reliable way is to try operations that involve the dtype and observe if the framework raises an error.


.. list-table:: dtype support across frameworks
   :header-rows: 1
   :widths: 22 10 10 10 12 36

   * - dtype
     - NumPy
     - JAX
     - PyTorch
     - TensorFlow
     - Notes
   * - ``bool_``
     - ✓
     - ✓
     - ✓
     - ✓
     - 
   * - **Integers**
     -
     -
     -
     -
     -
   * - ``int8``
     - ✓
     - ✓
     - ✓
     - ✓
     - 
   * - ``int16``
     - ✓
     - ✓
     - ✓
     - ✓
     - 
   * - ``int32``
     - ✓
     - ✓
     - ✓
     - ✓
     - 
   * - ``int64``
     - ✓
     - ⚠️
     - ✓
     - ✓
     - JAX requires ``jax_enable_x64=True``
   * - **Unsigned integers**
     -
     -
     -
     -
     -
   * - ``uint8``
     - ✓
     - ✓
     - ✓
     - ✓
     - 
   * - ``uint16``
     - ✓
     - ⚠️
     - ⚠️
     - ✓
     - JAX requires ``jax_enable_x64=True``; PyTorch support is limited/experimental
   * - ``uint32``
     - ✓
     - ⚠️
     - ⚠️
     - ✓
     - JAX requires ``jax_enable_x64=True``; PyTorch support is limited/experimental
   * - ``uint64``
     - ✓
     - ⚠️
     - ⚠️
     - ✓
     - JAX requires ``jax_enable_x64=True``; PyTorch support is limited/experimental
   * - **Floating point**
     -
     -
     -
     -
     -
   * - ``float16``
     - ✓
     - ✓
     - ✓
     - ✓
     - 
   * - ``bfloat16``
     - ✗
     - ✓
     - ✓
     - ✓
     - 
   * - ``float32``
     - ✓
     - ✓
     - ✓
     - ✓
     - 
   * - ``float64``
     - ✓
     - ⚠️
     - ✓
     - ✓
     - JAX requires ``jax_enable_x64=True``
   * - ``float128``
     - ⚠️
     - ✗
     - ✗
     - ✗
     - NumPy support is platform-dependent
   * - **Complex**
     -
     -
     -
     -
     -
   * - ``complex64``
     - ✓
     - ✓
     - ✓
     - ✓
     - 
   * - ``complex128``
     - ✓
     - ⚠️
     - ✓
     - ✓
     - JAX requires ``jax_enable_x64=True``
   * - ``complex256``
     - ⚠️
     - ✗
     - ✗
     - ✗
     - NumPy support is platform-dependent
   * - **Quantized**
     -
     -
     -
     -
     -
   * - ``qint8``
     - ✗
     - ✗
     - ✓
     - ✓
     - 
   * - ``quint8``
     - ✗
     - ✗
     - ✓
     - ✓
     - 
   * - ``qint16``
     - ✗
     - ✗
     - ✗
     - ✓
     - 
   * - ``quint16``
     - ✗
     - ✗
     - ✗
     - ✓
     - 
   * - ``qint32``
     - ✗
     - ✗
     - ✓
     - ✓
     - 
   * - **Miscellaneous**
     -
     -
     -
     -
     -
   * - ``unicode_``
     - ✓
     - ✗
     - ✗
     - ✗
     - Equivalent to ``np.str_``
   * - ``bytes_``
     - ✓
     - ✗
     - ✗
     - ✓
     - Equivalent to ``np.bytes_`` and ``tf.string``
   * - ``object_``
     - ✓
     - ✗
     - ✗
     - ✗
     - 
   * - ``void``
     - ✓
     - ✗
     - ✗
     - ✗
     - 