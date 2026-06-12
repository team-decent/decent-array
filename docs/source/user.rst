User Guide
==========

Installation
------------
Requires `Python 3.13+ <https://www.python.org/downloads/>`_

.. code-block:: bash

    pip install decent-array


dtypes
------

decent-array exposes a number of dtypes which are bound to the corresponding framework-native dtypes. The following
lists collect the dtypes always supported, and the dtypes supported only by some frameworks/under certain conditions.
The table below reports all the details.



dtypes always available
~~~~~~~~~~~~~~~~~~~~~~~

- bool
- int8
- int16
- int32
- uint8
- float16
- float32
- complex64

dtypes conditionally available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- int64
- uint16
- uint32
- uint64
- bfloat16
- float64
- float128
- complex128
- complex256
- qint8
- quint8
- qint16
- quint16
- qint32
- unicode
- bytes
- object
- void


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
   * - ``object``
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