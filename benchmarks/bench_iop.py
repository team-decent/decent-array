"""
Microbenchmark: ``iop.<fn>`` dispatch overhead vs native frameworks.

Measures the cost of calling top-level interoperability functions (which look
up the active backend on each call and dispatch to it) against calling each
framework's native equivalents directly. Same shape and intent as
``bench_array.py`` but for the function-style API surface rather than the
operator-style ``Array`` API. Iterates over every framework whose package is
importable; missing optional dependencies are skipped silently.

Where the two benchmarks share an op (``add``, ``mul``), differences in the
overhead column reflect dunder-method dispatch vs. module-level function
dispatch; the remaining ops (``sum``, ``dot``, ``norm``, ``mean``, ``sqrt``,
``sign``) only exist on this surface.

Run with::

    python benchmarks/bench_iop.py
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bench_common import (
    SIZES,
    BackendCase,
    activate_backend,
    discover_backends,
    fmt_row,
    parse_backends_arg,
    print_preamble,
    print_size_header,
    time_us_safe,
)

import decent_array.interoperability as iop
from decent_array import Array


def _native_ops(backend: str) -> dict[str, Callable[..., Any]]:
    """Return the native-framework equivalents of each ``iop.<fn>`` for ``backend``."""
    if backend == "numpy":
        import numpy as np  # noqa: PLC0415

        return {
            "add": np.add,
            "mul": np.multiply,
            "dot": np.dot,
            "sum": np.sum,
            "mean": np.mean,
            "norm": np.linalg.norm,
            "sqrt": np.sqrt,
            "sign": np.sign,
        }
    if backend == "pytorch":
        import torch  # noqa: PLC0415

        return {
            "add": torch.add,
            "mul": torch.mul,
            "dot": torch.dot,
            "sum": torch.sum,
            "mean": torch.mean,
            "norm": torch.linalg.norm,
            "sqrt": torch.sqrt,
            "sign": torch.sign,
        }
    if backend == "jax":
        import jax.numpy as jnp  # noqa: PLC0415

        return {
            "add": jnp.add,
            "mul": jnp.multiply,
            "dot": jnp.dot,
            "sum": jnp.sum,
            "mean": jnp.mean,
            "norm": jnp.linalg.norm,
            "sqrt": jnp.sqrt,
            "sign": jnp.sign,
        }
    if backend == "tensorflow":
        import tensorflow as tf  # noqa: PLC0415

        return {
            "add": tf.add,
            "mul": tf.multiply,
            "dot": lambda a, b: tf.tensordot(a, b, axes=1),
            "sum": tf.reduce_sum,
            "mean": tf.reduce_mean,
            "norm": tf.norm,
            "sqrt": tf.sqrt,
            "sign": tf.sign,
        }
    raise ValueError(f"unknown backend: {backend}")


def _bench_case(case: BackendCase) -> None:
    activate_backend(case.name)
    native = _native_ops(case.name)
    print(f"## {case.name}\n")
    for n in SIZES:
        a = case.make(n)
        b = case.make(n)
        d_a, d_b = Array(a), Array(b)

        print_size_header(n)
        rows = (
            ("add", lambda a=a, b=b: native["add"](a, b), lambda d_a=d_a, d_b=d_b: iop.add(d_a, d_b)),
            ("mul", lambda a=a, b=b: native["mul"](a, b), lambda d_a=d_a, d_b=d_b: iop.multiply(d_a, d_b)),
            ("dot", lambda a=a, b=b: native["dot"](a, b), lambda d_a=d_a, d_b=d_b: iop.dot(d_a, d_b)),
            ("sum", lambda a=a: native["sum"](a), lambda d_a=d_a: iop.sum(d_a)),
            ("mean", lambda a=a: native["mean"](a), lambda d_a=d_a: iop.mean(d_a)),
            ("norm", lambda a=a: native["norm"](a), lambda d_a=d_a: iop.norm(d_a)),
            ("sqrt", lambda a=a: native["sqrt"](a), lambda d_a=d_a: iop.sqrt(d_a)),
            ("sign", lambda a=a: native["sign"](a), lambda d_a=d_a: iop.sign(d_a)),
        )
        for op, native_fn, wrapped_fn in rows:
            n_us = time_us_safe(case, native_fn)
            w_us = time_us_safe(case, wrapped_fn)
            print(fmt_row(op, n_us, w_us))
        print()
    print()


def main() -> None:
    print_preamble("iop function-call overhead vs native frameworks")
    cases = discover_backends(only=parse_backends_arg())
    print(f"available backends: {', '.join(c.name for c in cases)}\n")
    for case in cases:
        _bench_case(case)


if __name__ == "__main__":
    main()
