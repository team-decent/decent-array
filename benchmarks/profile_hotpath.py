"""
Profile ``decent_array`` hot paths to find where wrapper overhead actually lives.

Runs a tight loop of representative operations (mix of ``Array`` operators and
``iop`` function calls) under :mod:`cProfile`, then prints the highest-impact
callees by cumulative time. The output answers "of the wrapper overhead we see
in ``bench_array.py`` / ``bench_iop.py``, where is the time actually spent?" —
turning a vague ratio number into specific functions to optimize.

Notes:
* cProfile adds per-call overhead of ~1 µs, which dominates anything sub-µs. The
  *relative* shape of the profile is still informative (which functions are
  called most often, and which take the largest share of cumulative time);
  don't read the absolute µs values.
* Repeats are deliberately on the small side so that pure-Python and mypyc-
  compiled runs both finish quickly. For mypyc-compiled modules cProfile only
  records entry/exit (compiled internals are opaque), so the profile is most
  informative against the pure-Python source.

Run with::

    python benchmarks/profile_hotpath.py
    python benchmarks/profile_hotpath.py --backend pytorch
    python benchmarks/profile_hotpath.py --topn 50
"""

from __future__ import annotations

import argparse
import cProfile
import importlib
import pstats
from io import StringIO

import numpy as np

import decent_array.interoperability as iop
from decent_array import Array


def _is_compiled() -> tuple[bool, str]:
    module = importlib.import_module("decent_array._array")
    path = module.__file__ or "<unknown>"
    return path.endswith((".so", ".pyd")), path


def hot_loop(a: Array, b: Array, iterations: int) -> None:
    """A representative mix of operator-style and function-style calls."""
    for _ in range(iterations):
        # Operator surface (Array dunders)
        _ = a + b
        _ = a - b
        _ = a * b
        _ = a / b
        _ = a + 2.0
        _ = -a
        _ = abs(a)
        _ = a**2.0
        # Function surface (iop dispatch)
        _ = iop.add(a, b)
        _ = iop.multiply(a, b)
        _ = iop.sum(a)
        _ = iop.vector_norm(a)
        _ = iop.dot(a, b)
        _ = iop.sqrt(a)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="numpy", choices=["numpy", "pytorch", "jax", "tensorflow"])
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--topn", type=int, default=30)
    parser.add_argument("--size", type=int, default=100, help="size of the arrays to operate on")
    args = parser.parse_args()

    iop.set_backend(args.backend)

    # Modest fixed-size array — small enough that wrapper overhead is the
    # dominant cost, large enough that the underlying ops aren't degenerate.
    a = iop.uniform(0.0, 1.0, shape=(args.size,))
    b = iop.uniform(0.0, 1.0, shape=(args.size,))

    # Warmup so first-call jit / cache effects don't pollute the profile.
    hot_loop(a, b, iterations=10)

    profiler = cProfile.Profile()
    profiler.enable()
    hot_loop(a, b, iterations=args.iterations)
    profiler.disable()

    compiled, path = _is_compiled()
    print(f"backend: {args.backend}    compiled: {'yes' if compiled else 'no'}")
    print(f"Array module: {path}")
    print(f"iterations: {args.iterations:_}\n")

    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.sort_stats("cumulative")
    stats.print_stats(args.topn)
    print(buf.getvalue())

    print("\n--- top callees by total (self) time ---\n")
    buf2 = StringIO()
    stats2 = pstats.Stats(profiler, stream=buf2)
    stats2.sort_stats("tottime")
    stats2.print_stats(args.topn)
    print(buf2.getvalue())


if __name__ == "__main__":
    main()
