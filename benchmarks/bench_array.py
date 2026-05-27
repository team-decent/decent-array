"""
Microbenchmark: ``decent_array.Array`` operator overhead vs native frameworks.

Measures the wrapper cost added by routing operators through ``Array.__add__``,
``Array.__neg__`` etc. against calling the framework's native operators
directly. Iterates over every framework whose package is importable; missing
optional dependencies are skipped silently.

The overhead column is ``wrapped / native`` runtime — values close to 1.0x mean
the wrapper is essentially free. Large values at small sizes are expected
(operator dispatch dominates) and should converge toward 1.0x as elementwise
work grows.

Run with::

    python benchmarks/bench_array.py
"""

from __future__ import annotations

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

from decent_array import Array


def _bench_case(case: BackendCase) -> None:
    activate_backend(case.name)
    print(f"## {case.name}\n")
    for n in SIZES:
        a = case.make(n)
        b = case.make(n)
        d_a, d_b = Array(a), Array(b)

        print_size_header(n)
        rows = (
            ("add", lambda a=a, b=b: a + b, lambda d_a=d_a, d_b=d_b: d_a + d_b),
            ("sub", lambda a=a, b=b: a - b, lambda d_a=d_a, d_b=d_b: d_a - d_b),
            ("mul", lambda a=a, b=b: a * b, lambda d_a=d_a, d_b=d_b: d_a * d_b),
            ("div", lambda a=a, b=b: a / b, lambda d_a=d_a, d_b=d_b: d_a / d_b),
            ("neg", lambda a=a: -a, lambda d_a=d_a: -d_a),
            ("abs", lambda a=a: abs(a), lambda d_a=d_a: abs(d_a)),
            ("pow", lambda a=a: a ** 2.0, lambda d_a=d_a: d_a ** 2.0),
        )
        for op, native_fn, wrapped_fn in rows:
            n_us = time_us_safe(case, native_fn)
            w_us = time_us_safe(case, wrapped_fn)
            print(fmt_row(op, n_us, w_us))
        print()
    print()


def main() -> None:
    print_preamble("Array operator overhead vs native frameworks")
    cases = discover_backends(only=parse_backends_arg())
    print(f"available backends: {', '.join(c.name for c in cases)}\n")
    for case in cases:
        _bench_case(case)


if __name__ == "__main__":
    main()
