"""
Shared helpers for ``bench_array.py`` and ``bench_iop.py``.

Three concerns live here so the benchmarks stay focused on the comparison logic:

* :func:`discover_backends` returns the subset of frameworks whose package is
  importable; backends with a missing optional dependency are skipped silently.
* :func:`is_compiled` / :func:`print_preamble` report whether the user is
  running against a mypyc-compiled build of ``decent_array`` or the pure-Python
  source — this materially changes overhead numbers, so the result is printed
  at the top of every run.
* :func:`time_us` / :func:`time_us_safe` wrap :mod:`timeit` to take the
  ``min`` of several auto-ranged repeats. ``min`` is the canonical choice: it
  reports the lower bound of the machine's per-call cost and is the metric
  least sensitive to background activity. A warmup call precedes timing so
  JIT-style backends (JAX) don't skew the first iteration.
"""

from __future__ import annotations

import importlib
import timeit
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SIZES: tuple[int, ...] = (10, 100, 1_000, 10_000)
REPEATS: int = 7


def _no_sync(_value: Any) -> None:  # noqa: ANN401
    """No-op sync used for synchronous backends (numpy, torch CPU, tf eager CPU)."""


def _sync_jax(value: Any) -> None:  # noqa: ANN401
    """Block until a JAX DeviceArray is materialized, unwrapping ``Array`` if needed."""
    # Imported lazily so the module can load even when decent_array isn't yet importable.
    from decent_array import Array  # noqa: PLC0415

    raw = value.value if isinstance(value, Array) else value
    raw.block_until_ready()


@dataclass(slots=True)
class BackendCase:
    """A discovered backend plus the helpers needed to drive it in a benchmark."""

    name: str
    make: Callable[[int], Any]
    sync: Callable[[Any], None]


def activate_backend(name: str) -> None:
    """Activate ``name`` as the live backend, resetting any previously active one.

    ``decent_array`` enforces a single-active-backend invariant per execution context;
    swapping between frameworks within one process requires resetting first.
    """
    from decent_array.interoperability._backend_manager import reset_backends, set_backend  # noqa: PLC0415

    reset_backends()
    set_backend(name)


def discover_backends(only: list[str] | None = None) -> list[BackendCase]:
    """Return one :class:`BackendCase` per importable framework, in a stable order.

    Args:
        only: Optional allowlist of backend names. When provided, frameworks not in the
            list are skipped entirely (their packages aren't even imported), and any
            requested name that isn't a known backend raises :class:`ValueError`.

    """
    import numpy as np  # always available — hard dependency  # noqa: PLC0415

    known = {"numpy", "pytorch", "jax", "tensorflow"}
    if only is not None:
        unknown = set(only) - known
        if unknown:
            raise ValueError(f"unknown backend(s): {sorted(unknown)}; known: {sorted(known)}")
        wanted = set(only)
    else:
        wanted = known

    cases: list[BackendCase] = []

    if "numpy" in wanted:
        cases.append(BackendCase("numpy", lambda n: np.random.rand(n), _no_sync))

    if "pytorch" in wanted:
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            pass
        else:
            cases.append(BackendCase("pytorch", lambda n: torch.from_numpy(np.random.rand(n)), _no_sync))

    if "jax" in wanted:
        try:
            import jax.numpy as jnp  # noqa: PLC0415
        except ImportError:
            pass
        else:
            cases.append(BackendCase("jax", lambda n: jnp.asarray(np.random.rand(n)), _sync_jax))

    if "tensorflow" in wanted:
        try:
            import tensorflow as tf  # noqa: PLC0415
        except ImportError:
            pass
        else:
            cases.append(BackendCase("tensorflow", lambda n: tf.constant(np.random.rand(n)), _no_sync))

    return cases


def parse_backends_arg() -> list[str] | None:
    """Parse the shared ``--backends`` CLI flag; returns ``None`` if not given."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--backends",
        type=str,
        default=None,
        help="comma-separated allowlist of backends (numpy,pytorch,jax,tensorflow); default = all available",
    )
    args = parser.parse_args()
    if args.backends is None:
        return None
    return [b.strip() for b in args.backends.split(",") if b.strip()]


def is_compiled() -> tuple[bool, str]:
    """Return ``(True, path)`` if the Array module loaded from a ``.so``/``.pyd``, else ``(False, .py path)``."""
    module = importlib.import_module("decent_array._array")
    path = module.__file__ or "<unknown>"
    return path.endswith((".so", ".pyd")), path


def print_preamble(title: str) -> None:
    compiled, path = is_compiled()
    print(f"# {title}\n")
    print(f"decent_array compiled: {'yes' if compiled else 'no'}")
    print(f"  Array loaded from: {path}")
    print(f"  timing: min over {REPEATS} repeats, iterations per repeat auto-tuned to ~0.2s\n")


def time_us(case: BackendCase, fn: Callable[[], Any]) -> float:
    """Per-call runtime in µs; min over :data:`REPEATS` measurements with autoranged N."""

    def runner() -> None:
        case.sync(fn())

    runner()  # warmup — material for JAX's first-call compilation
    timer = timeit.Timer(runner)
    n, _ = timer.autorange()
    times = timer.repeat(repeat=REPEATS, number=n)
    return (min(times) / n) * 1e6


def time_us_safe(case: BackendCase, fn: Callable[[], Any]) -> float | None:
    """Like :func:`time_us` but returns ``None`` if ``fn`` raises (e.g. TF 1D matmul)."""
    try:
        return time_us(case, fn)
    except Exception:  # noqa: BLE001
        return None


def fmt_row(op: str, native_us: float | None, wrapped_us: float | None) -> str:
    if native_us is None or wrapped_us is None:
        return f"   {op:<8}  {'n/a':>13}  {'n/a':>13}   {'n/a':>8}"
    ratio = wrapped_us / native_us if native_us > 0 else float("inf")
    return f"   {op:<8}  {native_us:>10.3f} µs  {wrapped_us:>10.3f} µs   {ratio:>6.2f}x"


def print_size_header(n: int) -> None:
    print(f"size = {n:_}")
    print(f"   {'op':<8}  {'native':>13}  {'wrapped':>13}   {'overhead':>8}")
