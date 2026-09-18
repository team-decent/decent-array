"""Numerical constants."""

import math

_CONSTANTS = {
    "e": math.e,
    "inf": math.inf,
    "nan": math.nan,
    "pi": math.pi,
}

e = _CONSTANTS["e"]
inf = _CONSTANTS["inf"]
nan = _CONSTANTS["nan"]
pi = _CONSTANTS["pi"]


def _reset() -> None:
    """Set/reset constants to the Python math defaults."""
    global e, inf, nan, pi  # noqa: PLW0603

    e = _CONSTANTS["e"]
    inf = _CONSTANTS["inf"]
    nan = _CONSTANTS["nan"]
    pi = _CONSTANTS["pi"]
