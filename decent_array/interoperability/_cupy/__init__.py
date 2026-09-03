"""CuPy backend package; importing it triggers backend registration."""

from .cupy_backend import CupyBackend

__all__ = ["CupyBackend"]
