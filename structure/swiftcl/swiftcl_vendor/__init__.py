from importlib.metadata import version

from .cl import ClComp

__version__ = version(__package__)
__all__ = ["ClComp"]
