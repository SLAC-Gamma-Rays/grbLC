"""Top-level package for grbLC submodule."""

__author__ = """Sam Young"""
__email__ = "youngsam@sas.upenn.edu"
__version__ = "0.0.0"

from . import convert, fitting, search  # noqa F401
from .fitting import Model, Lightcurve
