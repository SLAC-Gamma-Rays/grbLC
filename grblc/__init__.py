"""Top-level package for grbLC submodule."""

__author__ = """Sam Young"""
__email__ = "youngsam@sas.upenn.edu"
__version__ = "0.0.8"

from . import convert, fitting, search  # noqa F401
from .convert import convertGRB, get_dir, set_dir, toFlux
from .fitting import Lightcurve, Model, OutlierPlot
from .fitting.model import Models
from .search import ads, gcn
