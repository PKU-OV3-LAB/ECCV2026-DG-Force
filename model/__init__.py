from .registry import Registry, MODELS, DATASETS, POSTFUNCS

from .datasets import *  # Registry all Datasets to the DATASETS register
from .dg_force import dg_force

__version__ = "0.1.42"

__all__ = [
    "__version__",
    "MODELS",
    "DATASETS",
    "POSTFUNCS",
    "dg_force",
]
