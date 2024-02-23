"""
The `data` module is designed to store chemical data in a convenient way for machine learning models implemented with Pytorch. This module provides data structures for storing molecules, periodic crystals and their associated graphs. This module facilitates batching of multiple chemical structures, but also provides commonly used datasets.
"""

from .dataset import HDF5Dataset
from .base import StructureData, Batching, batching
from .loader import StructureLoader
from . import datasets

__all__ = [
    "HDF5Dataset",
    "StructureData",
    "Batching",
    "batching",
    "datasets",
    "StructureLoader",
]
