"""
Materials-toolkit is a python package created to facilitate the development of machine learning models in materials science.
"""

try:
    import torch_scatter
except:
    raise ImportError(
        "The torch_scatter module must be installed manually (depending on your CUDA version). See the documentation for details."
    )

from . import data

__version__ = "0.1.3"

__all__ = ["data"]
