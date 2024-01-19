import torch
import torch.nn as nn
from materials_toolkit.data.datasets import MaterialsProject
from materials_toolkit.data.filter import *
from torch_geometric.loader import DataLoader

from pymatgen.core.periodic_table import Element

import shutil

"""

shutil.rmtree("data/mp/processed")

mp = MaterialsProject(
    "data/mp",
    pre_filter=FilterAtoms(included=[2, 10, 18, 36, 54, 86, 118]),
)
filter = FilterAtoms(excluded=[8])

loader = DataLoader(mp, batch_size=32)

print(len(mp))

for structs in loader:
    print(structs)
    print(filter(structs))
    break
"""
