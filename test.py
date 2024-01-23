import torch
import torch.nn as nn

from materials_toolkit.data.datasets import MaterialsProject
from materials_toolkit.data.filter import *
from materials_toolkit.data.collate import *
from torch_geometric.loader import DataLoader

from pymatgen.core.periodic_table import Element


def generate_data_list(count=128):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    structures = []
    for _ in range(count):
        n = torch.randint(4, 8, tuple()).item()

        pos = torch.rand(n, 3).float()
        z = torch.randint(1, 128, (n,)).long()
        y = torch.randn(1, 7).float()
        periodic = torch.randint(0, 2, tuple()).bool().item()

        n_edges = torch.randint(4, 16, tuple()).item()
        edge_index = torch.randint(0, n, (2, n_edges)).long()
        if periodic:
            cell = torch.matrix_exp(torch.rand(1, 3, 3)).float()
            target_cell = torch.randint(-3, 4, (n_edges, 3)).long()
        else:
            cell = None
            target_cell = None

        n_triplets = torch.randint(4, 16, tuple()).item()
        triplet_index = torch.randint(0, n_edges, (2, n_triplets)).long()

        num_quadruplets = torch.randint(4, 16, tuple()).item()
        quadruplet_index = torch.randint(0, n_edges, (3, num_quadruplets)).long()

        struct = StructureData(
            z=z,
            pos=pos,
            cell=cell,
            y=y,
            edge_index=edge_index,
            target_cell=target_cell,
            triplet_index=triplet_index,
            quadruplet_index=quadruplet_index,
            periodic=periodic,
        )
        structures.append(struct)

    return structures


lst = generate_data_list(count=4)
for struct in lst:
    print(struct.edge_index)
batch = collate(lst)
print(batch)
print(batch.num_atoms, batch.num_atoms.sum().item())
print(batch.num_edges, batch.num_edges.sum().item())
print(batch.num_triplets, batch.num_triplets.sum().item())
print(batch.num_quadruplets, batch.num_quadruplets.sum().item())

print(batch.edge_index)

print(get_indexing(batch))

print(separate(batch, idx=torch.arange(4, dtype=torch.long)))

exit(0)
import shutil

shutil.rmtree("data/mp/processed")

mp = MaterialsProject(
    "data/mp",
    pre_filter=FilterAtoms(included=[2, 10, 18, 36, 54, 86, 118]),
)
filter = SequentialFilter(FilterAtoms(excluded=[8]), FilterNumberOfAtoms(max=6))

loader = DataLoader(mp, batch_size=32)

print(len(mp))

for structs in loader:
    print(structs)
    print(filter(structs))
    break
