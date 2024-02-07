import torch

from materials_toolkit.data import StructureData
from materials_toolkit.graph.fully_connected import indices_like, fully_connected


def graph_cutoff(
    batch: StructureData,
    cutoff: float,
    mask: torch.BoolTensor = None,
    self_loop: bool = True,
    directed: bool = True,
    epsilon: float = 1e-6,
):
    edges = fully_connected(
        batch.num_atoms, mask=mask, self_loop=self_loop, directed=directed
    )

    distance = torch.dist(batch.pos[edges[0]], batch.pos[edges[1]])

    mask = distance <= (cutoff + epsilon)

    edges = edges[:, mask]

    return edges


def graph_knn(
    batch: StructureData,
    knn: int,
    mask: torch.BoolTensor = None,
    self_loop: bool = True,
    epsilon: float = 1e-6,
):
    edges = fully_connected(
        batch.num_atoms, mask=mask, self_loop=self_loop, directed=True
    )

    distance = torch.dist(batch.pos[edges[0]], batch.pos[edges[1]])

    distance, perm = distance.sort()
    edges = edges[:, perm]

    perm = edges[0].argsort(stable=True)

    distance = distance[:, perm]
    edges = edges[:, perm]
