import torch
import torch.nn.functional as F

from torch_scatter import scatter_add

from typing import Tuple, Union, Optional

from .utils import shape, assert_tensor_match, build_shapes
from .meshgrid import sparse_meshgrid


class TripletsBuilder:
    def __init__(
        self,
        num_atoms: torch.LongTensor,
        edge_index: torch.LongTensor,
    ):
        self.edges = edge_index

        self.num_atoms = num_atoms
        self.struct_idx = torch.arange(self.shapes.b, device=self.shapes.device)
        self.batch = self.struct_idx.repeat_interleave(num_atoms)

        self.triplets = None

    def build(self):
        n_atoms = self.num_atoms.sum()

        num_edges = scatter_add(
            torch.ones_like(self.edges[0]),
            self.edges[0],
            dim=0,
            dim_size=n_atoms,
        )

        i_triplets, j_triplets = sparse_meshgrid(num_edges)

        mask = i_triplets != j_triplets
        i_triplets = i_triplets[mask]
        j_triplets = j_triplets[mask]

        self.triplets_index = torch.stack((i_triplets, j_triplets), dim=0)
