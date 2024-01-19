import torch
import torch.nn as nn
from torch_geometric.data import Batch

from materials_toolkit.data import StructureData

from typing import List
from abc import ABCMeta, abstractmethod


class Filter(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, struct: StructureData) -> bool | torch.BoolTensor:
        pass


class SequentialFilter(nn.ModuleList, Filter):
    def forward(self, struct: StructureData) -> bool | torch.BoolTensor:
        super().forward(struct)


class FilterNumberOfAtoms(Filter):
    max_atoms = torch.iinfo(torch.long).max

    def __init__(self, min: int = 0, max: int = max_atoms):
        super().__init__()

        self.min = nn.Parameter(
            torch.tensor(min, dtype=torch.long), requires_grad=False
        )
        self.max = nn.Parameter(
            torch.tensor(max, dtype=torch.long), requires_grad=False
        )

    def forward(self, struct: StructureData) -> bool | torch.BoolTensor:
        if isinstance(struct, Batch):
            return (self.min <= struct.num_atoms) & (struct.num_atoms <= self.max)

        return self.min.item() <= struct.num_atoms.item() <= self.max.item()


class FilterAtoms(Filter):
    def __init__(
        self,
        included: torch.LongTensor | List[int] = None,
        excluded: torch.LongTensor | List[int] = [],
    ):
        super().__init__()

        if included is None:
            included = torch.arange(128, dtype=torch.long)
        elif not isinstance(included, torch.Tensor):
            included = torch.tensor(included, dtype=torch.long)

        if not isinstance(excluded, torch.Tensor):
            excluded = torch.tensor(excluded, dtype=torch.long)

        included_mask = torch.full((128,), False, dtype=torch.bool)
        included_mask[included.long()] = True

        excluded_mask = torch.full((128,), False, dtype=torch.bool)
        excluded_mask[excluded.long()] = True

        self.included_mask = nn.Parameter(included_mask, requires_grad=False)
        self.excluded_mask = nn.Parameter(excluded_mask, requires_grad=False)

    def forward(self, struct: StructureData) -> bool | torch.BoolTensor:
        if isinstance(struct, Batch):
            return (self.min <= struct.num_atoms) & (struct.num_atoms <= self.max)

        return (self.included_mask[struct.z]).any() & (
            ~self.excluded_mask[struct.z]
        ).all()


class FilterNobleGas(FilterAtoms):
    def __init__(self):
        super().__init__(excluded=[2, 10, 18, 36, 54, 86, 118])
