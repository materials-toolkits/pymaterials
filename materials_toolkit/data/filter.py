import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_scatter import scatter_min, scatter_max

from materials_toolkit.data import StructureData

from typing import List
from abc import ABCMeta, abstractmethod


class Filter(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, struct: StructureData) -> bool:
        pass


class SequentialFilter(nn.ModuleList, Filter):
    def forward(self, struct: StructureData) -> bool:
        assert struct.num_structures == 1

        for filter in self:
            if not filter(struct):
                return False

        return True


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
        included = self.included_mask[struct.z]
        excluded = self.excluded_mask[struct.z]

        if isinstance(struct, Batch):
            return scatter_max(included, struct.batch_atoms) & (
                scatter_min(~excluded, struct.batch_atoms)
            )

        return included.any() & (~excluded).all()


class FilterNobleGas(FilterAtoms):
    def __init__(self):
        super().__init__(excluded=[2, 10, 18, 36, 54, 86, 118])


class FilterStable(Filter):
    def __init__(self, e_above_hull: float = 0.1):
        super().__init__()

        self.threshold = nn.Parameter(
            torch.tensor(e_above_hull, dtype=torch.float32), requires_grad=False
        )

    def forward(self, struct: StructureData) -> bool | torch.BoolTensor:
        if isinstance(struct, Batch):
            return struct.energy_above_hull <= self.threshold

        return (struct.energy_above_hull <= self.threshold).item()
