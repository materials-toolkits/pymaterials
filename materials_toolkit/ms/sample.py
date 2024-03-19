from collections.abc import Sequence
from typing import Iterator
from torch_geometric.data import InMemoryDataset, Data

import torch
import torch.nn.functional as F
import numpy as np

import os
import json
from typing import List

from ase.data import chemical_symbols

symboles = {s: z for z, s in enumerate(chemical_symbols)}


class SystemDataset(InMemoryDataset):
    def __init__(
        self,
        system: List[str],
        n: int = 16,
        multiple: int = 4,
        sample_per_compositon: int = 128,
    ):
        assert len(system) in (2, 3, 4)
        self.system = torch.tensor([symboles[sym] for sym in system], dtype=torch.long)

        comp_list = []

        if len(system) == 2:
            for i in range(n):
                c = torch.tensor([i, n - i])
                c //= torch.gcd(c[[0]], c[[1]])

                for mul in range(multiple):
                    comp_list.append(c * (mul + 1))
        elif len(system) == 3:
            for i in range(n):
                for j in range(n - i + 1):
                    c = torch.tensor([i, j, n - i - j])
                    div = torch.gcd(torch.gcd(c[[0]], c[[1]]), c[[2]])
                    c //= div

                    for mul in range(multiple):
                        comp_list.append(c * (mul + 1))
        else:
            for i in range(n):
                for j in range(n - i + 1):
                    for k in range(n - i - j + 1):
                        c = torch.tensor([i, j, k, n - i - j - k])
                        div = torch.gcd(
                            torch.gcd(torch.gcd(c[[0]], c[[1]]), c[[2]]), c[[3]]
                        )
                        c //= div

                        for mul in range(multiple):
                            comp_list.append(c * (mul + 1))

        self.compositions = torch.stack(comp_list * sample_per_compositon, dim=0)

        self.transform = None
        self.pre_transform = None
        self.pre_filter = None

    def download(self):
        pass

    def process(self):
        pass

    def get_num_atoms(self) -> torch.LongTensor:
        return self.compositions.sum(dim=1)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def len(self) -> int:
        return self.compositions.shape[0]

    def indices(self) -> Sequence:
        return range(self.compositions.shape[0])

    def get(self, idx: int) -> Data:
        if isinstance(idx, int):
            idx = torch.tensor([idx])

        idx = idx.flatten()

        comp = self.compositions[idx]
        num_atoms = comp.sum(dim=1)
        z = self.system.repeat(idx.shape[0]).repeat_interleave(comp.flatten())

        return Data(z=z, num_atoms=num_atoms)


from torch_geometric.loader import DataLoader

dataset = SystemDataset(["Ta", "O", "C"])
loader = DataLoader(dataset, batch_size=128, num_workers=0)

import tqdm

print(len(dataset))
for batch in tqdm.tqdm(loader):
    print(batch.z, batch.num_atoms)
