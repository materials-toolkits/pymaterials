import torch
import torch.utils.data
import torch_geometric.loader

from typing import Optional, List

from materials_toolkit.data.base import StructureData
from materials_toolkit.data.collate import collate


class Collater:
    def __init__(self, exclude_keys=None, dataset=None):
        self.dataset = dataset
        self.exclude_keys = exclude_keys

    def __call__(self, batch) -> StructureData:
        if isinstance(batch[0], int | torch.LongTensor):
            # batch = sorted(batch)
            return self.dataset.get(torch.tensor(batch))
        elif isinstance(batch[0], StructureData):
            return collate(batch)

        raise Exception("unkown type of data")


class StructureLoader(torch_geometric.loader.DataLoader):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        exclude_keys: Optional[List[str]] = None,
        read_multiple: bool = True,
        **kwargs,
    ):
        self.exclude_keys = exclude_keys

        if read_multiple:
            data = torch.arange(len(dataset), dtype=torch.long)
        else:
            data = dataset

        torch.utils.data.DataLoader.__init__(
            self,
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=Collater(exclude_keys, dataset=dataset),
            **kwargs,
        )
