import torch
import torch.utils.data
import torch_geometric.loader

from typing import Optional, List


class DataLoader(torch_geometric.loader.DataLoader):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        torch.utils.data.DataLoader.__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, exclude_keys),
            **kwargs,
        )
