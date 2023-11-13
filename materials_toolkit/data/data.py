import torch
from torch_geometric import data
from torch_geometric import loader

from typing import Any, Optional, Union


class StructureData(data.Data):
    def __init__(
        self,
        x: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.LongTensor] = None,
        edge_dst_offset: Optional[torch.LongTensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        triplet_index: Optional[torch.LongTensor] = None,
        triplet_attr: Optional[torch.Tensor] = None,
        quatruplet_index: Optional[torch.LongTensor] = None,
        quatruplet_attr: Optional[torch.Tensor] = None,
        periodic: Optional[Union[bool, torch.BoolTensor]] = False,
        **kwargs,
    ):
        super().__init__(
            x=x,
            pos=pos,
            cell=cell,
            y=y,
            z=z,
            edge_index=edge_index,
            edge_dst_offset=edge_dst_offset,
            edge_attr=edge_attr,
            triplet_index=triplet_index,
            triplet_attr=triplet_attr,
            quatruplet_index=quatruplet_index,
            quatruplet_attr=quatruplet_attr,
            periodic=periodic,
            **kwargs,
        )

    @property
    def num_edges(self):
        if hasattr(self, "edge_index") and self.edge_index is not None:
            return self.edge_index.shape[1]
        return 0

    @property
    def num_triplets(self):
        if hasattr(self, "triplet_index") and self.triplet_index is not None:
            return self.triplet_index.shape[1]
        return 0

    @property
    def num_quadruplets(self):
        if hasattr(self, "quatruplet_index") and self.quatruplet_index is not None:
            return self.quatruplet_index.shape[1]
        return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in kwargs:
            if key == "edge_index":
                return self.num_nodes
            if key == "triplet_index":
                return self.num_edges
            if key == "quatruplet_index":
                return self.num_edges

        return super().__inc__(key, value, *args, **kwargs)


class TestDataset(data.InMemoryDataset):
    def len(self) -> int:
        return 10

    def get(self, idx: int) -> StructureData:
        torch.manual_seed(idx)
        nodes = torch.randint(4, 8, (1,)).item()
        edges = torch.randint(4, 2 * nodes, (1,)).item()
        triplets = torch.randint(4, 2 * edges, (1,)).item()
        return StructureData(
            x=torch.randn((nodes, 64)),
            pos=torch.rand((nodes, 3)),
            cell=torch.matrix_exp(0.1 * torch.randn(1, 3, 3)),
            edge_index=torch.randint(0, nodes - 1, (2, edges)),
            triplet_index=torch.randint(0, edges - 1, (2, triplets)),
            y=torch.tensor([[nodes, edges, triplets]]),
        )


if __name__ == "__main__":
    batch_size = 4
    dataset = TestDataset()
    loader = loader.DataLoader(dataset, batch_size=batch_size)

    for batch in loader:
        break

    for idx in range(batch_size):
        assert (dataset[idx].edge_index == batch.get_example(idx).edge_index).all()
        assert (
            dataset[idx].triplet_index == batch.get_example(idx).triplet_index
        ).all()
