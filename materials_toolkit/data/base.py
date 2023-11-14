import torch
from torch_geometric.data import Data
from torch_geometric import loader

from typing import Any, Optional, Union, Dict


def batching(config: Dict[str, Any]):
    assert isinstance(config, dict)
    for key, value in config.items():
        assert isinstance(key, str) and isinstance(value, dict)
        assert "cat_dim" in value and "inc" in value
        assert isinstance(value["cat_dim"], int)
        assert value["inc"] == 0 or (
            isinstance(value["inc"], str)
            and (value["inc"] in config or value["inc"] in StructureData.batching)
        )

    def fn(cls):
        assert StructureData in cls.__bases__
        cls.batching.update(config)

    return fn


class StructureData(Data):
    batching = {
        "pos": {"cat_dim": 0, "inc": "num_nodes"},
        "z": {"cat_dim": 0, "inc": "num_nodes"},
        "cell": {"cat_dim": 0, "inc": 0},
        "y": {"cat_dim": 0, "inc": 0},
        "num_nodes": {"cat_dim": 0, "inc": 0},
        "periodic": {"cat_dim": 0, "inc": 0},
        "edge_index": {"cat_dim": 1, "inc": "num_nodes"},
        "edge_cell": {"cat_dim": 0, "inc": 0},
        "num_edges": {"cat_dim": 0, "inc": 0},
        "triplet_index": {"cat_dim": 1, "inc": "num_edges"},
        "num_triplets": {"cat_dim": 0, "inc": 0},
        "quadruplets_index": {"cat_dim": 1, "inc": "num_edges"},
        "num_quadruplets": {"cat_dim": 0, "inc": 0},
    }

    def __init__(
        self,
        pos: Optional[torch.FloatTensor] = None,
        z: Optional[torch.LongTensor] = None,
        cell: Optional[torch.FloatTensor] = None,
        y: Optional[torch.FloatTensor] = None,
        edge_index: Optional[torch.LongTensor] = None,
        edge_cell: Optional[torch.LongTensor] = None,
        triplet_index: Optional[torch.LongTensor] = None,
        quadruplets_index: Optional[torch.LongTensor] = None,
        periodic: Optional[Union[bool, torch.BoolTensor]] = False,
        **kwargs,
    ):
        if isinstance(periodic, bool):
            periodic = torch.tensor(periodic)

        if "num_nodes" not in kwargs:
            if pos is not None:
                kwargs["num_nodes"] = pos.shape[self.batching["pos"]["cat_dim"]]
            elif z is not None:
                kwargs["num_nodes"] = z.shape[self.batching["z"]["cat_dim"]]
        else:
            kwargs["num_nodes"] = 0
        if "num_edges" not in kwargs and edge_index is not None:
            kwargs["num_edges"] = edge_index.shape[
                self.batching["edge_index"]["cat_dim"]
            ]
        else:
            kwargs["num_edges"] = 0
        if "num_triplets" not in kwargs and triplet_index is not None:
            kwargs["num_triplets"] = triplet_index.shape[
                self.batching["triplet_index"]["cat_dim"]
            ]
        if "num_quadruplets" not in kwargs and quadruplets_index is not None:
            kwargs["num_quadruplets"] = quadruplets_index.shape[
                self.batching["quadruplets_index"]["cat_dim"]
            ]

        super().__init__(
            pos=pos,
            cell=cell,
            y=y,
            z=z,
            periodic=periodic,
            edge_index=edge_index,
            edge_cell=edge_cell,
            triplet_index=triplet_index,
            quadruplets_index=quadruplets_index,
            **kwargs,
        )

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> int:
        inc = self.batching.get(key, {"cat_dim": 0})["cat_dim"]

        return inc

    def has_inc(self, key: str) -> bool:
        return self.batching.get(key, {"inc": 0})["inc"] != 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> int:
        inc = self.batching.get(key, {"inc": 0})["inc"]

        if isinstance(inc, str):
            inc = getattr(self, inc)
            assert isinstance(inc, (int, torch.Tensor))

        if isinstance(inc, torch.Tensor):
            inc = inc.item()

        return inc


from torch_geometric.data import InMemoryDataset


class TestDataset(InMemoryDataset):
    def len(self) -> int:
        return 10

    def get(self, idx: int) -> StructureData:
        torch.manual_seed(idx)
        nodes = torch.randint(4, 8, (1,)).item()
        edges = torch.randint(4, 2 * nodes, (1,)).item()
        triplets = torch.randint(4, 2 * edges, (1,)).item()
        return StructureData(
            x=torch.randn((nodes, 64)),
            cell=torch.matrix_exp(0.1 * torch.randn(1, 3, 3)),
            natoms=torch.tensor(nodes),
            nedges=torch.tensor(edges),
            ntriplets=torch.tensor(triplets),
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
