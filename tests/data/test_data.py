import torch
import pytest
from materials_toolkit.data import StructureData


def compare_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype:
        return False

    if a.shape != b.shape:
        return False

    return (a == b).all().item()


def test_init():
    data = StructureData()
    assert compare_tensor(data.periodic, torch.tensor([False]))
    assert compare_tensor(data.num_atoms, torch.tensor([0]))
    assert compare_tensor(data.num_edges, torch.tensor([0]))
    assert compare_tensor(data.batch_atoms, torch.tensor([], dtype=torch.long))
    assert compare_tensor(data.batch_edges, torch.tensor([], dtype=torch.long))

    pos = torch.randn(12, 3)
    z = torch.randint(1, 93, (12,))
    data = StructureData(pos=pos, z=z)

    assert compare_tensor(data.pos, pos)
    assert compare_tensor(data.z, z)
    assert compare_tensor(data.periodic, torch.tensor([False]))
    assert compare_tensor(data.num_atoms, torch.tensor([12]))
    assert compare_tensor(data.num_edges, torch.tensor([0]))
    assert compare_tensor(data.batch_atoms, torch.zeros(12, dtype=torch.long))
    assert compare_tensor(data.batch_edges, torch.tensor([], dtype=torch.long))

    pos = torch.randn(12, 3)
    cell = torch.randn(1, 3, 3)
    z = torch.randint(1, 93, (12,))
    data = StructureData(pos=pos, z=z, cell=cell)

    assert compare_tensor(data.pos, pos)
    assert compare_tensor(data.z, z)
    assert compare_tensor(data.cell, cell)
    assert compare_tensor(data.periodic, torch.tensor([True]))
    assert compare_tensor(data.num_atoms, torch.tensor([12]))
    assert compare_tensor(data.num_edges, torch.tensor([0]))
    assert compare_tensor(data.batch_atoms, torch.zeros(12, dtype=torch.long))
    assert compare_tensor(data.batch_edges, torch.tensor([], dtype=torch.long))

    edge_index = torch.randint(0, 12, (2, 64))
    target_cell = torch.randint(-3, 4, (64, 3))
    data = StructureData(
        pos=pos, z=z, cell=cell, edge_index=edge_index, target_cell=target_cell
    )

    assert compare_tensor(data.edge_index, edge_index)
    assert compare_tensor(data.target_cell, target_cell)

    num_triplets = torch.randint(4, 1024, tuple()).item()
    triplet_index = torch.randint(0, edge_index.shape[1], (2, num_triplets)).long()

    num_quadruplets = torch.randint(4, 1024, tuple()).item()
    quadruplet_index = torch.randint(
        0, edge_index.shape[1], (3, num_quadruplets)
    ).long()

    data = StructureData(
        z=z,
        pos=pos,
        cell=cell,
        edge_index=edge_index,
        target_cell=target_cell,
        triplet_index=triplet_index,
        quadruplet_index=quadruplet_index,
    )

    assert compare_tensor(data.pos, pos)
    assert compare_tensor(data.z, z)
    assert compare_tensor(data.cell, cell)
    assert compare_tensor(data.periodic, torch.tensor([True]))
    assert compare_tensor(data.num_atoms, torch.tensor([12]))
    assert compare_tensor(data.num_edges, torch.tensor([64]))
    assert compare_tensor(data.batch_atoms, torch.zeros(12, dtype=torch.long))
    assert compare_tensor(data.batch_edges, torch.zeros(64, dtype=torch.long))
    assert compare_tensor(data.edge_index, edge_index)
    assert compare_tensor(data.target_cell, target_cell)
    assert compare_tensor(data.num_triplets, torch.tensor([num_triplets]))
    assert compare_tensor(data.num_quadruplets, torch.tensor([num_quadruplets]))
    assert compare_tensor(data.triplet_index, triplet_index)
    assert compare_tensor(data.quadruplet_index, quadruplet_index)


def test_check_size():
    pos = torch.randn(12, 3)
    cell = torch.randn(1, 3, 3)
    z = torch.randint(1, 93, (24,))

    with pytest.raises(AssertionError):
        StructureData(pos=pos, z=z, cell=cell)

    z = torch.randint(1, 93, (12,))
    edge_index = torch.randint(0, 12, (2, 64))
    target_cell = torch.randint(-3, 4, (63, 3))

    with pytest.raises(AssertionError):
        StructureData(
            pos=pos, z=z, cell=cell, edge_index=edge_index, target_cell=target_cell
        )