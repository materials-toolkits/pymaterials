import torch
from materials_toolkit.data import StructureData


def compare_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype:
        return False

    if a.shape != b.shape:
        return False

    return (a == b).all().item()


def test_data():
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

