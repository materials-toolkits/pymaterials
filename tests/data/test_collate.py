from typing import Iterator
import pytest
import torch

from materials_toolkit.data import StructureData, batching, Batching
from materials_toolkit.data.collate import *
from materials_toolkit.data.collate import SelectableTensor

import json


def test_selectable_interface():
    with pytest.raises(TypeError):
        SelectableTensor()

    with pytest.raises(TypeError):
        SelectableTensorMaping()

    class TestSelectable(SelectableTensor):
        pass

    with pytest.raises(TypeError):
        TestSelectable()

    class TestSelectable(SelectableTensor):
        def __getitem__(self, args: int | torch.LongTensor | tuple) -> torch.Tensor:
            return torch.tensor([])

        def index_select(self, dim: int, index: torch.LongTensor) -> torch.Tensor:
            return torch.tensor([])

        @property
        def shape(self) -> tuple:
            return tuple()

    TestSelectable()

    class TestSelectableMaping(SelectableTensorMaping):
        pass

    with pytest.raises(TypeError):
        TestSelectableMaping()

    class TestSelectableMaping(SelectableTensorMaping):
        def __getitem__(self, key: str) -> SelectableTensor:
            return TestSelectable()

        def __iter__(self) -> Iterator[str]:
            return

        def __len__(self) -> int:
            return 0

    TestSelectableMaping()


batching_config = {
    "scalar": Batching(),
    "matrix": Batching(shape=(1, 3, 3)),
    "variable_size_vector": Batching(shape=("vector_size",)),
    "vector_size": Batching(dtype=torch.long),
    "struct_idx": Batching(inc=1, shape=("vector_size",), dtype=torch.long, default=0),
    "index_size": Batching(dtype=torch.long),
    "index_vector_element": Batching(
        inc="vector_size", shape=("index_size",), dtype=torch.long
    ),
}
batch_ground_truth = {
    "scalar": torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32),
    "matrix": torch.tensor(
        [
            [[1.11, 1.12, 1.13], [1.21, 1.22, 1.23], [1.31, 1.32, 1.33]],
            [[2.11, 2.12, 2.13], [2.21, 2.22, 2.23], [2.31, 2.32, 2.33]],
            [[3.11, 3.12, 3.13], [3.21, 3.22, 3.23], [3.31, 3.32, 3.33]],
        ],
        dtype=torch.float32,
    ),
    "vector_size": torch.tensor([2, 5, 3], dtype=torch.long),
    "variable_size_vector": torch.tensor(
        [1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3], dtype=torch.float32
    ),
    "struct_idx": torch.tensor([0, 0, 1, 1, 1, 1, 1, 2, 2, 2], dtype=torch.long),
    "index_size": torch.tensor([4, 2, 5], dtype=torch.long),
    "index_vector_element": torch.tensor(
        [0, 0, 1, 0, 3, 4, 7, 8, 8, 8, 7],
        dtype=torch.long,
    ),
}


@batching(reset=True, **batching_config)
class CustomData(StructureData):
    pass


custom_batch = CustomData(**batch_ground_truth)

custom_structures = [
    CustomData(
        scalar=torch.tensor([1.1], dtype=torch.float),
        matrix=torch.tensor(
            [[[1.11, 1.12, 1.13], [1.21, 1.22, 1.23], [1.31, 1.32, 1.33]]]
        ),
        variable_size_vector=torch.tensor([1.1, 1.2], dtype=torch.float32),
        index_vector_element=torch.tensor([0, 0, 1, 0], dtype=torch.long),
    ),
    CustomData(
        scalar=torch.tensor([2.2], dtype=torch.float),
        matrix=torch.tensor(
            [[[2.11, 2.12, 2.13], [2.21, 2.22, 2.23], [2.31, 2.32, 2.33]]]
        ),
        variable_size_vector=torch.tensor(
            [2.1, 2.2, 2.3, 2.4, 2.5], dtype=torch.float32
        ),
        index_vector_element=torch.tensor([1, 2], dtype=torch.long),
    ),
    CustomData(
        scalar=torch.tensor([3.3], dtype=torch.float),
        matrix=torch.tensor(
            [[[3.11, 3.12, 3.13], [3.21, 3.22, 3.23], [3.31, 3.32, 3.33]]]
        ),
        variable_size_vector=torch.tensor([3.1, 3.2, 3.3], dtype=torch.float32),
        index_vector_element=torch.tensor([0, 1, 1, 1, 0], dtype=torch.long),
    ),
]


def test_get_indexing():
    indexing = get_indexing(batch_ground_truth, batching_config)

    assert (
        (indexing["num_structures"] == torch.tensor([3], dtype=torch.long)).all().item()
    )
    assert (
        (indexing["vector_size"] == torch.tensor([0, 2, 7, 10], dtype=torch.long))
        .all()
        .item()
    )
    assert (
        (indexing["index_size"] == torch.tensor([0, 4, 6, 11], dtype=torch.long))
        .all()
        .item()
    )

    indexing = get_indexing(custom_batch)

    assert (
        (indexing["num_structures"] == torch.tensor([3], dtype=torch.long)).all().item()
    )
    assert (
        (indexing["vector_size"] == torch.tensor([0, 2, 7, 10], dtype=torch.long))
        .all()
        .item()
    )
    assert (
        (indexing["index_size"] == torch.tensor([0, 4, 6, 11], dtype=torch.long))
        .all()
        .item()
    )


def test_collate():
    batch = collate(custom_structures)

    for key in batch_ground_truth.keys():
        assert (batch[key] - batch_ground_truth[key]).abs().max() < 1e-6

    keys = ["scalar", "index_vector_element", "index_size", "struct_idx", "vector_size"]
    batch = collate(custom_structures, keys=["scalar", "index_vector_element"])

    assert set(batch.keys) == set(keys + ["num_structures"])

    for key in keys:
        assert (batch[key] - batch_ground_truth[key]).abs().max() < 1e-6

    keys = ["matrix", "variable_size_vector", "struct_idx", "vector_size"]
    batch = collate(custom_structures, keys=["matrix", "variable_size_vector"])

    assert set(batch.keys) == set(keys + ["num_structures"]), f"{batch.keys}"

    for key in keys:
        assert (
            batch[key] - batch_ground_truth[key]
        ).abs().max() < 1e-6, f"{batch[key]} {batch_ground_truth[key]}"


def compare_structure_list(a: List[StructureData], b: List[StructureData]):
    assert len(a) == len(b)
    for struct_a, struct_b in zip(a, b):
        for key in struct_a.keys:
            assert (struct_a[key] - struct_b[key]).abs().max() < 1e-6


def test_separate():
    # to list of struct
    structures = separate(custom_batch)
    compare_structure_list(custom_structures, structures)

    structures = separate(custom_batch, idx=torch.tensor([0, 1, 2]))
    compare_structure_list(custom_structures, structures)

    structures = separate(custom_batch, idx=torch.tensor([0, 2]))
    compare_structure_list(custom_structures[::2], structures)

    structures = separate(custom_batch, idx=torch.tensor([1]))
    compare_structure_list(custom_structures[1::2], structures)

    structures = separate(custom_batch, idx=1)
    compare_structure_list(custom_structures[1::2], structures)

    # to iterator
    structures = list(separate(custom_batch, result="iterator"))
    compare_structure_list(custom_structures, structures)

    structures = list(
        separate(custom_batch, idx=torch.tensor([0, 1, 2]), result="iterator")
    )
    compare_structure_list(custom_structures, structures)

    structures = list(
        separate(custom_batch, idx=torch.tensor([0, 2]), result="iterator")
    )
    compare_structure_list(custom_structures[::2], structures)

    structures = list(separate(custom_batch, idx=torch.tensor([1]), result="iterator"))
    compare_structure_list(custom_structures[1::2], structures)

    structures = list(separate(custom_batch, idx=1, result="iterator"))
    compare_structure_list(custom_structures[1::2], structures)

    # to batch
    # structures = list(separate(custom_batch, result="batch"))
    # compare_structure_list(collate(custom_structures), structures)

    structures = separate(custom_batch, idx=torch.tensor([0, 1, 2]), result="batch")
    compare_structure_list([collate(custom_structures)], [structures])

    structures = separate(custom_batch, idx=torch.tensor([0, 2]), result="batch")
    compare_structure_list([collate(custom_structures[::2])], [structures])
