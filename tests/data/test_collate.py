from typing import Iterator
import pytest
import torch

from materials_toolkit.data.collate import *
from materials_toolkit.data.collate import SelectableTensor


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


def test_get_indexing():
    batching = {
        "scalar": Batching(),
        "matrix": Batching(shape=(3, 3)),
        "variable_size_vector": Batching(shape=("vector_size",)),
        "vector_size": Batching(dtype=torch.long),
        "struct_idx": Batching(inc=1, dtype=torch.long, default=0),
        "index_vector_element": Batching(
            inc="vector_size", shape=("vector_size",), dtype=torch.long
        ),
    }
    batch = {}
