import torch
from torch._tensor import Tensor
import torch.nn.functional as F

from materials_toolkit.data import StructureData
from materials_toolkit.data.base import Batching

from typing import Iterable, List, Tuple, Dict, Mapping, Callable


class SelectableTensor:
    def __getitem__(self, args: int | torch.LongTensor | tuple) -> torch.Tensor:
        pass

    def index_select(self, dim: int, index: torch.LongTensor) -> torch.Tensor:
        pass


class SelectableTensorMaping(Mapping[str, SelectableTensor]):
    def __getitem__(self, __key: str) -> SelectableTensor:
        pass


def get_indexing(
    batch: Mapping[str, SelectableTensor],
    batching: Dict[str, Batching] = None,
) -> Dict[str, torch.LongTensor]:
    if batching is None:
        assert hasattr(batch, "batching"), "Batching can't be inferred automatically."
        batching = batch.batching

    indexing = {}

    if isinstance(batch.keys, Iterable):
        keys = batch.keys
    else:
        keys = batch.keys()

    for key in keys:
        indices_name = (batching[key].shape[batching[key].cat_dim], batching[key].inc)

        for index in indices_name:
            if isinstance(index, str) and (index not in indexing):
                indexing[index] = F.pad(getattr(batch, index).cumsum(dim=0), (1, 0))

    return indexing


def _collate_key(
    key: str, data_args: Dict[str, torch.Tensor], batching: Dict[str, Batching], incs={}
) -> torch.Tensor:
    if isinstance(data_args[key], torch.Tensor):
        return data_args[key]

    cat_dim = batching[key].cat_dim

    data = torch.cat(data_args[key], cat_dim)

    shape = batching[key].shape
    if not isinstance(shape, tuple):
        shape = (shape,)
    inc = batching[key].inc
    size = shape[cat_dim]

    if isinstance(size, str):
        size = _collate_key(size, data_args, batching, incs)

    if inc != 0:
        if inc in incs:
            calc_inc = incs[inc]
        else:
            if isinstance(inc, str):
                dim = batching[inc].cat_dim
                calc_inc = F.pad(
                    _collate_key(inc, data_args, batching, incs).cumsum(dim=dim)[:-1],
                    (1, 0),
                )
            else:
                calc_inc = torch.arange(0, len(data_args[key]) * inc, inc)

            incs[inc] = calc_inc

        calc_inc = calc_inc.repeat_interleave(size)

        indexing = tuple(
            (None if i != cat_dim else slice(None)) for i, _ in enumerate(shape)
        )
        data += calc_inc[indexing]

    data_args[key] = data

    return data


def collate(structures: List[StructureData]) -> StructureData:
    cls = structures[0].__class__
    batching = cls.batching

    keys = set.union(*(set(struct.keys) for struct in structures))
    data_args = {key: [] for key in keys}

    for struct in structures:
        for key in keys:
            data_args[key].append(getattr(struct, key))

    incs = {}
    for key in keys:
        _collate_key(key, data_args, batching, incs)

    return cls(**data_args)


def _select_by_indices(
    batch: SelectableTensorMaping,
    idx: torch.LongTensor,
    keys: Iterable[str],
    batching: Dict[str, Batching],
    indexing: Dict[str, torch.LongTensor],
) -> Dict[str, torch.LongTensor]:
    indices_storage = {}  # Use a single instance of each index to save memory.
    result = {}

    for key in keys:
        cat_dim = batching[key].cat_dim
        cat_index = batching[key].shape[cat_dim]

        if cat_index not in indices_storage:
            if isinstance(cat_index, str):
                size = indexing[cat_index][idx + 1] - indexing[cat_index][idx]

                selected_idx = torch.arange(size.sum(), dtype=torch.long)
                offset = F.pad(size[:-1].cumsum(0), (1, 0))

                offset_neg = selected_idx[offset].repeat_interleave(size)
                offset_pos = indexing[cat_index][idx].repeat_interleave(size)

                selected_idx += offset_pos - offset_neg
            else:
                selected_idx = idx.repeat_interleave(cat_index)

            indices_storage[cat_index] = selected_idx

        result[key] = batch[key].index_select(cat_dim, indices_storage[cat_index])

    return result


def _decrement_in_place(
    data: [str, torch.Tensor],
    batching: Dict[str, Batching],
    indexing: Dict[str, torch.LongTensor],
) -> None:
    for key, tensor in data.items():
        inc = batching[key].inc
        cat_dim = batching[key].cat_dim

        if inc == 0:
            continue
        elif isinstance(inc, str):
            inc_tensor = indexing[inc]
        else:
            inc_tensor = torch.arange(
                0, inc * data.shape[cat_dim], inc, dtype=torch.LongTensor
            )


def separate(
    batch: SelectableTensorMaping,
    idx: int | torch.LongTensor = None,
    batching: Dict[str, Batching] = None,
    indexing: Dict[str, torch.LongTensor] = None,
) -> List[StructureData]:
    if batching is None:
        assert hasattr(batch, "batching"), "Batching can't be inferred automatically."
        batching = batch.batching

    if indexing is None:
        indexing = get_indexing(batch, batching)

    if isinstance(batch.keys, Callable):
        keys = batch.keys()
    else:
        keys = batch.keys

    if idx is not None:
        if isinstance(idx, int):
            idx = torch.tensor([idx], dtype=torch.long)
        assert isinstance(idx, torch.LongTensor), "idx must be None, int or LongTensor"

        data = _select_by_indices(batch, idx, keys, batching, indexing)
    else:
        data = {key: batch[key].clone() for key in keys}

    _decrement_in_place(data, batching, indexing)

    print(data)

    return []
