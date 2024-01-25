import torch
from torch._tensor import Tensor
import torch.nn.functional as F

from materials_toolkit.data import StructureData
from materials_toolkit.data.base import Batching

from typing import Iterable, List, Tuple, Dict, Mapping, Callable, Optional


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


def _select_indexing(
    idx: torch.LongTensor,
    keys: Iterable[str],
    batching: Dict[str, Batching],
    indexing: Dict[str, torch.LongTensor],
):
    keys_indexing = set()
    for key in keys:
        cat_dim = batching[key].cat_dim
        cat_index = batching[key].shape[cat_dim]

        if isinstance(cat_index, str):
            keys_indexing.add(cat_index)

    select_indexing = {}
    for cat_index in keys_indexing:
        size = indexing[cat_index][idx + 1] - indexing[cat_index][idx]

        selected_idx = torch.arange(size.sum(), dtype=torch.long)
        offset = F.pad(size[:-1].cumsum(0), (1, 0))

        offset_idx = indexing[cat_index][idx] - selected_idx[offset]

        selected_idx += offset_idx.repeat_interleave(size)

        select_indexing[cat_index] = selected_idx

    return select_indexing


def _select_and_decrement(
    batch: SelectableTensorMaping,
    keys: Iterable[str],
    batching: Dict[str, Batching],
    idx: torch.LongTensor,
    indexing: Dict[str, torch.LongTensor],
    select_indexing: Optional[Dict[str, torch.LongTensor]] = None,
) -> Dict[str, torch.LongTensor]:
    result = {}

    for key in keys:
        inc = batching[key].inc
        cat_dim = batching[key].cat_dim
        cat_index = batching[key].shape[cat_dim]

        if isinstance(cat_index, int):
            a, b = torch.meshgrid(
                (cat_index * idx, torch.arange(0, cat_index, dtype=torch.long))
            )
            current_idx = (a + b).flatten()
            data = batch[key].index_select(cat_dim, current_idx)
        elif select_indexing is None:
            data = batch[key].clone()
        else:
            current_idx = select_indexing[cat_index]
            data = batch[key].index_select(cat_dim, current_idx)

        if inc != 0:
            if isinstance(inc, str):
                offset = indexing[inc]
            else:
                offset = inc * idx

            size = indexing[cat_index][idx + 1] - indexing[cat_index][idx]
            index = torch.arange(size.shape[0], dtype=torch.long).repeat_interleave(
                size
            )

            selection = tuple(
                None if i != cat_dim else index for i, s in enumerate(data.shape)
            )
            data -= offset[selection]

        result[key] = data

    return result


def _to_list(
    cls: type,
    data: Dict[str, torch.Tensor],
    batching: Dict[str, Batching],
    indexing: Dict[str, torch.LongTensor],
) -> List[StructureData]:
    result = []
    for i in range(data["periodic"].shape[0]):
        kwargs = {}

        for key in data.keys():
            cat_dim = batching[key].cat_dim
            cat_index = batching[key].shape[cat_dim]

            if isinstance(cat_index, int):
                idx = torch.arange(cat_index * i, cat_index * (i + 1), dtype=torch.long)
            else:
                idx = torch.arange(
                    indexing[cat_index][i], indexing[cat_index][i + 1], dtype=torch.long
                )
            kwargs[key] = data[key].index_select(cat_dim, idx)

        result.append(cls(**kwargs))
    return result


def separate(
    batch: SelectableTensorMaping,
    idx: int | torch.LongTensor = None,
    batching: Dict[str, Batching] = None,
    indexing: Dict[str, torch.LongTensor] = None,
    cls: type = None,
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

    if cls is None:
        cls = batch.__class__

    if idx is not None:
        if isinstance(idx, int):
            idx = torch.tensor([idx], dtype=torch.long)
        assert isinstance(idx, torch.LongTensor), "idx must be None, int or LongTensor"
    else:
        idx = torch.arange(batch["periodic"].shape[0], dtype=torch.long)

    select_indexing = _select_indexing(idx, keys, batching, indexing)

    data = _select_and_decrement(batch, keys, batching, idx, indexing, select_indexing)

    return _to_list(cls, data, batching, indexing)
