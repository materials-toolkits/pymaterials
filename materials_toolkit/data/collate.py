import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

from materials_toolkit.data.base import Batching, StructureData

from typing import (
    Iterable,
    Iterator,
    List,
    Literal,
    Tuple,
    Dict,
    Mapping,
    Callable,
    Optional,
)
from abc import ABCMeta, abstractmethod, abstractproperty


class SelectableTensor(metaclass=ABCMeta):
    @abstractmethod
    def __getitem__(self, args: int | torch.LongTensor | tuple) -> torch.Tensor:
        pass

    @abstractmethod
    def index_select(self, dim: int, index: torch.LongTensor) -> torch.Tensor:
        pass

    @abstractproperty
    def shape(self) -> tuple:
        pass


class SelectableTensorMaping(Mapping[str, SelectableTensor]):
    @abstractmethod
    def __getitem__(self, key: str) -> SelectableTensor:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        return super().__len__()


def get_indexing(
    batch: Mapping[str, SelectableTensor],
    batching: Dict[str, Batching] = None,
) -> Dict[str, torch.LongTensor]:
    if batching is None:
        assert hasattr(batch, "batching"), "Batching can't be inferred automatically."
        batching = batch.batching

    indexing = {}

    if "num_structures" in batch:
        indexing["num_structures"] = batch["num_structures"]
        assert isinstance(indexing["num_structures"], torch.LongTensor)
        assert indexing["num_structures"].shape == (1,)

    if isinstance(batch.keys, Iterable):
        keys = batch.keys
    else:
        keys = batch.keys()

    for key in keys:
        if key == "num_structures":
            continue

        cat_dim = batching[key].cat_dim
        cat_shape = batching[key].shape[cat_dim]

        for index in (cat_shape, batching[key].inc):
            if isinstance(index, str) and (index not in indexing):
                indexing[index] = F.pad(batch[index].cumsum(dim=0), (1, 0))

        if isinstance(cat_shape, int):
            if "num_structures" not in indexing:
                indexing["num_structures"] = torch.tensor(
                    [batch[key].shape[cat_dim] // cat_shape], dtype=torch.long
                )
            else:
                assert indexing["num_structures"].item() == (
                    batch[key].shape[cat_dim] // cat_shape
                ), f"{key} {indexing['num_structures'].item()} {batch[key].shape[cat_dim] // cat_shape} {batch[key].shape[cat_dim]} {batch[key].shape} {cat_dim} {cat_shape}"

    return indexing


def _collate_key(
    key: str,
    data_args: Dict[str, torch.Tensor | List[torch.Tensor]],
    batching: Dict[str, Batching],
    incs={},
) -> torch.Tensor:
    if isinstance(data_args[key], torch.Tensor):
        return data_args[key]

    if key == "num_structures":
        data_args[key] = torch.tensor([sum(data_args[key])], dtype=torch.long)
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


def _include_dependancies_in_keys(
    keys: Iterable[str], existing_keys: Iterable[str], batching: Dict[str, Batching]
) -> List[str]:
    keys_with_deps = set()

    for key in keys:
        if key not in batching:
            continue

        keys_with_deps.add(key)

        inc = batching[key].inc
        if isinstance(inc, str):
            keys_with_deps.add(inc)

        dim_size = batching[key].shape[batching[key].cat_dim]
        if isinstance(dim_size, str):
            keys_with_deps.add(dim_size)

    for key in existing_keys:
        if key not in batching:
            continue

        if batching[key].default is not None:
            keys_with_deps.add(key)

    return list(keys_with_deps)


def collate(
    structures: List[StructureData],
    cls: type = None,
    batching: Dict[str, Batching] = None,
    keys: Iterable[str] = None,
) -> StructureData:
    assert len(structures) > 0

    if cls is None:
        cls = structures[0].__class__

    if batching is None:
        batching = cls.batching

    existing_keys = set.union(*(set(struct.keys) for struct in structures))
    if keys is None:
        keys = existing_keys

    keys = _include_dependancies_in_keys(keys, existing_keys, batching)

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
        if key not in batching:
            continue

        cat_dim = batching[key].cat_dim
        cat_index = batching[key].shape[cat_dim]

        if isinstance(cat_index, str):
            keys_indexing.add(cat_index)

    selecting_index = {}
    selected_index = {}
    for cat_index in keys_indexing:
        size = indexing[cat_index][idx + 1] - indexing[cat_index][idx]

        n = size.sum()
        if n == 0:
            continue

        selected_idx = torch.arange(n, dtype=torch.long)
        offset = F.pad(size.cumsum(0), (1, 0))

        offset_idx = indexing[cat_index][idx] - selected_idx[offset[:-1]]

        selected_idx += offset_idx.repeat_interleave(size)

        selecting_index[cat_index] = selected_idx
        selected_index[cat_index] = offset

    return selecting_index, selected_index


def _select_and_decrement(
    batch: SelectableTensorMaping,
    keys: Iterable[str],
    batching: Dict[str, Batching],
    idx: torch.LongTensor,
    indexing: Dict[str, torch.LongTensor],
    select_indexing: Optional[Dict[str, torch.LongTensor]] = None,
    indexing_dst: Optional[Dict[str, torch.LongTensor]] = None,
) -> Dict[str, torch.LongTensor]:
    result = {}

    for key in keys:
        if key not in batching:
            continue

        inc = batching[key].inc
        cat_dim = batching[key].cat_dim
        cat_index = batching[key].shape[cat_dim]

        if isinstance(cat_index, int):
            data = batch[key].index_select(cat_dim, idx)
        elif select_indexing is None:
            data = batch[key].clone()
        elif cat_index not in select_indexing:
            continue
        else:
            current_idx = select_indexing[cat_index]
            data = batch[key].index_select(cat_dim, current_idx)

        # TODO: skip when idx is None

        if inc != 0:
            if isinstance(inc, str):
                offset = indexing[inc][idx]
            else:
                offset = inc * idx

            size = indexing[cat_index][idx + 1] - indexing[cat_index][idx]
            index = torch.arange(size.shape[0], dtype=torch.long).repeat_interleave(
                size
            )

            selection = tuple(
                None if i != cat_dim else index for i, _ in enumerate(data.shape)
            )
            data -= offset[selection]

            if indexing_dst is not None:
                pass  # TODO inc dest tensor

        result[key] = data

    return result


def _destination_indexing(
    idx: torch.LongTensor, indexing: Dict[str, torch.LongTensor]
) -> Dict[str, torch.LongTensor]:
    selected = {}

    for key, index in indexing.items():
        size = index[idx + 1] - index[idx]
        selected[key] = F.pad(torch.cumsum(size, 0), (1, 0))

    return selected


def _to_generator(
    cls: type,
    data: Dict[str, torch.Tensor],
    idx: torch.LongTensor,
    batching: Dict[str, Batching],
    indexing: Dict[str, torch.LongTensor],
):
    for i in range(idx.shape[0]):
        kwargs = {}

        for key in data.keys():
            if key not in batching:
                continue

            cat_dim = batching[key].cat_dim
            cat_index = batching[key].shape[cat_dim]

            if isinstance(cat_index, int):
                idx = torch.arange(cat_index * i, cat_index * (i + 1), dtype=torch.long)
            else:
                idx = torch.arange(
                    indexing[cat_index][i], indexing[cat_index][i + 1], dtype=torch.long
                )
            kwargs[key] = data[key].index_select(cat_dim, idx)

        yield cls(**kwargs)


def separate(
    batch: SelectableTensorMaping,
    idx: int | torch.LongTensor = None,
    cls: type = None,
    indexing: Dict[str, torch.LongTensor] = None,
    result: Literal["list", "iterator", "batch"] = "list",
    keys: Iterable[str] = None,
) -> List[StructureData]:
    if cls is None:
        cls = batch.__class__

    assert hasattr(cls, "batching"), "Batching can't be determined."
    batching = cls.batching

    if indexing is None:
        indexing = get_indexing(batch, batching)

    if keys is None:
        if isinstance(batch.keys, Callable):
            keys = batch.keys()
        else:
            keys = batch.keys

    if idx is None and result == "batch":
        raise NotImplementedError("feature not implemented yet")  # TODO

    if idx is not None:
        if isinstance(idx, int):
            idx = torch.tensor([idx], dtype=torch.long)
        assert isinstance(idx, torch.LongTensor), "idx must be None, int or LongTensor"
    else:
        idx = torch.arange(indexing["num_structures"].item(), dtype=torch.long)

    selecting_index, selected_index = _select_indexing(idx, keys, batching, indexing)

    if result == "batch":
        indexing_dst = _destination_indexing(idx, indexing)
    else:
        indexing_dst = None

    data = _select_and_decrement(
        batch, keys, batching, idx, indexing, selecting_index, indexing_dst=indexing_dst
    )

    if result == "iterator":
        return _to_generator(cls, data, idx, batching, selected_index)
    elif result == "list":
        return list(_to_generator(cls, data, idx, batching, selected_index))

    return cls(**data)
