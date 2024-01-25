from __future__ import annotations
from sympy import group

import torch
import torch.nn.functional as F
from torch_geometric import data

# from torch_geometric.data.collate import collate
import h5py
import numpy as np
from tqdm import tqdm

from typing import Iterable, Iterator, List, Tuple, Any, Optional, Union, Callable, Dict
import os
import json
import hashlib

from .utils import uncompress_progress, download_progress, get_filename
from .base import StructureData, BatchingEncoder
from .collate import (
    SelectableTensor,
    SelectableTensorMaping,
    collate,
    separate,
    get_indexing,
)


"""
class Selector(h5py.Dataset):
    def __init__(self, dataset: h5py.Dataset, tensor: Optional[torch.Tensor] = None):
        self._dataset = dataset
        self._tensor = tensor

    def __getattr__(self, name: str) -> Any:
        if name in ("_dataset", "_tensor"):
            return super(Selector, self).__getattr__(name)

        return getattr(self._dataset, name)

    def __setattr__(self, name: str, value: Any):
        if name in ("_dataset", "_tensor"):
            return super(Selector, self).__setattr__(name, value)

        return setattr(self._dataset, name, value)

    def select_slice(self, dim: int, start: int, stop: int) -> torch.Tensor:
        ndim = self._dataset.ndim

        slices = tuple(
            slice(None) if i != dim else slice(start, stop) for i in range(ndim)
        )

        if self._tensor is None:
            return torch.from_numpy(self._dataset[slices])
        else:
            return self._tensor[slices]

    def __getitem__(
        self, indices: Union[None, int, slice, torch.LongTensor, List, Tuple]
    ) -> torch.Tensor:
        if self._tensor is None:
            if isinstance(indices, torch.Tensor):
                indices = indices.numpy()

            d = self._dataset[indices]
            if isinstance(d, np.ndarray):
                return torch.from_numpy(d)

            return torch.tensor([d])
        else:
            return self._tensor[indices]

    def __setitem__(
        self,
        indices: Union[None, int, slice, torch.LongTensor, List, Tuple],
        tensor: Union[np.ndarray, torch.Tensor],
    ):
        if self._tensor is not None:
            if isinstance(tensor, torch.Tensor):
                self._dataset[indices] = tensor
            else:
                self._dataset[indices] = torch.from_numpy(tensor)

        self._dataset[indices] = tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"


class HDF5GroupWrapper(h5py.Group):
    def __init__(self, id: Any, local: Optional[Dict[str, Union[Dict, torch.Tensor]]]):
        super().__init__(id)

        self.local = local

    def shape(self, key: str) -> tuple:
        return self.group[key].shape

    def __getitem__(self, key: str) -> Union[h5py.Group, Selector]:
        item = super().__getitem__(key)

        if isinstance(item, h5py.Dataset):
            item = Selector(
                item,
                None if self.local is None else self.local[key],
            )
        elif isinstance(item, h5py.Group):
            item = HDF5GroupWrapper(
                item.id,
                None
                if (self.local is None) or (key not in self.local)
                else self.local[key],
            )

        return item

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"


class HDF5FileWrapper(h5py.File):
    def __init__(self, *args, load: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.local: Dict[str, torch.Tensor] = None

        if load:
            self.load()

    @staticmethod
    def load_dataset(dataset: h5py.Group) -> Dict[str, Union[Dict, torch.Tensor]]:
        result = {}
        for key, value in dataset.items():
            if isinstance(value, h5py.Group):
                inner_dict = HDF5FileWrapper.load_dataset(value)
                result[key] = inner_dict
            else:
                result[key] = value[:]

        return result

    def load(self):
        self.local = HDF5FileWrapper.load_dataset(self)

    def shape(self, key: str) -> tuple:
        return self[key].shape

    def is_loaded(self) -> bool:
        return self.local is not None

    def __getitem__(self, key: str) -> Union[h5py.Group, Selector]:
        item = super().__getitem__(key)

        if isinstance(item, h5py.Dataset):
            item = Selector(
                item,
                None if self.local is None else self.local[key],
            )
        elif isinstance(item, h5py.Group):
            item = HDF5GroupWrapper(
                item.id,
                None
                if (self.local is None) or (key not in self.local)
                else self.local[key],
            )

        return item
"""


class HDF5TensorWrapper(SelectableTensor):
    def __init__(self, dataset: h5py.Dataset):
        self.dataset = dataset

    def __getitem__(self, args: int | torch.LongTensor | tuple) -> torch.Tensor:
        return torch.from_numpy(self.dataset[args])

    def index_select(self, dim: int, index: torch.LongTensor) -> torch.Tensor:
        idx = tuple(
            slice(None) if i != dim else index for i, s in enumerate(self.dataset.shape)
        )
        return torch.from_numpy(self.dataset[idx])

    @property
    def shape(self) -> tuple:
        return self.dataset.shape


class HDF5GroupWrapper(SelectableTensorMaping):
    def __init__(self, group: h5py.Group, caching: bool = False):
        self.group: h5py.Group = group
        self.caching: bool = caching
        self._cache: Dict[str, torch.Tensor] = {}

    def __iter__(self) -> Iterator[str]:
        return self.group.__iter__()

    def __len__(self) -> int:
        return self.group.__len__()

    def load_all(self):
        for key in self.group.keys():
            self._load(key)

    def _load(self, key: str) -> SelectableTensor:
        self._cache[key] = torch.from_numpy(self.group[key][:])

    def __getitem__(self, key: str) -> SelectableTensor:
        if self.caching:
            self._load(key)
            return self._cache[key]

        return HDF5TensorWrapper(self.group[key])


class HDF5Dataset(data.Dataset):
    data_class = StructureData

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[
            Callable[
                [Union[data.Data, data.HeteroData]], Union[data.Data, data.HeteroData]
            ]
        ] = None,
        pre_transform: Optional[
            Callable[
                [Union[data.Data, data.HeteroData]], Union[data.Data, data.HeteroData]
            ]
        ] = None,
        pre_filter: Optional[
            Callable[[Union[data.Data, data.HeteroData]], bool]
        ] = None,
        url: Optional[str] = None,
        md5: Optional[str] = None,
        data_file: Optional[str] = "data.hdf5",
        scalars_keys: Optional[List[str]] = [],
        compressed_file: Optional[str] = None,
        in_memory: Optional[bool] = False,
        **kwargs,
    ):
        self.url = url
        self.md5 = md5

        self.data_file = data_file
        self.compressed_file = compressed_file
        if self.compressed_file is None:
            self.compressed_file = get_filename(url)
        if os.path.splitext(self.compressed_file)[1] not in (
            ".zip",
            ".gzip",
            ".tar",
            ".gz",
        ):
            self.compressed_file = "compressed"
        self.in_memory = in_memory

        self.scalars_keys = scalars_keys

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            **kwargs,
        )

        file = h5py.File(self.processed_file, "r")

        self.indexing = {
            key: torch.from_numpy(d[:]) for key, d in file["indexing"].items()
        }

        self.data_hdf5 = HDF5GroupWrapper(file["data"], self.in_memory)

        if self.in_memory:
            self.data_hdf5.load_all()

    @property
    def raw_file_names(self) -> str:
        return self.compressed_file

    @property
    def processed_file_names(self) -> str:
        return self.data_file

    @property
    def downloaded_file(self) -> str:
        return os.path.join(self.raw_dir, self.compressed_file)

    @property
    def raw_file(self) -> str:
        return os.path.join(self.raw_dir, self.data_file)

    @property
    def processed_file(self) -> str:
        return os.path.join(self.processed_dir, self.data_file)

    @staticmethod
    def hash_md5(file: str, BUF_SIZE: int = 65536) -> str:
        md5 = hashlib.md5()

        with open(file, "rb") as f:
            while True:
                buffer = f.read(BUF_SIZE)
                if not buffer:
                    break
                md5.update(buffer)

        return md5.hexdigest()

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""

        download_progress(
            self.url, self.downloaded_file, desc=f"downloading {self.compressed_file}"
        )

        if (self.md5 is not None) and (
            HDF5Dataset.hash_md5(self.downloaded_file) != self.md5
        ):
            raise Exception(
                "md5 of the downloaded file doesn't match with the expected md5"
            )

    """
    @classmethod
    def _get_structure(
        cls,
        idx: int,
        group_data: Dict[str, torch.Tensor],
        group_slice: Dict[str, torch.Tensor],
        group_inc: Dict[str, torch.Tensor],
    ):
        cls_data = cls.data_class

        data_dict = {}
        for key in group_data.keys():
            if key in group_slice:
                start = group_slice[key][idx].item()
                end = group_slice[key][idx + 1].item()
            else:
                size = cls_data.batching[key].shape
                if isinstance(size, tuple):
                    size = size[cls_data.batching[key].cat_dim]
                assert isinstance(size, int)
                start, end = size * idx, size * (idx + 1)

            if key in group_inc:
                inc = group_inc[key][idx]
            elif isinstance(cls_data.batching[key].inc, int):
                inc = cls_data.batching[key].inc * idx
            else:
                inc = None

            data_key = group_data[key].select_slice(
                cls_data.batching[key].cat_dim, start, end
            )

            if inc != 0 and data_key.dtype != torch.bool:
                data_key -= inc

            data_dict[key] = data_key

        print(data_dict)
        return cls.data_class(**data_dict)

    @classmethod
    def separate(cls, data: data.Batch) -> List[StructureData]:
        data.to_data_list

    @classmethod
    def read_hdf5(cls, file: HDF5FileWrapper, idx: int) -> StructureData:
        return cls._get_structure(idx, file["data"], file["slice"], file["inc"])
    """

    @classmethod
    def create_dataset(
        cls,
        path: str,
        structures: List[StructureData],
        data_file: str = "data.hdf5",
        batching_file: str = "batching.json",
    ):
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, batching_file), "w") as fp:
            json.dump(structures[0].batching, fp, cls=BatchingEncoder)

        batched = collate(structures)
        indexing = get_indexing(batched)

        file = h5py.File(os.path.join(path, data_file), "w")

        group_data = file.create_group("data")
        for key in batched.keys:
            group_data.create_dataset(key, data=batched[key].numpy())

        group_indexing = file.create_group("indexing")
        for key, index in indexing.items():
            group_indexing.create_dataset(key, data=index.numpy())

        file.flush()
        file.close()

    """
    @staticmethod
    def write_hdf5(file: HDF5FileWrapper, structures: List[StructureData]):
        cls = type(structures[0])
        batched, slice_dict, inc_dict = collate(cls, data_list=structures)

        keys = list(filter(lambda key: key in structures[0], cls.batching.keys()))

        group_data = file.create_group("data")
        for key in keys:
            group_data.create_dataset(key, data=batched[key].numpy())

        group_slice = file.create_group("slice")
        for key in keys:
            if isinstance(cls.batching[key].shape, str):
                group_slice.create_dataset(key, data=slice_dict[key].numpy())
            elif isinstance(cls.batching[key].shape, tuple) and any(
                map(lambda x: isinstance(x, str), cls.batching[key].shape)
            ):
                group_slice.create_dataset(key, data=slice_dict[key].numpy())

        group_inc = file.create_group("inc")
        for key in keys:
            if isinstance(cls.batching[key].inc, str):
                group_inc.create_dataset(key, data=inc_dict[key].numpy())
    """

    def process(self):
        preprocessing = (self.pre_transform is not None) or (
            self.pre_filter is not None
        )

        dst = self.raw_dir if preprocessing else self.processed_dir
        if not os.path.exists(os.path.join(dst, self.data_file)):
            uncompress_progress(
                self.downloaded_file,
                self.data_file,
                dst,
                desc=f"unpack {self.compressed_file}",
            )

        if preprocessing:
            raw_data = h5py.File(self.raw_file, "r")

            data_hdf5 = HDF5GroupWrapper(raw_data["data"], self.in_memory)
            indexing = {
                key: torch.from_numpy(d[:]) for key, d in raw_data["indexing"].items()
            }

            class StructureIterator(Iterable[StructureData]):
                def __init__(
                    self,
                    data: HDF5GroupWrapper,
                    indexing: Dict[str, torch.LongTensor],
                    data_type,
                ):
                    self.data = data
                    self.indexing = indexing
                    self.data_type = data_type

                def __len__(self) -> int:
                    return 1 << 10
                    return self.indexing["num_structures"].items()

                def __iter__(self) -> StructureData:
                    for i in range(len(self)):
                        print(i)
                        yield separate(
                            self.data, i, cls=self.data_type, indexing=self.indexing
                        )

            processed_dataset = []
            it = StructureIterator(data_hdf5, indexing, self.data_class)

            for struct in tqdm(it, desc="preprocessing", unit="structure", leave=False):
                if self.pre_filter is not None:
                    if not self.pre_filter(struct):
                        continue

                if self.pre_transform is not None:
                    struct = self.pre_transform(struct)

                processed_dataset.append(struct)

            self.create_dataset(
                self.processed_dir, processed_dataset, data_file=self.data_file
            )

    def len(self) -> int:
        r"""Returns the number of data objects stored in the dataset."""
        return self.indexing["num_structures"].item()

    def get(self, idx: int | torch.LongTensor) -> StructureData:
        r"""Gets the data object at index :obj:`idx`."""
        return separate(
            self.data_hdf5,
            idx,
            cls=self.data_class,
            indexing=self.indexing,
            to_list=False,
        )

    def close(self):
        self.data_hdf5.close()
