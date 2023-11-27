from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric import data
from torch_geometric.data.collate import collate
import h5py
import numpy as np
from tqdm import tqdm

from typing import List, Tuple, Any, Optional, Union, Callable, Dict
import urllib.request
import os
import json
import hashlib
import shutil

from .utils import uncompress_progress, download_progress, get_filename
from .base import StructureData, BatchingEncoder


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

        self.data_hdf5 = HDF5FileWrapper(self.processed_file, "r")
        if self.in_memory:
            self.data_hdf5.load()

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

    @staticmethod
    def length_hdf5(file: HDF5FileWrapper) -> int:
        return file["data"]["periodic"].shape[0]

    @classmethod
    def read_hdf5(cls, file: HDF5FileWrapper, idx: int) -> StructureData:
        cls_data = cls.data_class

        group_data = file["data"]
        group_slice = file["slice"]
        group_inc = file["inc"]

        data_dict = {}
        for key in group_data.keys():
            if key in group_slice:
                start = group_slice[key][idx]
                end = group_slice[key][idx + 1]
            else:
                size = cls_data.batching[key].size
                assert isinstance(size, int)
                start, end = size * idx, size * (idx + 1)

            if key in group_inc:
                inc = group_inc[key][idx]
            else:
                inc = cls_data.batching[key].inc
                assert isinstance(inc, int)

            data_key = group_data[key][start:end]
            if inc != 0 and data_key.dtype != torch.bool:
                data_key -= inc

            data_dict[key] = data_key

        return cls.data_class(**data_dict)

    @classmethod
    def create_dataset(cls, path: str, structures: List[StructureData]):
        with open(os.path.join(path, "batching.json"), "w") as fp:
            json.dump(structures[0].batching, fp, cls=BatchingEncoder)

        cls.write_hdf5(
            HDF5FileWrapper(os.path.join(path, "data.hdf5"), "w"), structures
        )

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
            if isinstance(cls.batching[key].size, str):
                group_slice.create_dataset(key, data=slice_dict[key].numpy())

        group_inc = file.create_group("inc")
        for key in keys:
            if isinstance(cls.batching[key].inc, str):
                group_inc.create_dataset(key, data=inc_dict[key].numpy())

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
            raw_data = HDF5FileWrapper(self.raw_file, "r")

            processed_dataset = []
            length = self.length_hdf5(raw_data)
            length = 16
            idx = torch.arange(length)

            for index in tqdm(idx, desc="preprocessing", unit="structure", leave=False):
                structs = self.read_hdf5(raw_data, index, self.scalars_keys)

                if self.pre_filter is not None:
                    if not self.pre_filter(structs):
                        continue

                if self.pre_transform is not None:
                    structs = self.pre_transform(structs)

                processed_dataset.append(structs)

            processed_data = HDF5FileWrapper(self.processed_file, "w")
            print(processed_data)
            self.write_hdf5(processed_data, processed_dataset)

    def len(self) -> int:
        r"""Returns the number of data objects stored in the dataset."""
        return self.length_hdf5(self.data_hdf5)

    def get(self, idx: int) -> StructureData:
        r"""Gets the data object at index :obj:`idx`."""
        return self.read_hdf5(self.data_hdf5, idx)

    def close(self):
        self.data_hdf5.close()


def main():
    from .datasets import MaterialsProject

    dataset = MaterialsProject.read_hdf5(
        HDF5FileWrapper("materials-project/data.hdf5", "r"), 2
    )
    print(dataset)

    # HDF5Dataset.write_hdf5(HDF5FileWrapper("data.hdf5", "w"), dataset)

    exit(0)
    dataset = HDF5Dataset(
        root="./data/mp",
        url="https://huggingface.co/datasets/materials-toolkits/materials-project/resolve/main/materials-project.tar.gz",
        md5="03370e8fe6426f7fbe9494b11e215ee9",
        pre_transform=lambda x: x,
    )

    N = 1024
    length = torch.randint(4, 8, (N,))
    ptr = F.pad(torch.cumsum(length, 0), (1, 0))
    idx = torch.randperm(N)[:16]
    print(length)
    print(ptr)
    print(idx)
    print(length[idx])
    print(dataset.index_from_ptr(idx, ptr))


if __name__ == "__main__":
    main()
