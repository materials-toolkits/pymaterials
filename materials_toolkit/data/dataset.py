import torch
import torch.nn.functional as F
from torch_geometric import data
import h5py
import numpy as np
from tqdm import tqdm

from typing import List, Tuple, Any, Optional, Union, Callable, Dict
import urllib.request
import os
import hashlib
import shutil

from .utils import uncompress_progress, download_progress, get_filename
from .data import StructureData


def preprocess_jobs():
    pass


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
            return torch.from_numpy(self._dataset[indices])
        else:
            return self._tensor[indices]

    def __setitem__(
        self,
        indices: Union[None, int, slice, torch.LongTensor, List, Tuple],
        tensor: Union[np.ndarray, torch.Tensor],
    ):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()

        self._dataset[indices] = tensor

        if self._tensor is not None:
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            self._dataset[indices] = tensor


class HDF5FileWrapper(h5py.File):
    def __init__(self, *args, load: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.local: Dict[str, torch.Tensor] = None

        if load:
            self.load()

    @staticmethod
    def load_dataset(dataset: h5py.Dataset) -> Dict[str, torch.Tensor]:
        result = {}
        for key, value in dataset.items():
            if isinstance(value, h5py.Group):
                inner_dict = HDF5FileWrapper.load_dataset(value)
                for inner_key, inner_value in inner_dict.items():
                    result[os.path.join(key, inner_key)] = inner_value
            else:
                result[key] = torch.from_numpy(value[:])

        return result

    def load(self):
        self.local = HDF5FileWrapper.load_dataset(self)

    def shape(self, key: str) -> tuple:
        return self[key].shape

    def is_loaded(self) -> bool:
        return self.local is not None

    def __getitem__(self, key: str) -> Selector:
        return Selector(super().__getitem__(key), self.local)


class HDF5Dataset(data.Dataset):
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
        workers: Optional[int] = None,
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

        self.workers = workers

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
    def index_from_ptr(
        idx: torch.LongTensor,
        ptr: torch.LongTensor,
    ) -> torch.LongTensor:
        length = ptr.index_select(0, idx + 1) - ptr.index_select(0, idx)
        ptr = ptr.index_select(0, idx)

        print(idx, ptr, length)

        batched_idx = torch.arange(0, length.sum().item())
        offset_idx = ptr - F.pad(length[:-1].cumsum(0), (1, 0))

        atoms_idx = batched_idx + offset_idx.repeat_interleave(length)

        return atoms_idx

    @staticmethod
    def length_hdf5(file: HDF5FileWrapper) -> int:
        return file.shape("structures/natoms")[0]

    @staticmethod
    def read_hdf5(
        file: HDF5FileWrapper,
        idx: Union[int, np.ndarray, torch.LongTensor],
        scalars_keys: Optional[List[str]] = [],
    ) -> Union[StructureData, List[StructureData]]:
        single_strutcure = False
        if isinstance(idx, int):
            single_strutcure = True
            idx = torch.tensor([idx], dtype=torch.long)
        elif isinstance(idx, np.ndarray):
            idx = torch.from_numpy(idx).long()
        elif isinstance(idx, torch.LongTensor):
            if idx.ndim == 0:
                idx = idx.reshape(1)

        atoms_ptr = file["structures/atoms_ptr"]
        print(atoms_ptr)

        atoms_idx = HDF5Dataset.index_from_ptr(idx, atoms_ptr)

        if "structures/cell" in file:
            cell = torch.from_numpy(file["structures/cell"][idx])
            periodic = cell.det().abs() > 1e-3
        else:
            cell = None
            periodic = torch.full_like(idx, False, dtype=torch.bool)

        x = file["atoms/positions"][atoms_idx]
        z = file["atoms/atomic_number"][atoms_idx]
        scalars = {
            key: file[os.path.join("structures", key)][idx] for key in scalars_keys
        }
        print("cell", cell.shape)
        print("x", x.shape)
        print("z", z.shape)
        print("periodic", periodic.shape)
        for key, value in scalars.items():
            print(key, value.shape)

        if single_strutcure:
            return StructureData(cell=cell, x=x, z=z, periodic=periodic, **scalars)

        structs = []
        for i in idx:
            structs.append(
                StructureData(
                    cell=cell[i],
                    x=x[atoms_idx == i],
                    z=z[atoms_idx == i],
                    periodic=periodic[i],
                    **scalars,
                )
            )
        return structs

    @staticmethod
    def write_hdf5(file: HDF5FileWrapper, structures: List[StructureData]):
        structures = data.Batch.from_data_list(structures)

        for key, tensor in structures.to_dict().iter():
            file.create_dataset(key, data=tensor.numpy())

    def process(self):
        preprocessing = (self.pre_transform is not None) or (
            self.pre_filter is not None
        )

        dst = self.raw_dir if preprocessing else self.processed_dir
        if not os.path.exists(os.path.join(dst, self.data_file)):
            uncompress_progress(
                self.downloaded_file,
                os.path.join(".", self.data_file),
                dst,
                desc=f"unpack {self.compressed_file}",
            )

        if preprocessing:
            raw_data = HDF5FileWrapper(self.raw_file, "r")

            processed_dataset = []
            length = HDF5Dataset.length_hdf5(raw_data)
            idx = torch.arange(length)

            for index in tqdm(idx, desc="preprocessing", unit="structure", leave=False):
                structs = HDF5Dataset.read_hdf5(raw_data, index, self.scalars_keys)

                if self.pre_filter is not None:
                    if not self.pre_filter(structs):
                        continue

                if self.pre_transform is not None:
                    structs = self.pre_transform(structs)

                processed_dataset.append(structs)

            processed_data = HDF5FileWrapper(self.processed_file, "w")
            HDF5Dataset.write_hdf5(processed_data, processed_dataset)

    def len(self) -> int:
        r"""Returns the number of data objects stored in the dataset."""
        return HDF5Dataset.length_hdf5(self.data_hdf5)

    def get(self, idx: int) -> StructureData:
        r"""Gets the data object at index :obj:`idx`."""
        return HDF5Dataset.read_hdf5(self.data_hdf5, idx, self.scalars_keys)

    def close(self):
        self.data_hdf5.close()


if __name__ == "__main__":
    x = HDF5FileWrapper("data/mp/raw/data.hdf5", "r", load=True)

    print(x["structures/atoms_ptr"])

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
