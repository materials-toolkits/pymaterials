from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric import data

import h5py
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map

from typing import Iterable, Iterator, List, Tuple, Any, Optional, Union, Callable, Dict
import os
import shutil
import json
import hashlib
import warnings

from materials_toolkit.data.convex_hull import DatasetWithEnergy
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.composition import Composition
from pymatgen.core import Element
from torch_scatter import scatter_mul

from .utils import uncompress_progress, download_progress, get_filename
from .base import StructureData, BatchingEncoder
from .collate import (
    SelectableTensor,
    SelectableTensorMaping,
    collate,
    separate,
    get_indexing,
)


class HDF5TensorWrapper(SelectableTensor):
    def __init__(self, dataset: h5py.Dataset):
        self.dataset = dataset

    def __getitem__(self, args: int | torch.LongTensor | tuple) -> torch.Tensor:
        if isinstance(args, torch.Tensor):
            args = args.numpy()

        return torch.tensor(self.dataset[args])

    def index_select(self, dim: int, index: torch.LongTensor) -> torch.Tensor:
        if isinstance(index, torch.Tensor):
            index = index.numpy()

        idx = tuple(
            slice(None) if i != dim else index for i, _ in enumerate(self.dataset.shape)
        )
        return torch.tensor(self.dataset[idx])

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

    def __contains__(self, __key: object) -> bool:
        return self.group.__contains__(__key)

    def _load(self, key: str) -> SelectableTensor:
        self._cache[key] = torch.from_numpy(self.group[key][:])

    def __getitem__(self, key: str) -> SelectableTensor:
        if self.caching:
            self._load(key)
            return self._cache[key]

        return HDF5TensorWrapper(self.group[key])


def _process_phase_diagram(args):
    key, entry_lst = args
    try:
        hull = PhaseDiagram(entry_lst)
    except ValueError:
        hull = None
    return (key, hull)


class HDF5Dataset(data.Dataset, DatasetWithEnergy):
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
        use_convex_hull: bool = False,
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
            self.compressed_file = None
        self.in_memory = in_memory

        self.scalars_keys = scalars_keys

        self.use_convex_hull = use_convex_hull

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            **kwargs,
        )

        self.load()

    @property
    def raw_file_names(self) -> List[str]:
        if self.compressed_file is None:
            return []

        return [self.compressed_file]

    @property
    def processed_file_names(self) -> List[str]:
        return [self.data_file]

    @property
    def downloaded_file(self) -> str:
        if self.compressed_file is not None:
            return os.path.join(self.raw_dir, self.compressed_file)
        return None

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

    def entries(self, composition: torch.LongTensor) -> List[PDEntry]:
        result = []
        mask = torch.zeros(128, dtype=torch.long).scatter(
            0, composition.long(), torch.ones_like(composition, dtype=torch.long)
        )

        z = torch.from_numpy(self.data_hdf5["z"].dataset[:])
        num_atoms = torch.from_numpy(self.data_hdf5["num_atoms"].dataset[:])
        batch = torch.arange(num_atoms.shape[0]).repeat_interleave(num_atoms, 0)
        filtered = scatter_mul(mask[z], batch)
        idx = filtered.nonzero().flatten()

        for struct in separate(self.get(idx)):
            atomic_number, count = struct.z.unique(return_counts=True)

            comp = Composition(
                {z.item(): n.item() for z, n in zip(atomic_number, count)}
            )
            energy = struct["energy_pa"].item() * struct["num_atoms"].item()

            result.append(PDEntry(comp, energy))

        return result

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""

        if self.url is None:
            return

        print("compressed_file", self.compressed_file)
        if os.path.exists(self.compressed_file):
            return False

        download_progress(
            self.url, self.downloaded_file, desc=f"downloading {self.compressed_file}"
        )

        if (self.md5 is not None) and (
            HDF5Dataset.hash_md5(self.downloaded_file) != self.md5
        ):
            raise Exception(
                "md5 of the downloaded file doesn't match with the expected md5"
            )

    def compute_convex_hulls(self, file_name: str = None):
        if file_name is None:
            file_name = self.processed_file

        if os.path.normpath(file_name) == os.path.normpath(self.processed_file):
            self.close()

        file = h5py.File(file_name, "r+")
        data = HDF5GroupWrapper(file["data"])
        indexing = {key: torch.from_numpy(d[:]) for key, d in file["indexing"].items()}
        length = indexing["num_structures"].item()

        if "energy_above_hull" in file:
            return

        entries = []
        s = set()
        for struct in tqdm(
            separate(
                data,
                cls=self.data_class,
                indexing=indexing,
                result="iterator",
                keys=["z", "energy_pa"],
            ),
            total=length,
            desc="collect energy to calculate convex hull",
            leave=False,
        ):
            entry = DatasetWithEnergy.Entry(struct.z, struct.energy_pa)
            s.add(entry.key)
            entries.append(entry)

        systems = DatasetWithEnergy.Entry.inclusion_graph(entries)

        for key, entry_lst in tqdm(
            systems.items(),
            desc="calculate convex hull",
        ):
            try:
                hull = PhaseDiagram(entry_lst)
            except ValueError:
                hull = None

            self.phase_diagram[key] = hull

        e_above_hull = []

        for entry in tqdm(
            entries,
            desc="calculate energy above hull",
        ):
            energy = self.calculate_e_above_hull(entry=entry)
            e_above_hull.append(energy)

        energies = torch.tensor(
            e_above_hull, dtype=self.data_class.batching["energy_above_hull"].dtype
        )

        file["data"].create_dataset("energy_above_hull", data=energies)

        file.flush()
        file.close()

        if os.path.normpath(file_name) == os.path.normpath(self.processed_file):
            self.load()

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

    def _unzip_if_needed(self):
        unzipped = os.path.exists(self.raw_file)
        existing_zipped = self.downloaded_file is not None and os.path.exists(
            self.downloaded_file
        )

        if (not unzipped) and existing_zipped:
            uncompress_progress(
                self.downloaded_file,
                self.data_file,
                self.raw_dir,
                desc=f"unpack {self.compressed_file}",
            )

    def process(self):
        self._unzip_if_needed()

        if self.use_convex_hull:
            self.compute_convex_hulls(self.raw_file)

        preprocessing = (self.pre_transform is not None) or (
            self.pre_filter is not None
        )

        if preprocessing:
            raise NotImplementedError("preprocessing is not implemented yet")
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

        if not os.path.exists(self.processed_file):
            shutil.copyfile(self.raw_file, self.processed_file)

    def len(self) -> int:
        r"""Returns the number of data objects stored in the dataset."""
        return self.indexing["num_structures"].item()

    def get(self, idx: int | torch.LongTensor, keys: List[str] = None) -> StructureData:
        r"""Gets the data object at index :obj:`idx`."""

        return separate(
            self.data_hdf5,
            idx=idx,
            cls=self.data_class,
            indexing=self.indexing,
            result="batch",
            keys=keys,
        ).set_dataset(self)

    def close(self):
        if hasattr(self, "file") and self.file:
            self.file.close()

    def load(self):
        self.close()

        self.file = h5py.File(self.processed_file, "r")

        self.indexing = {
            key: torch.from_numpy(d[:]) for key, d in self.file["indexing"].items()
        }

        self.data_hdf5 = HDF5GroupWrapper(self.file["data"], self.in_memory)

        if self.in_memory:
            self.data_hdf5.load_all()

    def __del__(self):
        self.close()
