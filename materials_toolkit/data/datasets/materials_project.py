import torch
from typing import Callable, List, Optional, Union
from torch_geometric import data
from ..base import StructureData, batching, Batching
from ..dataset import HDF5Dataset


@batching(material_id=Batching(dtype=torch.long))
class MaterialsProjectData(StructureData):
    pass


class MaterialsProject(HDF5Dataset):
    data_class = MaterialsProjectData

    def __init__(
        self,
        root: str,
        transform: Callable[[data.Data], data.Data] | None = None,
        pre_transform: Callable[[data.Data], data.Data] | None = None,
        pre_filter: Callable[[data.Data], bool] | None = None,
        in_memory: bool | None = False,
        **kwargs
    ):
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            url="https://huggingface.co/datasets/materials-toolkits/materials-project/resolve/main/materials-project.tar.gz",
            md5="e8d2d6b877e5e59e214cb1a4519b7ac3",
            scalars_keys=["energy_pa"],
            in_memory=in_memory,
            **kwargs
        )
