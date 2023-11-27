import torch
from torch_geometric.data import Data
import numpy as np

import json
import typing
from typing import Any, Optional, Union, Dict, Callable
from dataclasses import dataclass

class BatchingEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, Batching):
            return {
                "cat_dim":obj.cat_dim,
                "inc":obj.inc,
                "size":obj.size,
                "dtype":obj.dtype,
                "default":obj.default
            }
        elif isinstance(obj, torch.dtype):
            return str(obj)
        
        return json.JSONEncoder.default(self, obj)


@dataclass(frozen=True)
class Batching:
    cat_dim: int = 0
    inc: int | str = 0
    size: int | str = 1
    dtype: torch.dtype = torch.float32
    default: Any = None

    def __post_init__(self):
        for var_name, var_type in typing.get_type_hints(self).items():
            if var_type == Any:
                continue

            assert isinstance(
                self.__getitem__(var_name), var_type
            ), f'Fields "{var_name}" of the "{self.__class__.__name__}" class must be of type "{var_type}".'

    def __getitem__(self, name: str) -> Any:
        if name in ["cat_dim", "inc", "size", "default", "dtype"]:
            return self.__getattribute__(name)
        else:
            raise KeyError
        
def batching(config: Dict[str, Batching]) -> Callable[[type], type]:
    assert isinstance(config, dict)
    for key, value in config.items():
        assert isinstance(key, str) and isinstance(value, Batching)

    def fn(cls):
        assert StructureData in cls.__bases__
        cls.batching = cls.batching.copy()
        cls.batching.update(config)

        return cls

    return fn


class StructureData(Data):
    """
    
    
    Extended description of function.

    Parameters
    ----------
    z : torch.LongTensor
        Description of z
    pos : Optional[torch.FloatTensor]
        Description of pos
    """

    batching: Dict[str, Batching] = {
        "pos": Batching(size="num_atoms"),
        "z": Batching(size="num_atoms", dtype=torch.long),
        "cell": Batching(),
        "y": Batching(),
        "num_atoms": Batching(default=0, dtype=torch.long),
        "batch_atoms": Batching(inc=1, size="num_atoms", default=0, dtype=torch.long),
        "periodic": Batching(dtype=torch.bool),
        "edge_index": Batching(
            cat_dim=1, inc="num_atoms", size="num_edges", dtype=torch.long
        ),
        "edge_cell": Batching(cat_dim=0, inc=0, size="num_edges", dtype=torch.long),
        "num_edges": Batching(default=0, dtype=torch.long),
        "batch_edges": Batching(inc=1, size="num_edges", default=0, dtype=torch.long),
        "triplet_index": Batching(
            cat_dim=1, inc="num_edges", size="num_triplets", dtype=torch.long
        ),
        "num_triplets": Batching(dtype=torch.long),
        "batch_triplets": Batching(inc=1, size="num_triplets", dtype=torch.long),
        "quadruplets_index": Batching(
            cat_dim=1, inc="num_edges", size="num_quadruplets", dtype=torch.long
        ),
        "num_quadruplets": Batching(dtype=torch.long),
        "batch_quadruplets": Batching(inc=1, size="num_quadruplets", dtype=torch.long),
    }

    def __init__(
        self,
        z: Optional[torch.LongTensor] = None,
        pos: Optional[torch.FloatTensor] = None,
        cell: Optional[torch.FloatTensor] = None,
        y: Optional[torch.FloatTensor] = None,
        edge_index: Optional[torch.LongTensor] = None,
        edge_cell: Optional[torch.LongTensor] = None,
        triplet_index: Optional[torch.LongTensor] = None,
        quadruplets_index: Optional[torch.LongTensor] = None,
        periodic: Optional[Union[bool, torch.BoolTensor]] = None,
        **kwargs,
    ):
        
        periodic = self._default_periodic(periodic, cell)

        self._merge_kwargs(
            kwargs,
            pos=pos,
            z=z,
            cell=cell,
            y=y,
            edge_index=edge_index,
            edge_cell=edge_cell,
            triplet_index=triplet_index,
            quadruplets_index=quadruplets_index,
            periodic=periodic,
        )

        self._to_tensor(kwargs)

        self._auto_fill(kwargs)

        super().__init__(**kwargs)

    @classmethod
    def _merge_kwargs(
        cls,
        kwargs: Dict[str, Any],
        pos: torch.FloatTensor,
        z: torch.LongTensor,
        cell: torch.FloatTensor,
        y: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_cell: torch.LongTensor,
        triplet_index: torch.LongTensor,
        quadruplets_index: torch.LongTensor,
        periodic: Union[bool, torch.BoolTensor],
    ):
        kwargs["pos"] = pos
        kwargs["z"] = z
        kwargs["cell"] = cell
        kwargs["y"] = y
        kwargs["edge_index"] = edge_index
        kwargs["edge_cell"] = edge_cell
        kwargs["triplet_index"] = triplet_index
        kwargs["quadruplets_index"] = quadruplets_index
        kwargs["periodic"] = periodic

    @classmethod
    def _to_tensor(cls, kwargs: Dict[str, Any]):
        for name, value in kwargs.items():
            if value is None:
                continue

            if isinstance(value, np.ndarray):
                kwargs[name] = torch.from_numpy(value)
            elif not isinstance(value, torch.Tensor):
                kwargs[name] = torch.tensor(value)

    @classmethod
    def _auto_fill(cls, kwargs: Dict[str, Any]) -> bool:
        for name, batching in cls.batching.items():
            if (name in kwargs) and (kwargs[name] is not None):
                continue

            from_key = next(
                filter(
                    lambda x: (kwargs[x] is not None)
                    and (cls.batching[x].size == name),
                    kwargs.keys(),
                ),
                None,
            )

            if from_key is not None:
                kwargs[name] = torch.tensor(
                    [kwargs[from_key].shape[batching.cat_dim]], dtype=batching.dtype
                )

        for name, batching in cls.batching.items():
            if (name in kwargs) and (kwargs[name] is not None):
                continue

            if batching.default is None:
                continue

            size = batching.size
            if isinstance(size, str) and (size in kwargs):
                size = kwargs[size].item()

            kwargs[name] = torch.full(
                (size,), fill_value=batching.default, dtype=batching.dtype
            )

    @classmethod
    def _default_periodic(
        cls,
        periodic: Union[bool, torch.BoolTensor] = None,
        cell: torch.FloatTensor = None,
    ) -> torch.BoolTensor:
        if periodic is None:
            periodic = cell is not None

        if isinstance(periodic, bool):
            periodic = torch.tensor(periodic)

        return periodic.flatten()

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> int:
        inc = self.batching.get(key, {"cat_dim": 0})["cat_dim"]

        return inc

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> int:
        inc = self.batching.get(key, {"inc": 0})["inc"]

        if isinstance(inc, str):
            inc = getattr(self, inc)
            assert isinstance(inc, (int, torch.Tensor))

        if isinstance(inc, torch.Tensor):
            inc = inc.item()

        return inc
