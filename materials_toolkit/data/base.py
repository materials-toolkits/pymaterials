import torch
import torch_geometric
import numpy as np

import json
import typing
from typing import Any, Optional, Union, Dict, Callable, Tuple
from dataclasses import dataclass


class BatchingEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Batching):
            return {
                "cat_dim": obj.cat_dim,
                "inc": obj.inc,
                "size": obj.shape,
                "dtype": obj.dtype,
                "default": obj.default,
            }
        elif isinstance(obj, torch.dtype):
            return str(obj)

        return json.JSONEncoder.default(self, obj)


@dataclass(frozen=True)
class Batching:
    """
    Configure how a list of :class:`StructureData` objects are collated into a batch.

    Parameters
    ----------
    cat_dim: int
        Concatenation dimension.
    inc: int | str
        Automatic increment. This argument is used for batching graphs.
    shape: int | str | tuple
        Shape of the tensor.
    dtype: torch.dtype
        Type of tensor.
    default: int
        Default value of the tensor.
    """

    cat_dim: int = 0
    inc: int | str = 0
    shape: int | str | tuple = 1
    dtype: torch.dtype = torch.float32
    default: Any = None

    def _assert_var(self, var: Any, name: str, type: type):
        assert isinstance(
            var, type
        ), f'Fields "{name}" of the "{self.__class__.__name__}" class must be of type "{type}".'

    def __post_init__(self):
        for var_name, var_type in typing.get_type_hints(self).items():
            if var_type == Any:
                continue

            var = self.__getitem__(var_name)

            if isinstance(var, tuple):
                for v in var:
                    self._assert_var(v, var_name, var_type)
            else:
                self._assert_var(var, var_name, var_type)

    def __getitem__(self, name: str) -> Any:
        if name in typing.get_type_hints(self):
            return self.__getattribute__(name)
        else:
            raise KeyError


import functools


def batching(**config: Dict[str, Batching]) -> Callable[[type], type]:
    """
    A decoration to define how :class:`StructureData` objects are collated into a batch.

    >>> @batching(custom_field=Batching(cat_dim=1, shape=(4, "num_atoms"), default=1.0)) # doctest: +SKIP
    ... class CustomData(StructureData):
    ...     pass
    ...
    ... data = CustomData(z=torch.tensor([8, 1, 1], dtype=torch.long))
    ... print(data)
    ... print(data.custom_field)
    CustomData(z=[3], target_cell=[0, 3], periodic=[1], num_atoms=[1], batch_atoms=[3], num_edges=[1], batch_edges=[0], custom_field=[4, 3])
    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])

    """

    assert isinstance(config, dict)
    for key, value in config.items():
        assert isinstance(key, str) and isinstance(value, Batching)

    @functools.wraps(config)
    def fn(cls):
        assert StructureData in cls.__bases__
        cls.batching = cls.batching.copy()
        cls.batching.update(config)

        return cls

    return fn


class StructureData(torch_geometric.data.Data):
    """
    A class for storing chemical structure data. This class is derived from :class:`torch_geometric.data.Data` `[doc] <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data>`_ and is able to store the molecule and the periodic crystal. The :class:`StructureData` class can also store the associated graph.

    Here is some sample code covering a few use cases:

    * Water molecule in 3D

    >>> StructureData(
    ...     z = torch.tensor([8, 1, 1],dtype=torch.long),
    ...     pos = torch.tensor([[0.0, 0.0, 0.0],
    ...                         [-0.7575408, 0.58707964, 0.0],
    ...                         [0.7575408, 0.58707964, 0.0]])
    ... )
    StructureData(pos=[3, 3], z=[3], target_cell=[0, 3], periodic=[1], num_atoms=[1], batch_atoms=[3], num_edges=[1], batch_edges=[0])

    * Water molecule with bonds

    >>> StructureData(
    ...     z = torch.tensor([8, 1, 1],dtype=torch.long),
    ...     edge_index = torch.tensor([[0, 0], [1, 2]],dtype=torch.long)
    ... )
    StructureData(edge_index=[2, 2], z=[3], target_cell=[2, 3], periodic=[1], num_atoms=[1], batch_atoms=[3], num_edges=[1], batch_edges=[2])

    * Peroveskite in 3D

    >>> StructureData(
    ...     z = torch.tensor([20, 22, 8, 8, 8],dtype=torch.long),
    ...     pos = torch.tensor([[0.0, 0.0, 0.0],
    ...                         [0.5, 0.5, 0.5],
    ...                         [0.0, 0.5, 0.5],
    ...                         [0.5, 0.0, 0.5],
    ...                         [0.5, 0.5, 0.0]]),
    ...     cell = torch.tensor([[[3.867, 0.0, 0.0],
    ...                           [0.0, 3.867, 0.0],
    ...                           [0.0, 0.0, 3.867]]])
    ... )
    StructureData(pos=[5, 3], z=[5], cell=[1, 3, 3], target_cell=[0, 3], periodic=[1], num_atoms=[1], batch_atoms=[5], num_edges=[1], batch_edges=[0])

    * Crystal with an associated graph

    >>> StructureData(
    ...     z = torch.tensor([26, 26], dtype=torch.long),
    ...     pos = torch.tensor([[0.0, 0.0, 0.0],
    ...                         [0.5, 0.5, 0.5]]),
    ...     cell = torch.tensor([[[2.818, 0.0, 0.0],
    ...                           [0.0, 2.818, 0.0],
    ...                           [0.0, 0.0, 2.818]]]),
    ...     edge_index = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ...                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long),
    ...     target_cell = torch.tensor([[0, 0, 0],
    ...                                 [0, 0, 1],
    ...                                 [0, 1, 0],
    ...                                 [0, 1, 1],
    ...                                 [1, 0, 0],
    ...                                 [1, 0, 1],
    ...                                 [1, 1, 0],
    ...                                 [1, 1, 1],
    ...                                 [1, 0, 0],
    ...                                 [-1, 0, 0],
    ...                                 [0, 1, 0],
    ...                                 [0,-1, 0],
    ...                                 [0, 0, 1],
    ...                                 [0, 0,-1]], dtype=torch.long)
    ... )
    StructureData(edge_index=[2, 14], pos=[2, 3], z=[2], cell=[1, 3, 3], target_cell=[14, 3], periodic=[1], num_atoms=[1], batch_atoms=[2], num_edges=[1], batch_edges=[14])

    Parameters
    ----------
    z: torch.LongTensor
        The atomic number of the atoms (shape ``[num_atoms]``).
    pos: Optional[torch.FloatTensor]
        The positions of the atoms (shape ``[num_atoms, 3]``). The position is given as a fractional coordinate if the structure is periodic.
    cell: Optional[torch.FloatTensor]
        Matrix representing the lattice of a crystal (shape ``[1, 3, 3]``).
    y: Optional[torch.FloatTensor]
        Global properties of the structure (shape ``[1, num_properties]``).
    edge_index: Optional[torch.LongTensor]
        Edge of the graph. Edges are stored as a pair of indexed nodes (shape ``[2, num_edges]``). See the documentation for :class:`torch_geometric.data.Data` for more information.
    target_cell: Optional[torch.LongTensor]
        The coordinate of the cell associated with the target node of an edge (shape ``[num_edges, 3]``). This parameter is used to represent multiple graphs in a periodic structure. An edge can be shared by several cells.
    triplet_index: Optional[torch.LongTensor]
        Triplets of vertices, stored as a pair of edges sharing a vertex. Triplet_index is (shape ``[2, num_triplets]``).
    quadruplets_index: Optional[torch.LongTensor]
        Quadruplets of vertices stored as triplets of edges forming a chain. quadruplets_index is (shape ``[3, num_quadruplets]``).
    periodic: Optional[bool | torch.BoolTensor]
        Specifies whether the structure is periodic. This value is determined automatically, but can be forced manually.
    """

    batching: Dict[str, Batching] = {
        "pos": Batching(shape=("num_atoms", 3)),
        "z": Batching(shape="num_atoms", dtype=torch.long),
        "cell": Batching(shape=(1, 3, 3)),
        "y": Batching(),
        "num_atoms": Batching(default=0, dtype=torch.long),
        "batch_atoms": Batching(inc=1, shape="num_atoms", default=0, dtype=torch.long),
        "periodic": Batching(dtype=torch.bool),
        "edge_index": Batching(
            cat_dim=1, inc="num_atoms", shape=(2, "num_edges"), dtype=torch.long
        ),
        "target_cell": Batching(
            cat_dim=0, inc=0, shape=("num_edges", 3), dtype=torch.long, default=0
        ),
        "num_edges": Batching(default=0, dtype=torch.long),
        "batch_edges": Batching(inc=1, shape="num_edges", default=0, dtype=torch.long),
        "triplet_index": Batching(
            cat_dim=1, inc="num_edges", shape=(2, "num_triplets"), dtype=torch.long
        ),
        "num_triplets": Batching(dtype=torch.long),
        "batch_triplets": Batching(inc=1, shape="num_triplets", dtype=torch.long),
        "quadruplets_index": Batching(
            cat_dim=1, inc="num_edges", shape=(3, "num_quadruplets"), dtype=torch.long
        ),
        "num_quadruplets": Batching(dtype=torch.long),
        "batch_quadruplets": Batching(inc=1, shape="num_quadruplets", dtype=torch.long),
    }

    def __init__(
        self,
        z: Optional[torch.LongTensor] = None,
        pos: Optional[torch.FloatTensor] = None,
        cell: Optional[torch.FloatTensor] = None,
        y: Optional[torch.FloatTensor] = None,
        edge_index: Optional[torch.LongTensor] = None,
        target_cell: Optional[torch.LongTensor] = None,
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
            target_cell=target_cell,
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
        target_cell: torch.LongTensor,
        triplet_index: torch.LongTensor,
        quadruplets_index: torch.LongTensor,
        periodic: Union[bool, torch.BoolTensor],
    ):
        kwargs["pos"] = pos
        kwargs["z"] = z
        kwargs["cell"] = cell
        kwargs["y"] = y
        kwargs["edge_index"] = edge_index
        kwargs["target_cell"] = target_cell
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
    def _infer_size(cls, kwargs: Dict[str, Any]) -> Dict[str, int]:
        sizes = {}

        for name, batching in kwargs.items():
            if batching is None:
                continue

            if cls.batching[name].shape is None:
                continue

            if isinstance(cls.batching[name].shape, str):
                dim_name = cls.batching[name].shape
                dim_size = batching.shape[0]

                if dim_name in sizes:
                    assert (
                        sizes[dim_name] == dim_size
                    ), f"multiple size for dim {dim_name}"
                else:
                    sizes[dim_name] = dim_size

            elif isinstance(cls.batching[name].shape, tuple):
                for i, dim_name in enumerate(cls.batching[name].shape):
                    if isinstance(dim_name, str):
                        dim_size = batching.shape[i]

                        if dim_name in sizes:
                            assert (
                                sizes[dim_name] == dim_size
                            ), f"multiple size for dim {dim_name}"
                        else:
                            sizes[dim_name] = dim_size

        for name, config in cls.batching.items():
            if name in sizes:
                continue

            if config.default is None:
                continue

            if isinstance(config.shape, str):
                continue
            elif isinstance(config.shape, tuple) and any(
                map(lambda s: isinstance(s, str), config.shape)
            ):
                continue

            sizes[name] = config.default

        return sizes

    @classmethod
    def _auto_fill(cls, kwargs: Dict[str, Any]) -> bool:
        sizes = cls._infer_size(kwargs)

        for name, batching in cls.batching.items():
            if (name in kwargs) and (kwargs[name] is not None):
                continue

            if batching.default is None:
                continue

            if isinstance(batching.shape, tuple):
                size = list(batching.shape)
            else:
                size = [batching.shape]

            for i, s in enumerate(size):
                if isinstance(s, str):
                    if s not in sizes:
                        break
                    size[i] = sizes[s]
            else:
                if name in sizes:
                    kwargs[name] = torch.full(
                        size, fill_value=sizes[name], dtype=batching.dtype
                    )
                else:
                    kwargs[name] = torch.full(
                        size, fill_value=batching.default, dtype=batching.dtype
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
