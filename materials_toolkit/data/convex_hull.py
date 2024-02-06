from __future__ import annotations
import warnings

import torch
from pymatgen.core import Element, Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

from abc import ABCMeta, abstractmethod
from typing import List, Dict, Set

import tqdm

from materials_toolkit.data.collate import separate

from .base import StructureData


class DatasetWithEnergy(metaclass=ABCMeta):
    def __init__(self):
        self.phase_diagram: Dict[str, PhaseDiagram] = {}

    @abstractmethod
    def entries(self, composition: torch.LongTensor) -> List[PDEntry]:
        pass

    @abstractmethod
    def compute_convex_hulls(self):
        pass

    def calculate_convex_hull(self, entry: Entry = None, z: torch.LongTensor = None):
        if entry is None:
            entry = DatasetWithEnergy.Entry(z)

        if entry in self.phase_diagram:
            return self.phase_diagram[entry]

        entries = self.entries(entry.atomic_numbers)
        try:
            diagram = PhaseDiagram(entries)
        except ValueError:
            diagram = None

        self.phase_diagram[entry] = diagram

        return self.phase_diagram[entry]

    def _calculate_e_above_hull(
        self, z: torch.LongTensor, energy_pa: torch.FloatTensor
    ) -> torch.FloatTensor:
        hull = self.calculate_convex_hull(z=z)

        if hull is None:
            return float("nan")

        total_energy = energy_pa.item() * z.shape[0]

        pd_entry = PDEntry(self.get_composition(z), total_energy)

        return hull.get_e_above_hull(pd_entry)

    def calculate_e_above_hull(
        self, struct: StructureData = None, entry: DatasetWithEnergy.Entry = None
    ) -> torch.FloatTensor:
        if entry is not None:
            hull = self.calculate_convex_hull(entry=entry)
            return hull.get_e_above_hull(entry.pd_entry)

        if struct.num_structures.item() > 1:
            energies = []
            for data in separate(struct, keys=["z", "energy_pa"]):
                energies.append(self._calculate_e_above_hull(data.z, data.energy_pa))
            energies = torch.tensor(energies, dtype=torch.float32)
        else:
            energies = torch.tensor(
                [self._calculate_e_above_hull(struct.z, struct.energy_pa)],
                dtype=torch.float32,
            )

        return energies.view(-1, 1)

    class Entry:
        def __init__(self, z: torch.LongTensor, energy_pa: torch.FloatTensor = None):
            self.z: torch.LongTensor = z
            self.energy_pa: torch.FloatTensor = energy_pa

        @property
        def atomic_numbers(self) -> torch.LongTensor:
            if hasattr(self, "_atomic_numbers"):
                return self._atomic_numbers

            self._atomic_numbers, self._count = self.z.unique(
                sorted=True, return_counts=True
            )
            return self._atomic_numbers

        @property
        def count(self) -> torch.LongTensor:
            if hasattr(self, "_count"):
                return self._count

            self._atomic_numbers, self._count = self.z.unique(
                sorted=True, return_counts=True
            )
            return self._count

        @property
        def composition(self) -> Composition:
            if hasattr(self, "_composition"):
                return self._composition

            self._composition = Composition(
                {z.item(): n.item() for z, n in zip(self.atomic_numbers, self.count)}
            )
            return self._composition

        @property
        def key(self) -> str:
            if hasattr(self, "_key"):
                return self._key

            self._key = "-".join(str(Element.from_Z(z)) for z in self.atomic_numbers)

            return self._key

        @property
        def pd_entry(self) -> PDEntry:
            if hasattr(self, "_pd_entry"):
                return self._pd_entry

            total_energy = self.z.shape[0] * self.energy_pa.item()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self._pd_entry = PDEntry(self.composition, total_energy)

            return self._pd_entry

        def __str__(self) -> str:
            return self.key

        def __repr__(self) -> str:
            return self.key

        def __len__(self) -> int:
            return self.atomic_numbers.shape[0]

        def __hash__(self) -> int:
            return hash(self.key)

        def __eq__(self, other: DatasetWithEnergy.Entry) -> bool:
            return (self.atomic_numbers == other.atomic_numbers).all()

        def __ne__(self, other: DatasetWithEnergy.Entry) -> bool:
            return not self.__eq__(other)

        def __contains__(self, other: DatasetWithEnergy.Entry) -> bool:
            if not isinstance(other, DatasetWithEnergy.Entry):
                return False

            return (
                (self.atomic_numbers[:, None] == other.atomic_numbers[None, :])
                .any(0)
                .all()
            )

        @classmethod
        def _recursive_build(
            cls,
            inclusion: Dict[DatasetWithEnergy.Entry, dict],
            entries: Dict[DatasetWithEnergy.Entry, List[DatasetWithEnergy.Entry]],
            entry: DatasetWithEnergy.Entry,
        ):
            new_key = True
            for key, subdict in inclusion.items():
                if key in entry and entry not in subdict:
                    new_key = False
                    cls._recursive_build(subdict, entries, entry)

            if new_key:
                inclusion[entry] = {}

        @classmethod
        def _recursive_add(
            cls,
            inclusion: Dict[DatasetWithEnergy.Entry, dict],
            filled_dict: Dict[DatasetWithEnergy.Entry, Set[DatasetWithEnergy.Entry]],
        ):
            for key, subdict in inclusion.items():
                if key not in filled_dict:
                    filled_dict[key] = {key}

                for subkey in subdict.keys():
                    if subkey not in filled_dict:
                        filled_dict[subkey] = {subkey}

                    filled_dict[subkey].update(filled_dict[key])

                cls._recursive_add(subdict, filled_dict)

        @classmethod
        def inclusion_graph(
            cls, entries: List[DatasetWithEnergy.Entry]
        ) -> Dict[str, List[DatasetWithEnergy.Entry]]:
            entries = sorted(entries, key=len)

            inclusion, dict_entries = {}, {}
            for entry in tqdm.tqdm(entries, desc="clusturise systems", leave=False):
                if entry in dict_entries:
                    dict_entries[entry].append(entry.pd_entry)
                else:
                    dict_entries[entry] = [entry.pd_entry]
                    cls._recursive_build(inclusion, dict_entries, entry)

            filled_dict = {}
            cls._recursive_add(inclusion, filled_dict)

            results = {}
            for key, included in filled_dict.items():
                results[key] = []
                for subkey in included:
                    results[key].extend(dict_entries[subkey])

            return results
