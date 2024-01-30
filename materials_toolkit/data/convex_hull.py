from __future__ import annotations

import torch
from pymatgen.core import Element, Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

from abc import ABCMeta, abstractmethod
from typing import List, Dict

from materials_toolkit.data.collate import separate

from .base import StructureData



class DatasetWithEnergy(metaclass=ABCMeta):
    def __init__(self):
        self.phase_diagram: Dict[str, PhaseDiagram] = {}

    @abstractmethod
    def entries(self, composition: torch.LongTensor) -> List[PDEntry]:
        pass

    def get_key(self, z: torch.LongTensor) -> str:
        atomic_numbers, count = z.unique(sorted=True, return_counts=True)

        comp = Composition({z.item(): n.item() for z, n in zip(atomic_numbers, count)})

        return "-".join(sorted(z.symbol for z in comp.elements))

    def get_atomic_numbers(self, z: torch.LongTensor) -> torch.LongTensor:
        return z.unique(sorted=True)

    def get_composition(self, z: torch.LongTensor) -> Composition:
        atomic_numbers, count = z.unique(sorted=True, return_counts=True)

        return Composition({z.item(): n.item() for z, n in zip(atomic_numbers, count)})

    def calculate_convex_hull(self, z: torch.LongTensor):
        key = self.get_key(z)

        if key in self.phase_diagram:
            return self.phase_diagram[key]

        entries = self.entries(self.get_atomic_numbers(z))
        try:
            diagram = PhaseDiagram(entries)
        except ValueError:
            diagram = None

        self.phase_diagram[key] = diagram

        return self.phase_diagram[key]

    def _calculate_e_above_hull(
        self, z: torch.LongTensor, energy_pa: torch.FloatTensor
    ) -> torch.FloatTensor:
        hull = self.calculate_convex_hull(z)

        if hull is None:
            return float("nan")

        total_energy = energy_pa.item() * z.shape[0]

        entry = PDEntry(self.get_composition(z), total_energy)

        return hull.get_e_above_hull(entry)

    def calculate_e_above_hull(self, struct: StructureData) -> torch.FloatTensor:
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

            self._key = "-".join(sorted(z.symbol for z in self.composition.elements))

            return self._key

        @property
        def pd_entry(self) -> PDEntry:
            if hasattr(self, "_pd_entry"):
                return self._pd_entry

            total_energy = self.z.shape[0] * self.energy_pa.item()
            self._pd_entry = PDEntry(self.composition, total_energy)

            return self._pd_entry

        def __str__(self) -> str:
            return self.key

        def __len__(self) -> int:
            return self.atomic_numbers.shape[0]

        def __contains__(self, other: Entry) -> bool:
            if not isinstance(other, Entry):
                return False

            return (
                (self.atomic_numbers[:, None] == other.atomic_numbers[None, :]).any(0).all()
            )

        @classmethod
        def inclusion_graph(cls, entries: List[Entry]) -> Dict[str, List[Entry]]:
            sorted(entries, key=len)
            print(entries[:10])
            print(entries[-10:])
