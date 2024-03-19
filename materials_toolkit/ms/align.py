import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean

from typing import Tuple

offset_range = torch.arange(-5, 6, dtype=torch.float32)
offsets = torch.stack(
    torch.meshgrid(offset_range, offset_range, offset_range, indexing="xy"), dim=3
).view(-1, 3)


def _get_shortest_paths(
    x_src: torch.FloatTensor,
    cell_src: torch.FloatTensor,
    x_dst: torch.FloatTensor,
    num_atoms: torch.LongTensor,
) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    idx = torch.arange(num_atoms.shape[0], dtype=torch.long, device=num_atoms.device)
    batch = idx.repeat_interleave(num_atoms)

    offset = offsets.clone().to(x_src.device)
    offset_euc = torch.einsum("ij,ljk->lik", offset, cell_src)

    paths = x_dst[:, None] + offset_euc[batch] - x_src[:, None]

    distance = paths.norm(dim=2)

    shortest_idx = distance.argmin(dim=1)
    idx = torch.arange(shortest_idx.shape[0], dtype=torch.long, device=idx.device)
    shortest_path = paths[idx, shortest_idx]

    return shortest_path


def _to_euc(x: torch.FloatTensor, cell: torch.FloatTensor, batch: torch.LongTensor):
    return torch.einsum("ij,ijk->ik", x, cell[batch])


def _to_inner(x: torch.FloatTensor, cell: torch.FloatTensor, batch: torch.LongTensor):
    return torch.einsum("ij,ijk->ik", x, cell[batch].inverse())


def _center_around_zero(x: torch.FloatTensor) -> torch.FloatTensor:
    return (x + 0.5) % 1.0 + 0.5


def polar(a: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    w, s, vh = torch.linalg.svd(a)
    u = w @ vh
    return u, (vh.mT.conj() * s[:, None, :]) @ vh


def rmsd(
    cell_src: torch.FloatTensor,
    x_src: torch.FloatTensor,
    cell_dst: torch.FloatTensor,
    x_dst: torch.FloatTensor,
    num_atoms: torch.LongTensor,
) -> float:
    batch_atoms = torch.arange(cell_src.shape[0], dtype=torch.long).repeat_interleave(
        num_atoms
    )

    _, cell_src = polar(cell_src)
    _, cell_dst = polar(cell_dst)
    x_src = _center_around_zero(x_src)
    x_dst = _center_around_zero(x_dst)

    x_src_euc = _to_euc(x_src, cell_src, batch_atoms)
    x_dst_euc = _to_euc(x_dst, cell_dst, batch_atoms)

    paths = _get_shortest_paths(x_src_euc, cell_src, x_dst_euc, num_atoms)

    avg_path = scatter_mean(paths, batch_atoms, dim=0, dim_size=num_atoms.shape[0])

    distance = (paths - avg_path[batch_atoms]).norm(dim=1)

    return scatter_mean(distance, batch_atoms, dim=0, dim_size=num_atoms.shape[0])
