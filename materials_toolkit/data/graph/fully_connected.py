import torch
import torch.nn.functional as F


def indices(n: int, device: torch.device) -> torch.LongTensor:
    return torch.arange(n, dtype=torch.long, device=device)


def indices_like(x: torch.Tensor) -> torch.LongTensor:
    return torch.arange(x.shape[0], dtype=torch.long, device=x.device)


def offset(num: torch.LongTensor) -> torch.LongTensor:
    return F.pad(num.cumsum(0), (1, 0))


def fully_connected(
    num_nodes: torch.LongTensor,
    mask: torch.BoolTensor = None,
    self_loop: bool = True,
    directed: bool = True,
) -> torch.LongTensor:
    if mask is None:
        num_edges = num_nodes.pow(2)
        batch_edges = indices_like(num_edges).repeat_interleave(num_edges, dim=0)
    else:
        num_edges = num_nodes[mask].pow(2)
        batch_edges = indices_like(num_nodes)[mask].repeat_interleave(num_edges, dim=0)

    edges_offset = offset(num_edges)
    nodes_offset = offset(num_nodes)

    if mask is None:
        idx_dec = batch_edges
    else:
        idx_dec = indices_like(num_nodes[mask]).repeat_interleave(num_edges, dim=0)

    idx_edges = indices_like(batch_edges) - edges_offset[idx_dec]

    src_edges = idx_edges // num_nodes[batch_edges]
    tgt_edges = idx_edges % num_nodes[batch_edges]

    if (not self_loop) and (not directed):
        mask = src_edges < tgt_edges
        batch_edges = batch_edges[mask]
        src_edges, tgt_edges = src_edges[mask], tgt_edges[mask]

    elif not directed:
        mask = src_edges <= tgt_edges
        batch_edges = batch_edges[mask]
        src_edges, tgt_edges = src_edges[mask], tgt_edges[mask]

    elif not self_loop:
        mask = src_edges != tgt_edges
        batch_edges = batch_edges[mask]
        src_edges, tgt_edges = src_edges[mask], tgt_edges[mask]

    edges = torch.stack((src_edges, tgt_edges), dim=0)
    batched_edges = edges + nodes_offset[batch_edges]

    return batched_edges
