import torch


def order_edges(idx: torch.LongTensor) -> torch.LongTensor:
    assert idx.ndim in (1, 2)

    if idx.ndim == 1:
        return idx.sort(dim=0).values

    for i in range(idx.shape[0]):
        perm = idx[-i - 1].argsort(stable=True)
        idx = idx[:, perm]

    return idx
