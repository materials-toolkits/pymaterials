import torch
import torch.nn as nn
from torch_geometric import data


class FilterNobleGas(nn.Module):
    def __init__(self):
        super().__init__()

        self.noble_z = nn.Parameter(
            torch.tensor([2, 10, 18, 36, 54, 86, 118], dtype=torch.long),
            requires_grad=False,
        )

    def __call__(self, struct=data.Data | data.HeteroData) -> bool:
        return (self.noble_z[:, None] != struct.z[None, :]).all()
