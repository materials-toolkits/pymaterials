import matplotlib.pyplot as plt
import torch

angles = torch.linspace(0, 2 * torch.pi, 4)[1:]
bases = torch.stack((torch.sin(angles), torch.cos(angles)), dim=1)

"""
comp = torch.tensor(
    [[1, 3, 9], [4, 3, 1], [2, 0, 3], [0, 1, 2], [0, 2, 5], [1, 1, 0]],
    dtype=torch.float,
)
x = (comp[:, :, None] * bases[None]).sum(dim=1) / comp.sum(dim=1)[:, None]
"""

n = 16
comp_list = []
for i in range(n):
    for j in range(n - i + 1):
        c = torch.tensor([i, j, n - i - j])
        div = torch.gcd(torch.gcd(c[[0]], c[[1]]), c[[2]])
        c //= div

        comp_list.append(c)
        # for mul in range(4):
        #    comp_list.append(c * (mul + 1))

comp = torch.stack(comp_list, dim=0)
print(comp.shape[0])
x = (comp.float()[:, :, None] * bases[None]).sum(dim=1) / comp.float().sum(dim=1)[
    :, None
]


plt.scatter(bases[:, 0], bases[:, 1], c="lime", zorder=100)
plt.scatter(x[:, 0], x[:, 1])
plt.plot(bases[[0, 1, 2, 0], 0], bases[[0, 1, 2, 0], 1], c="black")
plt.axis("off")
plt.axis("equal")
plt.show()
