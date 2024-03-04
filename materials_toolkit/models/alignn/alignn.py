"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean

from .utils import RBFExpansion, BaseSettings
from materials_toolkit.data import StructureData

from typing import Tuple, Union, Literal
import os, json


class ALIGNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn"]
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False
    num_classes: int = 2

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(self, edge_index, node_feats, edge_feats) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """

        e_src = self.src_gate(node_feats)
        e_dst = self.dst_gate(node_feats)
        m = e_src[edge_index[0]] + e_dst[edge_index[1]] + self.edge_gate(edge_feats)

        sigma = torch.sigmoid(m)
        Bh = self.dst_update(node_feats)
        sum_sigma_h = scatter_add(
            Bh[edge_index[0]] * sigma,
            edge_index[1],
            dim=0,
            dim_size=node_feats.shape[0],
        )
        sum_sigma = scatter_add(
            sigma, edge_index[1], dim=0, dim_size=node_feats.shape[0]
        )
        h = sum_sigma_h / (sum_sigma + 1e-6)
        x = self.src_update(node_feats) + h

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(self, batch, x, y, z):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(batch.edge_index, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(batch.triplet_index, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class AtomicNumberEncoding(nn.Module):
    def __init__(self):
        super().__init__()

        feature_json = os.path.join(os.path.dirname(__file__), "atom_init.json")
        with open(feature_json, "r") as fp:
            features_data = json.load(fp)

        z_dim = max(map(int, features_data.keys())) + 1
        latent_dim = len(features_data["1"])
        features = torch.zeros((z_dim, latent_dim), dtype=torch.float)
        for z, f in features_data.items():
            features[int(z)] = torch.tensor(f, dtype=torch.float)

        self.features = nn.Parameter(features, requires_grad=False)

    def forward(self, z: torch.LongTensor) -> torch.FloatTensor:
        return self.features[z]


class ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.config = config
        self.classification = config.classification

        self.z_encoding = AtomicNumberEncoding()
        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(config.hidden_features, config.hidden_features)
                for idx in range(config.gcn_layers)
            ]
        )

        if self.classification:
            self.fc = nn.Linear(config.hidden_features, config.num_classes)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(avg_gap, dtype=torch.float).log()
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(self, batch: StructureData, latent: bool = False):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """

        src_node_idx, dst_node_idx = batch.edge_index
        edges_r = torch.bmm(
            torch.transpose(batch.cell[batch.batch_edges], 1, 2),
            (
                -batch.pos[src_node_idx] + batch.pos[dst_node_idx] + batch.target_cell
            ).unsqueeze(2),
        ).squeeze(2)

        edges_u = edges_r[batch.triplet_index[0]]
        edges_v = edges_r[batch.triplet_index[1]]
        triplets_cos = torch.sum(edges_u * edges_v, dim=1) / (
            torch.norm(edges_u, dim=1) * torch.norm(edges_v, dim=1)
        )

        z = self.angle_embedding(triplets_cos)
        x = self.atom_embedding(self.z_encoding(batch.z))
        y = self.edge_embedding(edges_r.norm(dim=1))

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(batch, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(batch.edge_index, x, y)

        # norm-activation-pool-classify
        h = scatter_mean(
            x, batch.batch_atoms, dim=0, dim_size=batch.num_structures.item()
        )

        if latent:
            return h

        out = self.fc(h)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return torch.squeeze(out)
