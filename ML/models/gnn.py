import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv, GATConv, GraphConv

from models.modules import GCNModule, SAGEModule, GATModule, GATSepModule, GCNSepModule, ResidualModuleWrapper

NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}

MODULES = {
    'GCN': GCNModule,
    'SAGE': SAGEModule,
    'GAT': GATModule,
    'GAT-sep': GATSepModule,
    'GCN-sep': GCNSepModule,
}


class GNN(torch.nn.Module):
    def __init__(self, feat_dim, h_dims, conv_name, dropout, normalization,
                 use_input_weighting, use_skip, **gnn_params):
        super().__init__()
        self.use_input_weighting = use_input_weighting
        self.use_skip = use_skip
        if use_input_weighting:
            self.input_linear = nn.Linear(in_features=feat_dim, out_features=h_dims[0])
            dims = h_dims
        else:
            dims = [feat_dim] + h_dims
        conv_module = MODULES[conv_name]
        self.convs = nn.ModuleList(
            [self.init_conv(use_skip, conv_module, dims[i], dims[i+1],
                            dropout, NORMALIZATION[normalization], **gnn_params) for i in range(len(dims)-1)]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.act = F.relu

    def init_conv(self, use_skip, module, in_dim, out_dim, dropout, normalization, **kwargs):
        if use_skip:
            return ResidualModuleWrapper(module, in_dim, out_dim, dropout, normalization, **kwargs)
        return module(in_dim, out_dim, dropout, normalization, **kwargs)

    def forward(self, g, h):
        # self.normalization = self.normalization(h)
        if self.use_input_weighting:
            h = self.input_linear(h)
            h = self.dropout(h)
            h = self.act(h)
        for conv in self.convs[:-1]:
            h = conv(g, h)
            h = self.act(h)
        return self.convs[-1](g, h)


class EdgeDecoder(torch.nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.lin1 = nn.Linear(2 * h_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, 1)
        self.act = F.relu

    def forward(self, edges, h):
        # edges: edge_num * 2
        # print(h.shape)
        # print(edges)
        src, dst = edges
        x = torch.cat([h[src], h[dst]], dim=-1).unsqueeze(1)
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        return x.view(-1)
