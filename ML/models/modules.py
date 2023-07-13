import torch
from dgl.nn.pytorch import GraphConv, SAGEConv
from torch import nn
from dgl import ops
from dgl.nn.functional import edge_softmax



class ResidualModuleWrapper(nn.Module):
    def __init__(self, module, in_dim, out_dim, dropout, normalization, **kwargs):
        super().__init__()
        self.module = module(in_dim=in_dim, out_dim=out_dim,
                             dropout=dropout, normalization=normalization, **kwargs)

    def forward(self, graph, x):
        x_res = self.module(graph, x)
        if x.shape[1] == x_res.shape[1]:
            x_res = x + x_res
        return x_res


class FeedForwardModule(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x


class GCNModule(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, normalization, **kwargs):
        super().__init__()
        self.normalization = normalization(in_dim)
        self.feed_forward_module = FeedForwardModule(in_dim=in_dim,
                                                     out_dim=out_dim,
                                                     dropout=dropout)
        self.conv = GraphConv(in_dim, in_dim, norm='both', weight=False, bias=False, allow_zero_in_degree=True)

    def forward(self, g, h):
        h = self.normalization(h)
        h = self.conv(g, h)
        h = self.feed_forward_module(g, h)
        return h


class GCNSepModule(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, normalization, **kwargs):
        super().__init__()
        self.normalization = normalization(in_dim)
        self.feed_forward_module = FeedForwardModule(in_dim=2*in_dim,
                                                     out_dim=out_dim,
                                                     dropout=dropout)
        self.conv = GraphConv(in_dim, in_dim, norm='both', weight=False, bias=False, allow_zero_in_degree=True)

    def forward(self, g, h):
        h = self.normalization(h)
        message = self.conv(g, h)
        h = torch.cat([h, message], axis=1)
        h = self.feed_forward_module(g, h)
        return h


class SAGEModule(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, normalization):
        super().__init__()
        self.normalization = normalization(in_dim)
        self.conv = SAGEConv(in_dim, out_dim, 'mean')
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, h):
        h = self.normalization(h)
        h = self.conv(g, h)
        return self.dropout(h)


def _check_dim_and_num_heads_consistency(dim, num_heads):
    print(dim, num_heads)
    if dim % num_heads != 0:
        raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')


class GATModule(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, normalization, num_heads=4):
        super().__init__()
        self.dim = in_dim
        self.normalization = normalization(in_dim)
        _check_dim_and_num_heads_consistency(in_dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        self.input_linear = nn.Linear(in_features=in_dim, out_features=in_dim)

        self.attn_linear_u = nn.Linear(in_features=in_dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=in_dim, out_features=num_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.feed_forward_module = FeedForwardModule(in_dim=in_dim,
                                                     out_dim=out_dim,
                                                     dropout=dropout)

    def forward(self, graph, x):
        x = self.normalization(x)
        x = self.input_linear(x)

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.head_dim, self.num_heads)
        x = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.dim)

        x = self.feed_forward_module(graph, x)

        return x


class GATSepModule(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, normalization, num_heads=4):
        super().__init__()
        self.dim = in_dim
        self.normalization = normalization(in_dim)
        _check_dim_and_num_heads_consistency(in_dim, num_heads)
        self.dim = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        self.input_linear = nn.Linear(in_features=in_dim, out_features=in_dim)

        self.attn_linear_u = nn.Linear(in_features=in_dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=in_dim, out_features=num_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.feed_forward_module = FeedForwardModule(in_dim=2*in_dim,
                                                     out_dim=out_dim,
                                                     dropout=dropout)

    def forward(self, graph, x):
        x = self.normalization(x)
        x = self.input_linear(x)

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.head_dim, self.num_heads)
        message = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.dim)
        message = message.reshape(-1, self.dim)
        x = torch.cat([x, message], axis=1)

        x = self.feed_forward_module(graph, x)

        return x
