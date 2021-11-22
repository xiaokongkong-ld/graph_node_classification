import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from hgcn_conv import HGCNConv,HypAct
import layers.hyp_layers_new as hyp_layers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import manifolds
from layers.att_layers import GraphAttentionLayer
from layers.layers import GraphConvolution, Linear, get_dim_act
import utils_hpy.math_utils as pmath
from layers.layers import GraphConvolution, Linear

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MLP, self).__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(1000, 600)
        self.lin2 = Linear(600, 200)
        self.lin3 = Linear(200, 86)
        self.lin4 = Linear(86, 16)
        self.lin5 = Linear(16, 7)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin4(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin5(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(1433, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 7)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        # return x
        return F.log_softmax(x)

class HGCN_pyg(torch.nn.Module):
    """
    Hyperbolic-GCN.
    """
    def __init__(self, c, hidden_channels):
        super(HGCN_pyg, self).__init__()
        torch.manual_seed(1234567)
        self.c = c
        # self.manifold = getattr(manifolds, 'Euclidean')()
        self.manifold = getattr(manifolds, 'Hyperboloid')()
        # self.manifold = getattr(manifolds, 'PoincareBall')()
        act = getattr(F, 'relu')
        self.hconv1 = HGCNConv(self.manifold, 1433, hidden_channels, self.c)
        self.hconv2 = HGCNConv(self.manifold, hidden_channels, 7, self.c)
        self.hyp_act = HypAct(self.manifold, self.c, act)
        self.bn = torch.nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index):

        x = self.hconv1(x, edge_index)
        x = self.hyp_act(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.hconv2(x, edge_index)

        return x
        # return F.log_softmax(x)

class HGCN(torch.nn.Module):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c):
        super(HGCN, self).__init__()
        self.c = c
        # self.manifold = getattr(manifolds, 'Euclidean')()
        self.manifold = getattr(manifolds, 'Hyperboloid')()
        # self.manifold = getattr(manifolds, 'PoincareBall')()
        act = getattr(F, 'relu')
        # self.lin = Linear(64, 7, dropout=0.0, act=act, use_bias=True)
        self.hgcov1 = hyp_layers.HyperbolicGraphConvolution(self.manifold, 1433, 100, 1, 1, 0.0, act, 1, 0, 0)
        self.hgcov2 = hyp_layers.HyperbolicGraphConvolution(self.manifold, 100, 7, 1, 1, 0.0, act, 1, 0, 0)
        self.encode_graph = True

    def forward(self, x, adj):
        # print('......................xtan............................')

        x_tan = self.manifold.proj_tan0(x, 1)

        x_hyp = self.manifold.expmap0(x_tan, 1)
        # print(x_hyp)
        x_hyp = self.manifold.proj(x_hyp, 1)
        h = self.hgcov1.forward(x_hyp, adj)
        h = self.hgcov2.forward(h, adj)

        h = self.manifold.proj_tan0(self.manifold.logmap0(h, c=self.c), c=self.c)
        # h = self.lin(h)
        return F.log_softmax(h)
        # return h