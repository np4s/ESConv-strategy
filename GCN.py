import torch

import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=128):
        super(GCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.gc1 = GraphConvolution(num_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, 64)
        self.gc3 = GraphConvolution(64, num_classes)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.log_softmax(self.gc3(x, adj), dim=1)
        return x.max(1)[1]


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(
            torch.zeros(size=(in_features, out_features))))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(
                torch.zeros(size=(out_features,))))
        else:
            self.register_parameter('bias', None)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias

        return torch.sparse.mm(adj, x)

