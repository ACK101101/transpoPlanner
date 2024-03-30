import torch
import torch_geometric.nn as gnn
import torch.nn as nn
import torch.nn.functional as F
import random
from itertools import combinations


class NormalizeFeatures:
    """normalize specified column indices across data obj"""
    def __init__(self, node_cols=None, edge_cols=None):
        self.node_cols = node_cols
        self.edge_cols = edge_cols

    def __call__(self, x, edge_attr):
        if self.node_cols is not None:
            for col in self.node_cols:
                mean = x[:, col].mean()
                std = x[:, col].std()
                x[:, col] = (x[:, col] - mean) / (std + 1e-8)

        if self.edge_cols is not None:
            for col in self.edge_cols:
                mean = edge_attr[:, col].mean()
                std = edge_attr[:, col].std()
                edge_attr[:, col] = (edge_attr[:, col] - mean) / (std + 1e-8)

        return x, edge_attr
        
        
class EdgeConditionedConvolution(nn.Module):
    """GNN block: updates node and edge embeddings"""
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim, dropout_p=0.2):
        super(EdgeConditionedConvolution, self).__init__()
        self.conv = gnn.NNConv(node_dim, out_dim, 
                               nn=nn.Sequential(nn.Linear(edge_dim, hidden_dim),
                                                nn.BatchNorm1d(hidden_dim),
                                                nn.Dropout(p=dropout_p),
                                                nn.LeakyReLU(),
                                                nn.Linear(hidden_dim, hidden_dim*out_dim),
                                                nn.BatchNorm1d(hidden_dim*out_dim),
                                                nn.LeakyReLU())
                                )

    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        # graph conv
        x = self.conv(x, edge_index, edge_attr)
        
        return x


class EdgeRegressionModel(nn.Module):
    """Links GNN layers and regression head for edge-wise and graph vector representations"""
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim, dropout_p=0.1):
        super(EdgeRegressionModel, self).__init__()
        # preprocess
        self.normalize = NormalizeFeatures([0, 6, 7], [0, 2, 7, 8, 9])
        self.node_encode = nn.Sequential(nn.Linear(node_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.Dropout(p=dropout_p),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.LeakyReLU())
        self.edge_encode = nn.Sequential(nn.Linear(edge_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.Dropout(p=dropout_p),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.LeakyReLU())
        # conv
        self.conv1 = EdgeConditionedConvolution(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        self.conv2 = EdgeConditionedConvolution(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        # postprocess
        self.regressor = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Dropout(p=dropout_p),
                                       nn.LeakyReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.LeakyReLU(),
                                       nn.Linear(hidden_dim, out_dim)
                                       )
        self.graph_encode = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.Dropout(p=dropout_p),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.LeakyReLU())
        self.decode = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.Dropout(p=dropout_p),
                                    nn.LeakyReLU(),
                                    nn.Linear(hidden_dim, hidden_dim))
        # # pool
        # self.pool1 = gnn.SAGPooling(hidden_dim, ratio=0.5)
        # self.conv3 = EdgeConditionedConvolution(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        # self.pool2 = gnn.SAGPooling(hidden_dim, ratio=1)

    def forward(self, x, edge_index, edge_attr, batch) -> tuple[torch.Tensor, torch.Tensor]:
        # normalize features
        x, edge_attr = self.normalize(x, edge_attr)
        
        # encode
        x = self.node_encode(x)
        edge_attr = self.edge_encode(edge_attr)
        
        # convolutional steps
        x = self.conv1(x, edge_index, edge_attr, batch)
        x = F.normalize(x)
        x = self.conv2(x, edge_index, edge_attr, batch)
        x = F.normalize(x)
        
        # edge predictions
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]
        x_edge = torch.cat([x_i, x_j], dim=1)
        edge_scores = self.regressor(x_edge)
        
        # graph predictions
        x = self.graph_encode(x)
        graph_embed = gnn.global_mean_pool(x, batch)
        graph_embed = self.decode(graph_embed)
        
        return edge_scores, graph_embed
