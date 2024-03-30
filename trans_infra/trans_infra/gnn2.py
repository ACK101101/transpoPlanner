import torch
import torch_geometric.nn as gnn
import torch.nn as nn
import torch.nn.functional as F
from trans_infra.trans_infra.utils import EdgeSelector, NormalizeFeatures


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.2):
        super(MLPBlock, self).__init__()
        # layer 1
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(p=dropout_p)
        # layer 2
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.BatchNorm1d(output_dim)
        self.drop2 = nn.Dropout(p=dropout_p)
    
    def forward(self, x):
        # layer 1
        x = self.lin1(x)
        x = self.norm1(x)
        x = self.drop1(x)
        x = F.leaky_relu(x)
        # layer 2
        x = self.lin2(x)
        x = self.norm2(x)
        x = self.drop2(x)
        x = F.leaky_relu(x)
        return x
        
        
class EdgeConditionedConvolution(nn.Module):
    """GNN block: updates node and edge embeddings"""
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, dropout_p=0.2):
        super(EdgeConditionedConvolution, self).__init__()
        self.conv = gnn.NNConv(node_dim, output_dim, 
                               nn=MLPBlock(edge_dim, hidden_dim, 
                                           hidden_dim*output_dim, 
                                           dropout_p=dropout_p))

    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        # graph conv
        return self.conv(x, edge_index, edge_attr)


class EdgeRegressionModel(nn.Module):
    """Links GNN layers and regression head for edge-wise and graph vector representations"""
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, dropout_p=0.1):
        super(EdgeRegressionModel, self).__init__()
        # preprocess
        self.normalize = NormalizeFeatures([0, 6, 7], [0, 2, 7, 8, 9])
        self.node_encode = MLPBlock(node_dim, hidden_dim, hidden_dim, 
                                    dropout_p=dropout_p)
        self.edge_encode = MLPBlock(edge_dim, hidden_dim, hidden_dim, 
                                    dropout_p=dropout_p)
        # conv
        self.conv1 = EdgeConditionedConvolution(hidden_dim, hidden_dim, hidden_dim, 
                                                hidden_dim)
        self.conv2 = EdgeConditionedConvolution(hidden_dim, hidden_dim, hidden_dim, 
                                                hidden_dim)
        # postprocess
        self.regressor = MLPBlock(hidden_dim*2, hidden_dim, output_dim, 
                                  dropout_p=dropout_p)
        self.graph_encode = MLPBlock(hidden_dim, hidden_dim, hidden_dim, 
                                     dropout_p=dropout_p)
        self.graph_decode = MLPBlock(hidden_dim, hidden_dim, hidden_dim, 
                                     dropout_p=dropout_p)

    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        # normalize features
        print(f"    x {x.shape}: {x[0]}")
        print(f"    edge_attr {edge_attr.shape}: {edge_attr[0]}")
        x_norm, edge_attr_norm = self.normalize(x, edge_attr)
        print(f"    x_norm {x_norm.shape}: {x_norm[0]}")
        print(f"    edge_attr_norm {edge_attr_norm.shape}: {edge_attr_norm[0]}")
        
        # encode
        x_encode = self.node_encode(x_norm)
        edge_attr_encode = self.edge_encode(edge_attr_norm)
        print(f"    x_encode {x_encode.shape}: {x_encode[0]}")
        print(f"    edge_attr_encode {edge_attr_encode.shape}: {edge_attr_encode[0]}")
        
        # convolutional steps
        x_conv1 = self.conv1(x_encode, edge_index, edge_attr_encode, batch)
        x_conv1 = F.normalize(x_conv1)
        print(f"    x_conv1 {x_conv1.shape}: {x_conv1[0]}")
        x_conv2 = self.conv2(x_conv1, edge_index, edge_attr_encode, batch)
        x_conv2 = F.normalize(x_conv2)
        print(f"    x_conv2 {x_conv2.shape}: {x_conv2[0]}")
        
        # edge predictions
        x_i = x_conv2[edge_index[0]]
        x_j = x_conv2[edge_index[1]]
        x_edges = torch.cat([x_i, x_j], dim=1)
        edge_score_logits = self.regressor(x_edges)
        
        # graph predictions
        graph_encode = self.graph_encode(x_conv2)
        graph_pool = gnn.global_mean_pool(graph_encode, batch)
        graph_decode = self.decode(graph_pool)
        
        return edge_score_logits, graph_decode
