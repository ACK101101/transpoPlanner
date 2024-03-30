import networkx as nx
import osmnx as ox
import random
from itertools import combinations
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeSelector(nn.Module):
    def __init__(self):
        super(EdgeSelector, self).__init__()

    def forward(self, edge_scores, k, temperature):
        # Apply Gumbel-Softmax to get a probability distribution over edges
        gumbel_vec = -torch.empty_like(edge_scores).exponential_().log()
        print(f"    gumbel_vec {gumbel_vec.shape}")
        gumbel_vec = (edge_scores + gumbel_vec) / temperature
        edge_probs = F.softmax(gumbel_vec, dim=0)
        print(f"    edge_probs {edge_probs.shape}")

        # Select top-k edges using Straight-Through Gumbel-Softmax
        _, edge_indices = torch.topk(edge_probs, k, dim=0)
        print(f"    edge_indices {edge_indices.shape}")
        edge_onehot = torch.zeros_like(edge_scores).scatter_(0, edge_indices.view(-1, 1), 1.0)
        edge_onehot_sg = edge_onehot - edge_probs.detach() + edge_probs
        
        # Select corresponding action indices (maximum feature) for the selected edges
        _, action_indices = torch.max(edge_scores[edge_indices], dim=1)
        print(f"    action_indices {action_indices.shape}")
        action_onehot = torch.zeros_like(edge_scores).scatter_(1, action_indices.unsqueeze(-1), 1.0)
        action_onehot_sg = action_onehot - edge_probs.detach() + edge_probs

        return edge_onehot_sg, action_onehot_sg, edge_indices.view(-1, 1), action_indices.unsqueeze(-1)


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
    
    
def load_network(path) -> nx.Graph:
    """loads in modified OSM graph"""
    G_trans = ox.load_graphml(
                path,
                node_dtypes={'idx':int, 'x':float, 'y':float, 'general0':float, 'general1':float, 
                            'general2':float, 'general3':float, 'general4':float},    
                edge_dtypes={'u':int, 'v':int, 'speed':float, 'capacity':float, 'length':float,
                            'general0':float, 'general1':float, 'general2':float, 'general3':float})
    G_trans = G_trans.to_undirected()
    return G_trans

def make_trans_graph(graph, u_list, v_list):
    osmid_2_idx = {k: v for (k, v) in zip(graph.nodes, range(len(graph.nodes))) }
    
    nodes, edges = graph.nodes(data=True), graph.edges(data=True)
    layout = { n[0] : [ n[1]['x'], n[1]['y'] ] for n in graph.nodes(data=True) }
    
    node_color = [ v['color'] for (k, v) in nodes ]
    edge_color = [ d['color'] for (u, v, d) in edges ]      
    edge_weight = [ 0.5 if u not in u_list or v not in v_list else 3 for (u, v, d) in edges ]
    node_weight = [1] * len(nodes)

    nx.draw_networkx(graph, pos=layout,
                    edge_color=edge_color, 
                    node_color=node_color,
                    width=edge_weight,
                    node_size=node_weight,
                    with_labels=False)

def cosine_similarity(A, B):
    return torch.dot(A, B) / (torch.norm(A) * torch.norm(B))

def mean_cosine_similarity(embeds, num_edge_samples=0):
    """check if GNN is over smoothing"""
    score = 0
    if num_edge_samples != 0:
        sampled = random.sample(list(combinations(list(range(embeds.shape[0])), 2)), 
                                num_edge_samples)
        for i in tqdm.tqdm(range(num_edge_samples)):
            u, v = sampled[i][0], sampled[i][1]
            score += cosine_similarity(embeds[u], embeds[v])
        
        return score / num_edge_samples
    else:
        for i in range(embeds.shape[0]):
            for j in range(embeds.shape[0]):
                if i != j:  score += cosine_similarity(embeds[i], embeds[j])
                
        return score / embeds.shape[0] ** 2