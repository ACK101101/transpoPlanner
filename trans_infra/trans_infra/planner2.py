import tqdm
import gc
from itertools import combinations
from collections import deque
import osmnx as ox
import networkx as nx
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from trans_infra.trans_infra.gnn import EdgeRegressionModel
from trans_infra.trans_infra.simulator import run_simulation_parallel

#! add loss to motivate higher scores? problem is IDK MAXIMUM? how to deal with this **
    #! track change in final score across osm

#! fix paths for checkpointing
#! fix viz

#! Make Q net take in edge embed not graph


class DQN(nn.Module):
    """f(s) -> q : takes concat graph embeddings of states and predicts their q-vals"""
    def __init__(self, graph_dim, hidden_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(graph_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.fc4 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
class DQNLightning(pl.LightningModule):
    def __init__(self, node_dim, edge_dim, erm_hidden_dim, graph_dim, hidden_dim, action_dim, 
                 num_sample_actions, criterion, gamma, epsilon_start, epsilon_decay, epsilon_min, 
                 batch_size, memory_size, learning_rate, dataset, pop_size, episode_len, 
                 num_episodes, num_runs, bootstrap=None):
        super(DQNLightning, self).__init__()
        # models
        self.edge_model = EdgeRegressionModel(node_dim, edge_dim, erm_hidden_dim, action_dim)
        self.q_net = DQN(graph_dim, hidden_dim)
        self.t_net = DQN(graph_dim, hidden_dim)
        # training
        self.criterion = criterion
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.compute_eps()
        self.batch_size = batch_size
        self.memory = deque([], maxlen=memory_size)
        self.learning_rate = learning_rate
        self.dataset = dataset
        # maps
        self.idx2gen_edges = {0: 'car', 1: 'bus', 2: 'pedbike', 3: 'other'}         
        self.gen2speed = {"car": 48300, "pedbike": 16100, "bus": 48300, "other": 16100}
        self.gen2capacity = {"car": 1100, "pedbike": 8000, "bus": 6000, "other": 5000}
        self.gen2color_edges = {"car": "red", "pedbike": "green", "bus": "yellow", "other": "gray"}
        # init sim
        self.pop_size = pop_size
        self.episode_len = episode_len
        self.num_episodes = num_episodes
        self.num_sample_actions = num_sample_actions
        self.num_runs = num_runs
        self.path = "./osm_dataset/raw/copenhagen.osm"
        self.load_sim()
        if bootstrap: self.bootstrap_memory(bootstrap)

    def setup(self, stage):
        """splits data list into train and val datasets"""
        generator = torch.Generator().manual_seed(42)
        train_data, val_data = torch.utils.data.random_split(self.dataset, [0.8, 0.2], generator)
        self.train_dataset = train_data
        self.val_dataset = val_data
    
    def configure_optimizers(self):
        optimizer = optim.Adam(list(self.edge_model.parameters()) + list(self.q_net.parameters()), 
                               lr=self.learning_rate)
        return optimizer

    def load_network(self) -> nx.Graph:
        """loads in modified OSM graph"""
        G_trans = ox.load_graphml(
                    self.path,
                    node_dtypes={'idx':int, 'x':float, 'y':float, 'general0':float, 'general1':float, 
                                'general2':float, 'general3':float, 'general4':float},    
                    edge_dtypes={'u':int, 'v':int, 'speed':float, 'capacity':float, 'length':float,
                                'general0':float, 'general1':float, 'general2':float, 'general3':float})
        G_trans = G_trans.to_undirected()
        return G_trans

    def load_sim(self):
        self.graph = self.load_network()
        self.data2nx = dict(zip(range(self.graph.number_of_nodes()),
                            self.graph.nodes()))

    def compute_eps(self):
        return self.epsilon_min + (self.epsilon_start - self.epsilon_min) \
                                * np.exp(-1. * self.global_step / self.epsilon_decay)
    
    def select_action(self, edge_scores) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """return best actions given edge_scores"""
        # if explore randomly choose k edges
        if np.random.rand() <= self.epsilon:
            edge_idxs = np.random.choice(edge_scores.shape[0], 
                                       self.num_sample_actions, replace=False)
            action_vecs = edge_scores[edge_idxs]
            actions = torch.max(action_vecs, 1)
            return edge_idxs, actions.indices, actions.values
        # if exploit choose k edges with highest max element
        max_elements = torch.max(edge_scores, axis=1)
        edge_idxs = torch.argsort(max_elements.values)[-self.num_sample_actions:]
        action_vecs = edge_scores[edge_idxs]
        actions = torch.max(action_vecs, 1)
        return edge_idxs, actions.indices, actions.values
    
    def take_actions(self, data, edge_idxs, action_idxs):
        """modify data object to reflect action"""
        new_data = Data(x=data.x.clone(), edge_index=data.edge_index.clone(), 
                        edge_attr=data.edge_attr.clone(), batch=data.batch.clone())
        
        for j in range(len(action_idxs)):
            edge_idx = edge_idxs[j]
            action_idx = action_idxs[j].clone().item()
            # offset action index by u, v, speed, capacity
            u, v = data.edge_index[0][edge_idx].item(), data.edge_index[1][edge_idx].item()
            
            i = -1
            for i in range(data.edge_index.shape[1]):
                u2, v2 = data.edge_index[0][i].item(), data.edge_index[1][i].item()
                if u == v2 and v == u2:
                    break
            
            # get edge type from map
            edge_type = self.idx2gen_edges[action_idx]
            # do same for v, u
            sym_edge_index = i
            # change edge type encoding            
            new_data.edge_attr[edge_idx, 3:7] = 0               # TODO: make indexing not hardcoded
            new_data.edge_attr[edge_idx][3+action_idx] = 1
            new_data.edge_attr[sym_edge_index, 3:7] = 0
            new_data.edge_attr[sym_edge_index][3+action_idx] = 1
            # change speed
            new_data.edge_attr[edge_idx][8] = self.gen2speed[edge_type]
            new_data.edge_attr[sym_edge_index][8] = self.gen2speed[edge_type]
            # change capacity
            new_data.edge_attr[edge_idx][9] = self.gen2capacity[edge_type]
            new_data.edge_attr[sym_edge_index][9] = self.gen2capacity[edge_type]
        
        return new_data
    
    def forward(self, data):
        # get state from graph_embedding
        edge_scores, curr_state = self.edge_model(data.x, data.edge_index, 
                                                  data.edge_attr, data.batch)
        # print(f"    edge_scores: {edge_scores[5]}")
        # print(f"    curr_state: {curr_state}")
        
        # get actions from edge_scores
        edge_idxs, action_idxs, action_vals = self.select_action(edge_scores)
        # print(f"    action_idxs: {action_idxs}")
        # print(f"    action_vals: {action_vals}")
        # modify edge attrs by actions
        new_data = self.take_actions(data, edge_idxs, action_idxs)
        
        pred_action_val = torch.sum(action_vals)
        
        return edge_idxs, action_idxs, pred_action_val, new_data
        
    def sample_env(self, data, isNew=False):
        """get (s, a, r, s_+1) observation and save to replay"""
        # refresh sim if new
        if isNew:
            self.path = data.path[0]
            print(self.path)
            self.load_sim()
            
        # run sim and get reward in curr state
        curr_score_df = run_simulation_parallel(self.pop_size, self.episode_len, 
                                                self.graph, self.num_runs)
        
        # forward
        edge_idxs, action_idxs, pred_action_val, next_state = self(data)
        
        # modify networkx graph
        for i in range(len(edge_idxs)):
            edge_idx = edge_idxs[i]
            action_idx = action_idxs[i].clone().item()
            u, v = data.edge_index[0][edge_idx].item(), data.edge_index[1][edge_idx].item()
            u_p, v_p = self.data2nx[u], self.data2nx[v]
            edge_type = self.idx2gen_edges[action_idx]
            self.graph.edges[u_p, v_p, 0]["general"] = edge_type
            self.graph.edges[u_p, v_p, 0]["speed"] = self.gen2speed[edge_type]
            self.graph.edges[u_p, v_p, 0]["capacity"] = self.gen2capacity[edge_type]
            self.graph.edges[u_p, v_p, 0]["color"] = self.gen2color_edges[edge_type]
        
        # run sim and get reward in curr state
        next_score_df = run_simulation_parallel(self.pop_size, self.episode_len, 
                                                self.graph, self.num_runs)
        
        for j in range(len(curr_score_df)):
            reward = next_score_df["social_score"][j] - curr_score_df["social_score"][j]
            done = False if j != len(curr_score_df)-1 else True
            self.store_transition(data, pred_action_val, reward, next_state, done)
            # print(f"    pred_action_val: {pred_action_val}")
            # print(f"    reward: {reward}")
        
        self.log('mean rewards', next_score_df['social_score'].mean() - curr_score_df['social_score'].mean())

    def get_data(self, data):
        print("getting data\n")
        for _ in tqdm.tqdm(range(self.num_episodes)):
            isNew = False if data.path[0] == self.path else True
            self.sample_env(data, isNew)

    def training_step(self, data):
        while len(self.memory) < self.batch_size:
            self.get_data(data)
        
        if self.global_step % 4 == 0:
            self.get_data(data)
    
        replay_batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*replay_batch)
        
        states = [state.to(self.device) for state in states]
        next_states = [next_state.to(self.device) for next_state in next_states]
        state_batch = Data(x=torch.cat([data.x for data in states]),
                           edge_index=torch.cat([data.edge_index + i * states[0].num_nodes \
                               for i, data in enumerate(states)], dim=1),
                           edge_attr=torch.cat([data.edge_attr for data in states]),
                           batch=torch.cat([torch.full((data.num_nodes,), i, dtype=torch.long) \
                               for i, data in enumerate(states)])).to(self.device)
        next_state_batch = Data(x=torch.cat([data.x for data in next_states]),
                                edge_index=torch.cat([data.edge_index + i * next_states[0].num_nodes \
                                    for i, data in enumerate(next_states)], dim=1),
                                edge_attr=torch.cat([data.edge_attr for data in next_states]),
                                batch=torch.cat([torch.full((data.num_nodes,), i, dtype=torch.long) \
                                    for i, data in enumerate(next_states)])).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.Tensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.Tensor(dones).unsqueeze(1).to(self.device)
        
        state_edge_scores, state_graph_embeds = self.edge_model(state_batch.x, state_batch.edge_index, 
                                                                state_batch.edge_attr, state_batch.batch)
        next_state_edge_scores, next_state_graph_embeds = self.edge_model(next_state_batch.x, next_state_batch.edge_index, 
                                                                next_state_batch.edge_attr, next_state_batch.batch)

        q_values = self.q_net(state_graph_embeds)

        with torch.no_grad():
            next_q_values = self.t_net(next_state_graph_embeds)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        dqn_loss = self.criterion(q_values, target_q_values)
        edge_loss = self.criterion(actions, q_values)
        loss = dqn_loss + edge_loss

        # Logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('dqn_loss', dqn_loss, prog_bar=True)
        self.log('edge_loss', edge_loss, prog_bar=True)
        self.log('epsilon', self.epsilon)
        
        if self.global_step % 25 == 0:
            self.epsilon = self.compute_eps()
        
        if self.global_step % 50 == 0:
            self.update_target_model()
            print(" updated target model")
            gc.collect()
            print("gc")
        print("done with train step")
        
        output = {'loss': loss, 'dqn_train_loss': dqn_loss, 'edge_train_loss': edge_loss}
        # self.train_step_outputs.append(output)
        return output

    def validation_step(self, data):
        # refresh sim
        self.path = data.path[0]
        self.load_sim()
        
        for _ in range(self.num_episodes):
            # run sim and get reward in curr state
            curr_score_df = run_simulation_parallel(self.pop_size, self.episode_len, 
                                                    self.graph, self.num_runs)
            # forward
            edge_idxs, action_idxs, pred_action_val, next_state = self(data)
            
            # modify networkx graph
            for i in range(len(edge_idxs)):
                edge_idx = edge_idxs[i]
                action_idx = action_idxs[i].clone().item()
                u, v = data.edge_index[0][edge_idx].item(), data.edge_index[1][edge_idx].item()
                u_p, v_p = self.data2nx[u], self.data2nx[v]
                edge_type = self.idx2gen_edges[action_idx]
                self.graph.edges[u_p, v_p, 0]["general"] = edge_type
                self.graph.edges[u_p, v_p, 0]["speed"] = self.gen2speed[edge_type]
                self.graph.edges[u_p, v_p, 0]["capacity"] = self.gen2capacity[edge_type]
                self.graph.edges[u_p, v_p, 0]["color"] = self.gen2color_edges[edge_type]
            
            # run sim and get reward in curr state
            next_score_df = run_simulation_parallel(self.pop_size, self.episode_len, 
                                                    self.graph, self.num_runs)

            # save test data
            val_batch = []
            for j in range(len(curr_score_df)):
                reward = next_score_df["social_score"][j] - curr_score_df["social_score"][j]
                done = False if j != len(curr_score_df)-1 else True
                val_batch.append((data, pred_action_val, reward, next_state, done))
                  
        # compute loss
        states, actions, rewards, next_states, dones = zip(*val_batch)
        
        states = [state.to(self.device) for state in states]
        next_states = [next_state.to(self.device) for next_state in next_states]
        state_batch = Data(x=torch.cat([data.x for data in states]),
                           edge_index=torch.cat([data.edge_index + i * states[0].num_nodes \
                               for i, data in enumerate(states)], dim=1),
                           edge_attr=torch.cat([data.edge_attr for data in states]),
                           batch=torch.cat([torch.full((data.num_nodes,), i, dtype=torch.long) \
                               for i, data in enumerate(states)])).to(self.device)
        next_state_batch = Data(x=torch.cat([data.x for data in next_states]),
                                edge_index=torch.cat([data.edge_index + i * next_states[0].num_nodes \
                                    for i, data in enumerate(next_states)], dim=1),
                                edge_attr=torch.cat([data.edge_attr for data in next_states]),
                                batch=torch.cat([torch.full((data.num_nodes,), i, dtype=torch.long) \
                                    for i, data in enumerate(next_states)])).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.Tensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.Tensor(dones).unsqueeze(1).to(self.device)
        
        state_edge_scores, state_graph_embeds = self.edge_model(state_batch.x, state_batch.edge_index, 
                                                                state_batch.edge_attr, state_batch.batch)
        next_state_edge_scores, next_state_graph_embeds = self.edge_model(next_state_batch.x, next_state_batch.edge_index, 
                                                                          next_state_batch.edge_attr, next_state_batch.batch)

        q_values = self.q_net(state_graph_embeds)
        with torch.no_grad():
            next_q_values = self.t_net(next_state_graph_embeds)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        dqn_loss = self.criterion(q_values, target_q_values)
        edge_loss = self.criterion(actions, q_values)
        loss = dqn_loss + edge_loss
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('dqn_val_loss', dqn_loss, prog_bar=True)
        self.log('edge_val_loss', edge_loss, prog_bar=True)
        print("done with val step")
        
        output = {'val_loss': loss, 'dqn_val_loss': dqn_loss, 'edge_val_loss': edge_loss}
        # self.validation_step_outputs.append(output)
        return output

    def test_step(self, data):
        # refresh sim
        self.path = data.path[0]
        print(self.path)
        self.load_sim()
        
        start_val, end_val = 0, 0
        for i in range(self.num_episodes):
            print(f"episode {i}")
            # run sim and get reward in curr state
            curr_score_df = run_simulation_parallel(self.pop_size, self.episode_len, 
                                                    self.graph, self.num_runs)
            # forward
            edge_idxs, action_idxs, q_val, next_state = self(data)
            print(f"    action_idxs: {action_idxs}")
            print(f"    q_val: {q_val}")
            
            # modify networkx graph
            u_list, v_list = [], []
            for i in range(len(edge_idxs)):
                edge_idx = edge_idxs[i]
                action_idx = action_idxs[i].clone().item()
                u, v = data.edge_index[0][edge_idx].item(), data.edge_index[1][edge_idx].item()
                u_p, v_p = self.data2nx[u], self.data2nx[v]
                u_list.append(u_p)
                v_list.append(v_p)
                edge_type = self.idx2gen_edges[action_idx]
                self.graph.edges[u_p, v_p, 0]["general"] = edge_type
                self.graph.edges[u_p, v_p, 0]["speed"] = self.gen2speed[edge_type]
                self.graph.edges[u_p, v_p, 0]["capacity"] = self.gen2capacity[edge_type]
                self.graph.edges[u_p, v_p, 0]["color"] = self.gen2color_edges[edge_type]
                print(f"    type changed from {self.graph.edges[u_p, v_p, 0]['general']} to {self.idx2gen_edges[max_q_action_idx]}")
            
            self.make_trans_graph(u_list, v_list)
            
            # run sim and get reward in curr state
            next_score_df = run_simulation_parallel(self.pop_size, self.episode_len, 
                                                    self.graph, self.num_runs)

            # save test data
            test_batch = []
            for j in range(len(curr_score_df)):
                reward = next_score_df["social_score"][j] - curr_score_df["social_score"][j]
                test_batch.append((data, q_val, reward, next_state, False))
            
            if i == 0:
                start_val = curr_score_df['social_score'].mean()
            elif i == self.num_episodes - 1:
                end_val = next_score_df['social_score'].mean()
                
            print(f"    avg score change: {next_score_df['social_score'].mean() - curr_score_df['social_score'].mean()}")
        
        print(f"total score change: {end_val - start_val}")
        # compute loss
        states, actions, rewards, next_states, dones = zip(*test_batch)
        
        state_batch = Batch.from_data_list(states).to(self.device)
        next_state_batch = Batch.from_data_list(next_states).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.Tensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.Tensor(dones).unsqueeze(1).to(self.device)
        
        state_edge_scores, state_graph_embeds = self.edge_model(state_batch.x, state_batch.edge_index, 
                                                                state_batch.edge_attr, state_batch.batch)
        next_state_edge_scores, next_state_graph_embeds = self.edge_model(next_state_batch.x, next_state_batch.edge_index, 
                                                                          next_state_batch.edge_attr, next_state_batch.batch)

        print(f"    state graph embed similarity: {self.mean_cosine_similarity(state_graph_embeds)}")
        print(f"    next graph embed similarity: {self.mean_cosine_similarity(next_state_graph_embeds)}")

        q_values = self.q_net(state_graph_embeds)
        with torch.no_grad():
            next_q_values = self.t_net(next_state_graph_embeds)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        dqn_loss = self.criterion(q_values, target_q_values)
        edge_loss = self.criterion(actions, q_values)
        loss = dqn_loss + edge_loss
        
        print(f"    loss: {loss}")
        print(f"        dqn_loss: {dqn_loss}")
        print(f"        edge_loss: {edge_loss}")
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('dqn_test_loss', dqn_loss, prog_bar=True)
        self.log('edge_test_loss', edge_loss, prog_bar=True)
        print("done with test step")
        
        output = {'test_loss': loss, 'dqn_test_loss': dqn_loss, 'edge_test_loss': edge_loss}
        # self.validation_step_outputs.append(output)
        return output
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False)
    
    #! change for real test data
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False)
    
    def update_target_model(self):
        self.t_net.load_state_dict(self.q_net.state_dict())
        print("updated target model")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['replay_memory'] = self.memory
        path = self.logger.root_dir
        torch.save(self.memory, path + 'replay_memory.pth')
        print(f"saving replay")
        return checkpoint
    
    def bootstrap_memory(self, bootstrap):
        # path = "./logs/planner2replay_memory.pth"
        self.memory = torch.load(bootstrap)
        print(f"bootstrap replay from {bootstrap}")
    
    def on_load_checkpoint(self, checkpoint):
        self.memory = checkpoint['replay_memory']
        # self.memory = torch.load('./checkpoints/replay_memory.pth')
        print(f"loaded replay memory")
        
    def cosine_similarity(self, A, B):
        return torch.dot(A, B) / (torch.norm(A) * torch.norm(B))
    
    def mean_cosine_similarity(self, embeds, num_edge_samples=0):
        """check if GNN is over smoothing"""
        score = 0
        if num_edge_samples != 0:
            sampled = random.sample(list(combinations(list(range(embeds.shape[0])), 2)), 
                                    num_edge_samples)
            for i in tqdm.tqdm(range(num_edge_samples)):
                u, v = sampled[i][0], sampled[i][1]
                score += self.cosine_similarity(embeds[u], embeds[v])
            
            return score / num_edge_samples
        else:
            for i in range(embeds.shape[0]):
                for j in range(embeds.shape[0]):
                    if i != j:  score += self.cosine_similarity(embeds[i], embeds[j])
                    
            return score / embeds.shape[0] ** 2
        
    
    def make_trans_graph(self, u_list, v_list):
        osmid_2_idx = {k: v for (k, v) in zip(self.graph.nodes, range(len(self.graph.nodes))) }
        
        nodes, edges = self.graph.nodes(data=True), self.graph.edges(data=True)
        layout = { n[0] : [ n[1]['x'], n[1]['y'] ] for n in self.graph.nodes(data=True) }
        
        node_color = [ v['color'] for (k, v) in nodes ]
        edge_color = [ d['color'] for (u, v, d) in edges ]      
        edge_weight = [ 0.5 if u not in u_list or v not in v_list else 3 for (u, v, d) in edges ]
        node_weight = [1] * len(nodes)

        nx.draw_networkx(self.graph, pos=layout,
                        edge_color=edge_color, 
                        node_color=node_color,
                        width=edge_weight,
                        node_size=node_weight,
                        with_labels=False)
        
                    