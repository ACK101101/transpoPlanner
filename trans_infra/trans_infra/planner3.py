import tqdm
import gc
from itertools import combinations
from collections import deque
import networkx as nx
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
import trans_infra.trans_infra.utils as utils
from trans_infra.trans_infra.dqn import DQN
from trans_infra.trans_infra.gnn import EdgeRegressionModel
from trans_infra.trans_infra.simulator import run_simulation_parallel

#! fix viz
#? only 1 action at a time, test other 3 options to get better labels?

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
        # training params
        self.criterion = criterion
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.compute_eps()
        self.batch_size = batch_size
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
        self.refresh_sim(self.path)
        self.memory = deque([], maxlen=memory_size)
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

    def refresh_sim(self, path):
        self.path = path
        print(self.path)
        self.graph = utils.load_network(self.path)
        self.data2nx = dict(zip(range(self.graph.number_of_nodes()),
                            self.graph.nodes()))

    def compute_eps(self):
        return self.epsilon_min + (self.epsilon_start - self.epsilon_min) \
                                * np.exp(-1. * self.global_step / self.epsilon_decay)
    
    def select_actions(self, edge_scores):
        """Use Gumbel-Softmax to sample edges"""
        edge_probs = F.gumbel_softmax(edge_scores, self.epsilon)
        max_elements, _ = torch.max(edge_probs, axis=1)
        edge_idxs = torch.argsort(max_elements)[-self.num_sample_actions:]
        action_vecs = edge_scores[edge_idxs]
        action_values, action_idxs = torch.max(action_vecs, axis=1)
        actions_sum = torch.sum(action_values)
        
        return edge_idxs, action_idxs, actions_sum
    
    def find_sym_idx(self, data, u, v):
        i = -1
        for i in range(data.edge_index.shape[1]):
            u2, v2 = data.edge_index[0][i].item(), data.edge_index[1][i].item()
            if u == v2 and v == u2:
                break
        return i
        
    def take_actions(self, data, edge_idxs, action_idxs):
        """modify data object to reflect action"""
        new_data = Data(x=data.x.clone(), edge_index=data.edge_index.clone(), 
                        edge_attr=data.edge_attr.clone(), batch=data.batch.clone())
        
        for j in range(len(action_idxs)):
            edge_idx = edge_idxs[j]
            action_idx = action_idxs[j].clone().item()
            # offset action index by u, v, speed, capacity
            u, v = data.edge_index[0][edge_idx].item(), data.edge_index[1][edge_idx].item()
            sym_edge_index = self.find_sym_idx(data, u, v)
            
            # get edge type from map
            edge_type = self.idx2gen_edges[action_idx]
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
        
        # get actions from edge_scores
        edge_idxs, action_idxs, actions_sum = self.select_actions(edge_scores)
        # modify edge attrs by actions
        new_data = self.take_actions(data, edge_idxs, action_idxs)
        
        return edge_idxs, action_idxs, actions_sum, new_data
    
    def modify_graph(self, edge_idxs, action_idxs, data):
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
        
        return u_list, v_list
        
    def compute_and_store_rewards(self, buffer, curr_score_df, next_score_df, data, 
                                  pred_action_val, next_state):
        next_state_detach = next_state.detach()
        for j in range(len(curr_score_df)):
            reward = next_score_df["social_score"][j] - curr_score_df["social_score"][j]
            done = False if j != len(curr_score_df)-1 else True
            buffer.append((data, pred_action_val, reward, next_state_detach, done))
        del next_state
        
        return buffer
    
    def sample_env(self, data, isNew=False):
        """get (s, a, r, s_+1) observation and save to replay"""
        # refresh sim if new
        if isNew:
            self.refresh_sim(data.path[0])
            
        # run sim and get reward in curr state
        curr_score_df = run_simulation_parallel(self.pop_size, self.episode_len, 
                                                self.graph, self.num_runs)
        
        # forward
        edge_idxs, action_idxs, actions_sum, next_state = self(data)
        
        # modify networkx graph
        _, _ = self.modify_graph(edge_idxs, action_idxs, data)
        
        # run sim and get reward in curr state
        next_score_df = run_simulation_parallel(self.pop_size, self.episode_len, 
                                                self.graph, self.num_runs)
        
        self.memory = self.compute_and_store_rewards(self.memory, curr_score_df, next_score_df,
                                                     data, actions_sum, next_state)
        
        self.log('mean rewards', next_score_df['social_score'].mean() - curr_score_df['social_score'].mean())

    def get_data_from_sim(self, data):
        print("getting data\n")
        for _ in tqdm.tqdm(range(self.num_episodes)):
            isNew = False if data.path[0] == self.path else True
            self.sample_env(data, isNew)

    def get_data_from_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        
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
        
        return state_batch, next_state_batch, actions, rewards, dones

    def clean_up_batch(self, states, actions, rewards, next_states, dones):
        del states
        del actions
        del rewards
        del next_states
        del dones

    def embed_from_state_batch(self, state_batch):
        state_edge_scores, state_graph_embeds =  self.edge_model(state_batch.x, state_batch.edge_index, 
                                                                 state_batch.edge_attr, state_batch.batch)
        return state_edge_scores, state_graph_embeds

    def training_step(self, data):
        
        while len(self.memory) < self.batch_size:
            self.get_data_from_sim(data)
        
        if self.global_step % 8 == 0:
            self.get_data_from_sim(data)

        replay_batch = random.sample(self.memory, self.batch_size)
        state_batch, next_state_batch, actions, rewards, dones = self.get_data_from_batch(replay_batch)
        
        _, state_graph_embeds = self.embed_from_state_batch(state_batch)
        _, next_state_graph_embeds = self.embed_from_state_batch(next_state_batch)

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
        
        self.clean_up_batch(state_batch, next_state_batch, actions, rewards, dones)
        
        if self.global_step % 25 == 0:
            self.epsilon = self.compute_eps()
            gc.collect()
            torch.mps.empty_cache()
            ("cleared up space?")
        
        if self.global_step % 50 == 0:
            self.update_target_model()
            print(" updated target model")
        
        print("done with train step")
        output = {'loss': loss, 'dqn_train_loss': dqn_loss, 'edge_train_loss': edge_loss}

        return output

    def validation_step(self, data):
        self.refresh_sim(data.path[0])
        
        for _ in range(self.num_episodes):

            curr_score_df = run_simulation_parallel(self.pop_size, self.episode_len, 
                                                    self.graph, self.num_runs)
            
            edge_idxs, action_idxs, pred_action_val, next_state = self(data)
            

            _, _ = self.modify_graph(edge_idxs, action_idxs, data)
            

            next_score_df = run_simulation_parallel(self.pop_size, self.episode_len, 
                                                    self.graph, self.num_runs)

            val_batch = []
            val_batch = self.compute_and_store_rewards(val_batch, curr_score_df, next_score_df, 
                                                       data, pred_action_val, next_state)
            
            self.log('mean val rewards', next_score_df['social_score'].mean() - curr_score_df['social_score'].mean())
        
        state_batch, next_state_batch, actions, rewards, dones = self.get_data_from_batch(val_batch)
        
        _, state_graph_embeds = self.embed_from_state_batch(state_batch)
        _, next_state_graph_embeds = self.embed_from_state_batch(next_state_batch)

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
        
        self.clean_up_batch(state_batch, next_state_batch, actions, rewards, dones)
        print("done with val step")
        output = {'val_loss': loss, 'dqn_val_loss': dqn_loss, 'edge_val_loss': edge_loss}
        
        return output

    def test_step(self, data):
        # refresh sim
        self.refresh_sim(data.path[0])
        
        start_val, end_val = 0, 0
        for i in range(self.num_episodes):
            print(f"episode {i}")

            curr_score_df = run_simulation_parallel(self.pop_size, self.episode_len, 
                                                    self.graph, self.num_runs)

            edge_idxs, action_idxs, pred_action_val, next_state = self(data)
            print(f"    action_idxs: {action_idxs}")
            print(f"    pred_action_val: {pred_action_val}")
            
            u_list, v_list = self.modify_graph(edge_idxs, action_idxs, data)
            
            utils.make_trans_graph(self.graph, u_list, v_list)
            
            next_score_df = run_simulation_parallel(self.pop_size, self.episode_len, 
                                                    self.graph, self.num_runs)

            # save test data
            test_batch = []
            test_batch = self.compute_and_store_rewards(test_batch, curr_score_df, next_score_df, 
                                                       data, pred_action_val, next_state)
            
            if i == 0:
                start_val = curr_score_df['social_score'].mean()
            elif i == self.num_episodes - 1:
                end_val = next_score_df['social_score'].mean()
                
            print(f"    avg score change: {next_score_df['social_score'].mean() - curr_score_df['social_score'].mean()}")
        
        print(f"total score change: {end_val - start_val}")
        # compute loss
        state_batch, next_state_batch, actions, rewards, dones = self.get_data_from_batch(test_batch)
        
        _, state_graph_embeds = self.embed_from_state_batch(state_batch)
        _, next_state_graph_embeds = self.embed_from_state_batch(next_state_batch)
        print(f"    state graph embed similarity: {utils.mean_cosine_similarity(state_graph_embeds)}")
        print(f"    next graph embed similarity: {utils.mean_cosine_similarity(next_state_graph_embeds)}")

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

        return output

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
        path = self.logger.root_dir + f"/version_{self.logger.version}/"
        torch.save(self.memory, path + 'replay_memory.pth')
        print(f"saving replay")
        return checkpoint
    
    def bootstrap_memory(self, bootstrap):
        # path = "./logs/planner2replay_memory.pth"
        self.memory = torch.load(bootstrap)
        print(f"bootstrap replay from {bootstrap}")
    
    def on_load_checkpoint(self, checkpoint):
        self.memory = checkpoint['replay_memory']
        # path = self.logger.root_dir + f"{self.logger.name}/" + f"version_{self.logger.version}/"
        # self.memory = torch.load(path + 'replay_memory.pth')
        print(f"loaded replay memory")
        
                    