import mesa
import networkx as nx
import osmnx as ox
import numpy as np
import functools
from scipy.sparse import dok_matrix

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def social_score(model):
    """Calculate node connectivity of the k_core of the social graph"""
    # TODO: penalize by islands removed in k_core
    if model.time != 0 and model.time % 23 == 0:
        social_graph = nx.from_scipy_sparse_array(model.A_social)
        k_core = nx.k_core(social_graph)
        return nx.node_connectivity(k_core)
    else:
        return 0

class TransInfraNetworkModel(mesa.Model):
    """Transit & place network inhabited by agents"""
    def __init__(self, num_agents, graph_file) -> None:
        super().__init__()
        # init environment
        self.graph_file = graph_file
        self.G_trans = self.load_network()                                  # load in modified OSM network
        self.nodes = self.G_trans.nodes(data=True)
        self.space = mesa.space.NetworkGrid(self.G_trans)                   # built-in class for network spaces
        self.time = 0                                                       # 60 minute time increments
        
        # save node types
        self.needs_type = ["sleep", "work", "social"]
        self.sleep_nodes = [k for k, v in self.nodes if v['general'] == 'sleep']
        self.work_nodes = [k for k, v in self.nodes if v['general'] == 'work']
        self.social_nodes = [k for k, v in self.nodes if v['general'] == 'social']
        self.node_types = {"sleep": self.sleep_nodes, "work": self.work_nodes, 
                           "social": self.social_nodes}
        # set global resource thresholds
        self.thresh_sleep, self.thresh_work, self.thresh_social = 8, 8, 8   # sets thresh for agent-wise needs (ie. sleep)
        self.threshes = np.array([self.thresh_sleep, self.thresh_work, self.thresh_social])
        self.social_prob = 0.9  # ? what to set random social prob to?
        
        # init agents
        self.num_agents = num_agents
        self.A_social = dok_matrix((self.num_agents, self.num_agents))      # sparse matrix representation for social ties adjacency list
        self.schedule = mesa.time.RandomActivation(self)                    # controls move order of agents @ each time step
        self.populate()
        self.shortest_path.cache_clear()                                    # clears cache after precomputing commutes
        
        # built-in class for collecting info for analysis / viz
        self.datacollector = mesa.DataCollector(model_reporters={'Social Score': social_score},
                                                agent_reporters={})
        self.datacollector.collect(self)
    
    def noncar_weight(self, u:int, v:int, d: dict):
        if d[0]["general"] == "pedbike" or d[0]["general"] == "bus" or d[0]["general"] == "other":
            return d[0]["length"] / d[0]["speed"]
        else:
            return None
    
    def car_weight(self, u:int, v:int, d: dict):
        if d[0]["general"] == "car":
            return d[0]["length"] / d[0]["speed"]
        else:
            return None
    
    @functools.cache
    def shortest_path(self, source, target) -> tuple[int, list[int]]:
        # look for noncar path
        try:
            nc_dist, nc_path = nx.bidirectional_dijkstra(self.G_trans,
                                                source, target,
                                                weight=self.noncar_weight)
        except:
            nc_dist, nc_path = np.inf, []
        # look for car path
        try:
            c_dist, c_path = nx.bidirectional_dijkstra(self.G_trans,
                                                source, target,
                                                weight=self.car_weight)
        except:
            c_dist, c_path = np.inf, []
        # if no path
        if nc_dist == np.inf and c_dist == np.inf:  
            print("uh oh no path")
            return np.inf, []
        # if at least one path, return shortest
        return (nc_dist, nc_path) if nc_dist <= c_dist else (c_dist, c_path)
        
    def populate(self) -> None:
        """populate network with agents, assign home and workplace, add each to schedule """
        for i in range(self.num_agents):
            sleep_node = np.random.choice(self.sleep_nodes)
            work_node = np.random.choice(self.work_nodes)
            a = PersonAgent(i, self, sleep_node, work_node)
            self.schedule.add(a)
            self.space.place_agent(a, sleep_node)
            a.update_dist_costs()                   # init agent target 
    
    def step(self) -> None:
        """defines what happens each global time step"""
        self.schedule.step()
        self.datacollector.collect(self)
        self.time += 1
    
    def load_network(self) -> nx.Graph:
        """loads in modified OSM graph"""
        G_trans = ox.load_graphml(
                    self.graph_file,
                    node_dtypes={'building':str, 'general':str, 'color':str, 
                                 'x':float, 'y':float, 'geometry':str},
                    edge_dtypes={'osmid':int, 'highway':str, 'general':str, 'color':str, 
                                 'length':float, 'speed':float, 'geometry':str})
        G_trans = G_trans.to_undirected()                                   # make undirected
        return G_trans


class PersonAgent(mesa.Agent):
    """An agent that moves around the network collecting resources"""

    def __init__(self, unique_id: int, model: mesa.Model, home: int, work: int) -> None:
        super().__init__(unique_id, model)
        # save home & work node ids, precompute & save commute
        self.home, self.work = home, work
        self.commute_dist, self.commute = self.model.shortest_path(self.home, self.work)
        # init model resource counts
        self.curr_sleep, self.curr_work, self.curr_social = 0, 0, 0
        self.curr_resources = np.array([self.curr_sleep, self.curr_work, self.curr_social])
        # save path finding search results
        self.sleep_path, self.work_path, self.social_path = None, None, None
        self.sleep_dist, self.work_dist, self.social_dist = np.inf, np.inf, np.inf
        # save agent goal info
        self.dist_costs = []
        self.t_need = None      # most urgent need
        self.path = []          # path to target node
        self.dist_cost = 0      # cost of path to target node
    
    def step(self) -> None:
        """Defines what happens each local time step"""
        # TODO: add traffic / congestion
            # pre load next edge to simulate congestion?
        
        # check if at target node, if yes clear path and update
        if [self.pos] == self.path:
            self.path = []
            self.update_dist_costs()
            
        # check if can talk, talk if true
        canTalk = self.canTalk()
        if canTalk:
            self.talk()
        
        # if en route to a target node, keep moving
        if len(self.path) > 1:  
            # print(f"{self.model.time} Agent {self.unique_id} moving")
            self.move()
        # if at target node and node type matches target need
        elif self.t_need == self.id_2_resource(self.pos):
            # print(f"{self.model.time} Agent {self.unique_id} collecting")
            self.collect()
            self.update_t_need()
        # if target node type and target resource don't match, find new target
        else:
            # print(f"{self.model.time} Agent {self.unique_id} finding new path")
            self.set_new_path()
    
    def id_2_resource(self, node_id: int) -> str:
        return self.model.nodes[node_id]["general"]
    
    def update_t_need(self) -> None:
        # select most urgent need weighted by extra costs
        scale_dist_cost = (np.array(self.dist_costs) / max(self.dist_costs)) * 8
        urgency = self.model.threshes / (self.curr_resources + scale_dist_cost + 1)
        need_idx = max(enumerate(urgency), key=lambda x: x[1])[0]
        self.t_need = self.model.needs_type[ need_idx ]
    
    def update_dist_costs(self) -> None:
        # TODO: make sure edge weights are augmented by traffic
        # TODO: store old values and only update new ones
            # init with costs to other places from sleep
            # if collected -> update value only for resource collected
            # if moved -> don't update
            # if set new path -> that is this func 
            # ^^^^^^
        """Computes next path based on global resources thresholds,
            current resource counts, and costs/benefits of traveling"""

        # if at sleep
        if self.id_2_resource(self.pos) == "sleep":
            self.social_dist, self.social_path = self.search("social")
            self.dist_costs = [ 0, self.commute_dist, self.social_dist ]
        # if at work
        elif self.id_2_resource(self.pos) == "work":
            self.social_dist, self.social_path = self.search("social")
            self.dist_costs = [ self.commute_dist, 0, self.social_dist ]
        # if at social
        else:
            self.sleep_dist, self.sleep_path = self.search("sleep")
            self.work_dist, self.work_path = self.search("work")
            self.dist_costs = [ self.sleep_dist, self.work_dist, 0 ]
        
        self.update_t_need()
            
    def search(self, need_type: str) -> tuple[int, list[int]]:
        # TODO: scale total_ties
        # TODO: if no path found, stay
        """Searches for closest node within target class"""
        
        min_cost, min_path = np.inf, []
        for t_node in self.model.node_types[need_type]:
            # get shortest path
            cost, path = self.model.shortest_path(self.pos, t_node)
            if need_type == "social":
                # get friends at node
                agents_at_node = self.model.space.get_cell_list_contents([t_node])
                total_ties = sum([ self.model.A_social[self.unique_id, a.unique_id] 
                                    for a in agents_at_node ])
                cost = cost / (total_ties + 1)
            if cost < min_cost: 
                min_cost, min_path = cost, path
        return min_cost, min_path
    
    def set_new_path(self) -> None:
        # set path based on most urgent need
        if self.id_2_resource(self.pos) == "sleep" and self.t_need == "work": 
            self.path = self.commute
            self.dist_cost = self.commute_dist
        elif self.id_2_resource(self.pos) == "work" and self.t_need == "sleep": 
            start = len(self.commute)-1
            self.path = [ self.commute[i] for i in range(start, -1, -1) ]
            self.dist_cost = self.commute_dist
        elif self.id_2_resource(self.pos) == "social":
            if self.t_need == "sleep":
                self.path = self.sleep_path
                self.dist_cost = self.sleep_dist
            else:
                self.path = self.work_path
                self.dist_cost = self.work_dist
        else:   # need social
            self.path = self.social_path
            self.dist_cost = self.social_dist
            
    def move(self) -> None:
        """Move to next node in path"""
        
        # if time to target <= 1 time step, move to target
        if self.dist_cost <= 1:
            self.path = self.path[-1:]
            self.model.space.move_agent(self, self.path[-1])
        # else, calc where can go in 1 time step
        else:
            time_step_left, steps = 1, -1
            while time_step_left > 0:
                steps += 1
                edge = self.model.G_trans.edges[ self.path[0], self.path[1], 0 ]
                time_step_left -= (edge["length"] / edge["speed"])
            
            self.path = self.path[steps:]
            self.model.space.move_agent(self, self.path[0])
    
    def collect(self) -> None:
        """increment resource counts
            IF at social node, update ties"""
            
        need_idx = self.model.needs_type.index(self.t_need)
        self.curr_resources[need_idx] += 1
    
    def canTalk(self) -> bool:
        """Check if agent can talk at node"""
        
        nodemates = self.model.space.get_cell_list_contents([self.pos])
        resource_tag = self.id_2_resource(self.pos)
        # can't socialize if no other agents or if at other node
        if len(nodemates) <= 1 or resource_tag == "other": 
            return False
        rand = np.random.rand(1)[0]
        # chance of socializing at work or home
        if (resource_tag == "work" or resource_tag == "sleep") and rand < self.model.social_prob:
            return False
        # can socialize at social node or at work / home when >= 0.5
        return True
         
    def talk(self):
        """Chose agent in same node to build social tie with"""
        nodemates = self.model.space.get_cell_list_contents([self.pos])
        # prevent from choosing self
        nodemates.remove(self)
        # rank nodemates by A_social 
        nodemate_ties = [ self.model.A_social[self.unique_id, mate.unique_id] for mate in nodemates ]
        rank_d = { k: v for (k, v) in zip(nodemates, nodemate_ties) }
        # choose best friend
        other = max(rank_d, key=rank_d.get)
        # increment ties
        self.model.A_social[self.unique_id, other.unique_id] += 1
        self.model.A_social[other.unique_id, self.unique_id] += 1
            