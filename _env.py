import random
import networkx as nx
import numpy as np
import gym
from gym import spaces
from tqdm import tqdm
from typing import OrderedDict

from lnd import LNDRouting

class LNEnv(gym.Env): 
    LND_RISK_FACTOR = 0.000000015
    A_PRIORI_PROB = 0.6
 
    def shortest_path(self, u, v, proto='dijkstra'):
        try:
            if proto == 'LND':
                if not self.routingObj:
                    self.routingObj = LNDRouting()
                path = self.routingObj.routePath(self.g, u, v, 100)['path']
            else:
                path = nx.shortest_path(self.g, u, v, method=proto)
        except:
            path = []
        return path

    def __init__(self, L, transactions, G, max_steps=10, train=True) -> None:
        self.max_steps = max_steps
        self.subset = transactions
        self.train = train
        self.G = G
        self.g = L
 
        self.action_size = self.max_steps - 1
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_size, ), dtype=np.float32)
        self.observation_size = len(self.g.nodes)
        observation_spaces = {
          'observation': spaces.Box(low=0, high=1, shape=(self.observation_size * 2, ), dtype=np.float32),
        }     
        self.observation_space = spaces.Dict(observation_spaces) 
        
        self.id_to_idx = {}
        self.index = []
        for idx, id in enumerate(self.g.nodes):
            self.id_to_idx[id] = idx
            self.index += [self.g.nodes[id]['index']]
        self.index = np.array(self.index)
        
        self.reward = []
        self.heatmap = []
        self.routingObj = None
        
        if self.train:
            routed = []
            for tx in tqdm(self.subset, desc='Routing..'):
                u, v, amount = tx[0], tx[1], tx[2]
                guided_path = self.shortest_path(u, v)
                routed.append((u, v, amount, guided_path))
            self.subset = routed        

    def get_observation(self):
        observation = OrderedDict([
          ('observation', np.hstack((self.index, self.current_observation)))
        ])
        return observation
    
    def reset(self):
        tx = random.choice(self.subset)
        self.u, self.v, self.amount, self.guided_path = tx[0], tx[1], tx[2], tx[3]     
        self.path = [self.u]
        self.agent_pos_idx = self.id_to_idx[self.u]
        self.target_pos_idx = self.id_to_idx[self.v]
        self.current_observation = np.zeros((self.observation_size, ), dtype=np.float32)
        self.current_observation[self.target_pos_idx] = 0.5
        self.current_observation[self.agent_pos_idx] = 1
        return self.get_observation()   
            
    def step(self, action):
        reward = 0
        done = True
        self.current_observation = np.zeros((self.observation_size, ), dtype=np.float32)
        self.current_observation[self.target_pos_idx] = 0.5
        self.current_observation[self.agent_pos_idx] = 1

        for idx, a in enumerate(action):
            neighbors = [n for n in self.g.neighbors(self.path[-1]) if n not in self.path]
            neighbors_count = len(neighbors)
            if neighbors_count:
                direction = a / 2 + 0.5 # rescale to 0..1
                direction = int(direction * (neighbors_count - 1))
                next_node = neighbors[direction]
                self.path += [next_node]
                self.current_observation[self.id_to_idx[next_node]] = 1
            else:
                break 

        if self.train:
                self.heatmap += self.path
                reward = self.compute_reward() 
                self.reward.append(reward)          
        
        return self.get_observation(), reward, done, {}

    def render(self, mode='console'):
        if mode != 'console':
          raise NotImplementedError()
        pass

    def close(self):
        pass
        
    def check_path(self):
        for i in range(0, len(self.path) - 1):
            if not self.g.has_edge(self.path[i], self.path[i + 1]):
                return False
            if self.path[i] == self.v or self.path[i + 1] == self.v:
                return True
        return False 

    def get_cost(self, u, v, amount):
        e = self.g.edges[u, v]
        if 'fee_base_sat' in e:
            fee = e['fee_base_sat'] + amount * e['fee_rate_sat']
            alt = (amount + fee) * e["delay"] * self.LND_RISK_FACTOR + fee
            return -alt / 1000
        elif 'weight' in e:
            return -e['weight'] / 1000
        else:
            return 0

    def get_path_cost(self):
        total_cost = 0
        for i in range(1, len(self.path) - 1):
            if self.g.has_edge(self.path[i], self.path[i + 1]):
                total_cost += self.get_cost(self.path[i], self.path[i + 1], self.amount)
        return total_cost / len(self.path)

    def get_guided_bonus(self):
        length = min(len(self.path), len(self.guided_path))
        right_steps = 0
        for i in range(1, length):
            if self.path[i] == self.guided_path[i]:
                right_steps += 1 
        return (0.99 * 10 * 32 ** right_steps) / (32 ** len(self.guided_path))
   
    def compute_reward(self):
        reward = 0
        done = self.check_path()
        if done:
            reward += 10
            reward += self.get_path_cost()
        reward += self.get_guided_bonus()
        return reward
         
    def get_reward(self):
        return self.reward
        
    def get_path(self):
        if self.v in self.path:
            return self.path[:self.path.index(self.v) + 1]
        return self.path
        
    def get_heatmap(self):
        unique, counts = np.unique(self.heatmap, return_counts=True)
        return np.asarray((unique, counts)).T