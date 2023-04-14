import os, time, sys, gym, random, pickle, argparse
import networkx as nx
import numpy as np
import pandas as pd
from gym import spaces
from tqdm import tqdm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='_env', type=str)
parser.add_argument('--sample', default=0, type=int)
parser.add_argument('--subset', default='centralized', type=str)
parser.add_argument('--subgraph', default=100, type=int)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--timesteps', default=1e5, type=int)
parser.add_argument('--max_steps', default=10, type=int)
parser.add_argument('--attempts', default=100, type=int)
parser.add_argument('--approach', default='PPO', type=str)
args = parser.parse_args()

max_steps = args.max_steps
total_timesteps = args.timesteps
approach = args.approach
epochs = args.epochs
attempts = args.attempts
subgraph = args.subgraph
idx = args.sample
k = args.subset
if args.env == '_env':
    version='_env'
    from _env import LNEnv

def neighbors_count(G, id):
    return len(list(G.neighbors(id)))

def max_neighbors(G):
    max_neighbors = 0
    for id in G.nodes:
      max_neighbors = max(max_neighbors, neighbors_count(G, id))
    return max_neighbors

def test_path(u, v, amount=100, max_steps=10):
    e_.subset = [(u, v, amount, None)]
    obs = e_.reset()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = e_.step(action)
    if e_.check_path():
           return e_.get_path()  

base_dir = './'
snapshots_dir = os.path.join(base_dir, 'snapshots')
weights_dir = os.path.join(base_dir, 'weights')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

sample = nx.read_gpickle(os.path.join(snapshots_dir, f'graph-sample-{subgraph}.pickle'))
G = sample['undirected_graph']
print(f"undirected_graph, n: {len(G.nodes)} e: {len(G.edges)}")
print('available samples:')
for i in sample['samples'].keys():
        g = sample['samples'][i]['subgraph']
        print(f'i: {i}, n: {len(g.nodes)}, e: {len(g.edges)}, max_neighbors: {max_neighbors(g)}')
print('')

# max_steps=10
# total_timesteps=1e5
# approach='PPO'

for a in range(attempts):
    g = sample['samples'][idx]['subgraph']
    if k == 'randomized': #here we have 50/50 testset
        s = sample['samples'][idx]['subgraph_transactions'][k][:1000]
        s_ = sample['samples'][idx]['subgraph_transactions'][k][1000:]
    else:
        s = sample['samples'][idx]['subgraph_transactions'][k]
        s_ = s

    print(f"train {k}: i: {idx}, n: {len(g.nodes)}, e: {len(g.edges)}, max neighbors: {max_neighbors(g)}")
    print(f"{k} {len(s)} txs")
    e = LNEnv(g, s, G, max_steps=max_steps)
    e_ = LNEnv(g, [], G, max_steps=max_steps, train=False)
    #check_env(e)

    lf = os.path.join(results_dir, f'{approach}-{k}-{version}-{subgraph}-{idx}.log')
    log = pd.read_csv(lf, sep=';', compression='zip') if os.path.exists(lf) else None
    f = os.path.join(weights_dir, f'{approach}-{k}-{version}-{subgraph}-{idx}.sav')

    if approach == 'PPO':
        model_class = PPO
    else:
        model_class = None
        print(f'{approach} - not implemented!')
        raise ValueError

    if os.path.exists(f) and model_class:
        model = model_class.load(f, e, force_reset=False)
        print(f'{approach}-model is loaded: {f}')
    else:
        model = model_class("MultiInputPolicy", e) 
            
    for epoch in range(1, epochs + 1):
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

        train_score = 0 
        for tx in tqdm(s):
            r = test_path(tx[0], tx[1], tx[2])
            if r:
                train_score += 1
        train_score = train_score / len(s)
        mean_reward = np.mean(e.get_reward())
        print(f'train score: {train_score}, mean reward: {mean_reward}, epoch: {epoch}/{epochs}, v: {version}, a: {a}')
        model.save(f)
        model.save(f + f'{train_score:.3f}')
        print('saved:', f + f'{train_score:.3f}')

        test_score = 0
        if s != s_:
            test_score = 0 
            for tx in tqdm(s_):
                r = test_path(tx[0], tx[1], tx[2])
                if r:
                    test_score += 1
            test_score = test_score / len(s)
            print(f'test score: {test_score}')
        
        log = pd.concat([log, pd.DataFrame.from_dict({'time' : time.time(),
                                                'approach' : approach,
                                                'subset' : k,
                                                'subgraph' : subgraph,
                                                'idx' : idx,
                                                'train_score' : train_score,
                                                'test_score' : test_score,
                                                'mean_reward' : mean_reward,
                                                'epoch' : epoch,
                                                'epochs' : epochs,
                                                'version' : version,
                                                'max_steps' : max_steps,
                                                'total_timesteps' : total_timesteps,
                                                'filename' : f,
                                                'n': len(g.nodes),
                                                'e': len(g.edges),
                                                'max_neighbors':max_neighbors(g)}, orient='index').T], ignore_index=True)
        log.to_csv(lf, sep=';', index=False, compression='zip')
