import os, glob, sys, time, argparse, gym, random, pickle
import networkx as nx
import numpy as np
import pandas as pd
from gym import spaces
from tqdm import tqdm
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from codecarbon import EmissionsTracker, OfflineEmissionsTracker

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='_env', type=str)
parser.add_argument('--sample', default=0, type=int)
parser.add_argument('--subset', default='centralized', type=str)
parser.add_argument('--subgraph', default=100, type=int)
parser.add_argument('--change', default=False, type=str)
parser.add_argument('--em', default=False, type=bool)
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--timesteps', default=1e4, type=int)
parser.add_argument('--max_steps', default=10, type=int)
parser.add_argument('--approach', default='PPO', type=str)
parser.add_argument('--suffix', default='-', type=str)

args = parser.parse_args()

suffix = args.suffix
max_steps = args.max_steps
total_timesteps = args.timesteps
approach = args.approach
epochs = args.epochs
change = args.change
subgraph = args.subgraph
estimate_emission = args.em
idx = args.sample
k = args.subset
if args.env == '_env':
    version='_env'
    from _env import LNEnv

def shortest_path_length(G, u, v):
    path_len = 0
    try:
          path_len = len(nx.shortest_path(G, u, v))
    except:
          pass
    return path_len

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

sample = nx.read_gpickle(os.path.join(snapshots_dir, f'graph-sample-{subgraph}.pickle'))
G = sample['undirected_graph']
print(f"undirected_graph, n: {len(G.nodes)} e: {len(G.edges)}")

sample = nx.read_gpickle(os.path.join(base_dir, f'graph-sample-{subgraph}{suffix}.pickle'))

print('available samples:')
for i in sample['samples'].keys():
        g = sample['samples'][i]['subgraph']
        print(f'i: {i}, n: {len(g.nodes)}, e: {len(g.edges)}, max_neighbors: {max_neighbors(g)}')
print('')

if not change:
    changes = [k.split(suffix)[1] for k in sample['samples'][idx].keys() if suffix in k]
    changes.remove('1%')
    changes.remove('3%')
    changes.remove('7%')
    changes.remove('99%')
    print(f'changes: {changes}')
else:
    changes = [change]

for c in changes:
    g = sample['samples'][idx][f'subgraph{suffix}{c}']

    if k == 'randomized':
        s = sample['samples'][idx]['subgraph_transactions'][k][:1000]
        s_ = sample['samples'][idx]['subgraph_transactions'][k][1000:]
    else:
        s = sample['samples'][idx]['subgraph_transactions'][k]
        s_ = s

    print(f"train {k}: i: {idx}, n: {len(g.nodes)}, e: {len(g.edges)}, max_neighbors: {max_neighbors(g)}")
    print(f"{k} {len(s)} txs, change: {suffix}{c}")
    e = LNEnv(g, s, G, max_steps=max_steps)
    e_ = LNEnv(g, [], G, max_steps=max_steps, train=False)
    #check_env(e)

    lf = os.path.join(results_dir, f'{approach}-{k}-{version}-{subgraph}-{idx}{suffix}.log')
    log = pd.read_csv(lf, sep=';', compression='zip') if os.path.exists(lf) else None

    f = os.path.join(weights_dir, f'{approach}-{k}-{version}-{subgraph}-{idx}.sav')
    best_score_file_name = sorted(glob.glob(f + '*'))
    best_score_file_name = best_score_file_name[-1] if best_score_file_name else None
    best_score = float(best_score_file_name.split('.sav')[1]) if best_score_file_name else 0
    f = best_score_file_name

    if approach == 'PPO':
        model_class = PPO
    else:
        model_class = None
        print(f'{approach} - not implemented!')
        raise ValueError

    if os.path.exists(f):
        model = model_class.load(f, e, force_reset=False)
        print(f'loaded: {f}, best_score: {best_score}')
    else:
        model = model_class("MultiInputPolicy", e) 
            
    base_score = 0 
    for tx in tqdm(s):
        r = shortest_path_length(g, tx[0], tx[1])
        if r > 0 and r <= max_steps:
                base_score += 1
    base_score = base_score / len(s)

    train_score = 0 
    for tx in tqdm(s):
        r = test_path(tx[0], tx[1], tx[2])
        if r:
                train_score += 1
    train_score = train_score / len(s)
    print(f'pretrain score: {train_score}, base score: {base_score}, epoch: 0/{epochs}, v: {version}, c: {suffix}{c}')

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
                                                'mean_reward' : 0,
                                                'epoch' : 0,
                                                'epochs' : epochs,
                                                'version' : version,
                                                'max_steps' : max_steps,
                                                'total_timesteps' : total_timesteps,
                                                'filename' : f,
                                                'n': len(g.nodes),
                                                'e': len(g.edges),
                                                'max_neighbors':max_neighbors(g),
                                                'change' : c,
                                                'suffix' : suffix,
                                                'best_score' : best_score,
                                                'base_score' : base_score}, orient='index').T], ignore_index=True)
    log.to_csv(lf, sep=';', index=False, compression='zip')

    for epoch in range(1, epochs + 1):
        if os.path.exists(os.path.join(results_dir, 'emissions.csv')): 
            os.remove(os.path.join(results_dir, 'emissions.csv'))

        if estimate_emission:
            with OfflineEmissionsTracker(country_iso_code="CAN", measure_power_secs=1, tracking_mode='process') as tracker:
                model.learn(total_timesteps=total_timesteps, progress_bar=True)
        else:
            model.learn(total_timesteps=total_timesteps, progress_bar=True)

        em = {}
        if os.path.exists(os.path.join(results_dir, 'emissions.csv')): 
            em = pd.read_csv(os.path.join(results_dir, 'emissions.csv'))
            em = em[['timestamp', 'duration', 'emissions', 
                'emissions_rate', 'cpu_power', 'gpu_power', 'ram_power', 'cpu_energy',
                'gpu_energy', 'ram_energy', 'energy_consumed']]
            em = em.iloc[0]
            print(em)
            os.remove(os.path.join(results_dir, 'emissions.csv'))

        train_score = 0 
        for tx in tqdm(s):
            r = test_path(tx[0], tx[1], tx[2])
            if r:
                train_score += 1
        train_score = train_score / len(s)
        mean_reward = np.mean(e.get_reward())
        print(f'train score: {train_score}, base score: {base_score}, mean_reward: {mean_reward}, epoch: {epoch}/{epochs}, v: {version}, c: {suffix}{c}')

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
                                                'max_neighbors':max_neighbors(g),
                                                'change' : c,
                                                'suffix' : suffix,
                                                'best_score' : best_score,
                                                'base_score' : base_score, **em}, orient='index').T], ignore_index=True)
        log.to_csv(lf, sep=';', index=False, compression='zip')
