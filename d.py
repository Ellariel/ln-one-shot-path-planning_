from base import Routing
import os, pickle

base_dir = '.\snapshots'

with open(os.path.join(base_dir, f'transactions.pickle'), 'rb') as f:
    G = pickle.load(f)
T = G['transations']
G = G['undirected_graph']
print(f'undirected graph, n: {len(G.nodes)} e: {len(G.edges)}')
print(f'transations count, t: {len(T)}')

u, v, amount = T[0]
print(f'u: {u}, v: {v}, amount:{amount}')
d = Routing()
d.Dijkstra(G, u, v, amount)
