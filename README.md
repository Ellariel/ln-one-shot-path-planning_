# ln-one-shot-path-planning


## Sources
Native pathfinding algorithms are based on works of Kumble, Roos & Epema (2021) [](https://github.com/SatwikPrabhu/Attacking-Lightning-s-anonymity)
Dick Epema


https://ieeexplore.ieee.org/document/9566199
https://arxiv.org/pdf/2107.10070.pdf

S. P. Kumble and S. Roos, "Comparative Analysis of Lightning's Routing Clients," 2021 IEEE International Conference on Decentralized Applications and Infrastructures (DAPPS), United Kingdom, 2021, pp. 79-84, doi: 10.1109/DAPPS52256.2021.00014.


algorithms This includes a simulator to simulate transactions using LND(https://github.com/lightningnetwork/lnd/blob/master/routing/pathfind.go), c-Lightning(https://github.com/ElementsProject/lightning/blob/f3159ec4acd1013427c292038b88071b868ab1ff/common/route.c) and Eclair(https://github.com/ACINQ/eclair/blob/master/eclair-core/src/main/scala/fr/acinq/eclair/router/Router.scala).

The experiment is run on a snapshot of the Lightning Network obtained from https://ln.bigsun.xyz. The set of adversaries is a mixture of nodes with high centrality, low centrality and random nodes. The snapshot as well as the centralities of all nodes are found in data/Snapshot and data/Centrality respectively.
