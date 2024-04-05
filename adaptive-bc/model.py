import networkx as nx
from node import Node
import numpy as np
# from numpy.random import RandomState, MT19937
import random

import pickle
import bz2

class Model:
    def __init__(self, seed_sequence, **kwparams) -> None:
        # # Print the length and first few elements of C to debug
        # print(f'Length of C: {len(kwparams["C"])}')
        # print(f'First few elements of C: {kwparams["C"][:5]}')
        # set random state for model instance to ensure repeatability
        self.seed_sequence = seed_sequence
        try:
            self.spawn_key = seed_sequence.spawn_key[0]
        except:
            self.spawn_key = None

        # each instance gets its own RNG
        self.RNG = np.random.default_rng(seed_sequence)
        # set state of random module
        random.seed(seed_sequence)
        # self.random_state = seed_sequence # RandomState(MT19937(seed_sequence)) or random_state???

        # set model params
        self.trial = kwparams['trial']                      # trial ID (for saving model)
        self.max_steps = kwparams['max_steps']              # bailout time
        self.N = kwparams['N']                              # number of nodes
        self.p = kwparams['p']                              # p in G(N, p), probability of edge creation
        self.tolerance = kwparams['tolerance']              # convergence tolerance
        # self.alpha = kwparams['alpha']                      # convergence parameter 每个个体的收敛参数，这里每个个体的收敛参数都是相同的，所以这里直接用一个值即可
        self.alphas = kwparams.get('alphas', [0.1] * self.N) # New: Initialize alphas list for individual node's alpha values
        self.C = kwparams['C']                              # confidence bound
        self.beta = kwparams['beta']                        # rewiring threshold
        self.M = kwparams['M']                              # num of edges to rewire each step
        self.K = kwparams['K']                              # num of node pairs to update opinions at each step
        self.full_time_series = kwparams['full_time_series']       # save time series opinion data
        self.emotion_history = []  # 用于存储每个时间步骤的情绪分布


        # generate network and set attributes:
        # opinions, initial_opinions, initial_edges, nodes, edges
        self.__initialize_network()

        # X = opinion data
        if self.full_time_series:
            self.X_data = np.ndarray((self.max_steps, self.N))   # storing time series opinion data
            self.X_data[0, :] = self.X                           # record initial opinions
            self.edge_changes = []                                  # record edge changes
        else:
            self.X_data = np.ndarray((int(self.max_steps / 250) + 1, self.N))     # store opinions every 250 time steps
            self.X_data[0, :] = self.X                           # record initial opinions
            self.G_snapshots = []                           # record network snapshots every 250 time steps

        self.num_discordant_edges = np.empty(self.max_steps)# track number of discordant edges
        self.stationary_counter = 0                         # determining if we reached a stationary state
        self.stationary_flag = 0                            # flagging stationary state
        self.convergence_time = None                        # record convergence time

        # if beta == 1, no rewiring, standard BC applies
        self.rewiring = False if int(kwparams['beta'] == 1) else True

        # before running model, calculate network assortativity
        self.start_assortativity = nx.degree_assortativity_coefficient(nx.Graph(self.edges))

    def __initialize_network(self) -> None:

        print('initializing network')
        # random initial opinions from [0, 1] uniformly
        # opinions = self.RNG.random(self.N)
        # Generate initial opinions from a normal distribution
        # opinions = np.random.normal(loc=0.5, scale=0.1, size=self.N)
        # 0<=opinions<1/3,为负面情绪。1/3<=opinions<2/3,为中性情绪。2/3<=opinions<=1,为正面情绪
        opinions = np.random.uniform(0,1,self.N)        
        # generate G(N, p) random graph
        # G = nx.fast_gnp_random_graph(n=self.N, p=self.p, seed=self.seed_sequence, directed=False)
        # Generate a barabasi_albert scale-free network
        G = nx.barabasi_albert_graph(self.N,2,seed=self.seed_sequence)

        # random confidence bounds for each agent if providing list # 这部分可以不要
        # if type(self.C) is not list:
        #     self.C = [self.C] * self.N
        
        # Print the length and first few elements of self.C to ensure it's correctly set
        # print(f'Length of self.C after processing: {len(self.C)}')
        # print(f'First few elements of self.C: {self.C[:5]}')

        nodes = []
        for i in range(self.N):
            node_neighbors = list(G[i])
            node = Node(id=i, initial_opinion=opinions[i], neighbors=node_neighbors, confidence_bound=self.C[i],alpha=self.alphas[i])
            nodes.append(node)
            
        edges = [(u, v) for u, v in G.edges()]

        self.X = opinions
        self.initial_X = opinions
        self.edges = edges.copy()
        self.initial_edges = edges.copy()
        self.nodes = nodes

    def opinion_to_emotion(self, opinion):
        # 情绪分类逻辑
        if 0 <= opinion < 1/3:
            return '负面情绪'
        elif 1/3 <= opinion < 2/3:
            return '中性情绪'
        else:
            return '正面情绪'
    # run the model
    def run(self, test=False) -> None:
        time = 0

        def rewire():
            # get discordant edges
            discordant_edges = [(i, j) for i, j in self.edges if abs(self.X[i] - self.X[j]) > self.beta]

            # if test and discordant_edges:
            #     print(f'discordant edges: {discordant_edges}')

            self.num_discordant_edges[time] = len(discordant_edges)

            # if len of discordant edges >= M, choose M at random using self.RNG
            # else choose all discordant edges to rewire
            if len(discordant_edges) > self.M:
                index = self.RNG.choice(a=len(discordant_edges), size=self.M, replace=False)
                edges_to_cut = [discordant_edges[i] for i in index]
            else:
                edges_to_cut = discordant_edges

            # cut and connect new edges
            for edge in edges_to_cut:
                self.edges.remove(edge)
                i, j = edge[0], edge[1]
                self.nodes[i].erase_neighbor(j)
                self.nodes[j].erase_neighbor(i)

                # pick either i or j to rewire
                random_node = self.RNG.integers(2)
                i = i if random_node == 0 else j
                selected_node = self.nodes[i]
                new_neighbor = selected_node.rewire(self.X, self.RNG)
                self.nodes[new_neighbor].add_neighbor(i)
                new_edge = (i, new_neighbor)

                # record data
                self.edges.append(new_edge)

                if self.full_time_series:
                    self.edge_changes.append((time, edge, new_edge))

            if self.full_time_series == False and time % 250 == 0:
                G = nx.Graph()
                G.add_nodes_from(range(self.N))
                G.add_edges_from(self.edges)
                self.G_snapshots.append((time, G))

        # update opinions using deffuant-weisbuch
        def dw_step():
            index = self.RNG.integers(low=0, high=len(self.edges), size=self.K)
            node_pairs = [self.edges[i] for i in index]

            # for each pair, update opinions in Model and Node
            X_new = self.X.copy()
            for u, w in node_pairs:
                # # Check the value of self.C[u]
                # print(f"Value of self.C[{u}]: {self.C[u]}")
                # print(f"Value of self.C[{w}]: {self.C[w]}")
                
                # assumptions here
                # using confidence bound of the receiving agent
                if abs(self.X[u] - self.X[w]) <= self.C[u]:
                    X_new[u] = self.X[u] + self.nodes[u].alpha * (self.X[w] - self.X[u])
                    self.nodes[u].update_opinion(X_new[u])

                # check other agent is withing their own bounds
                if abs(self.X[w] - self.X[u]) <= self.C[w]:
                    X_new[w] = self.X[w] + self.nodes[w].alpha * (self.X[u] - self.X[w])
                    self.nodes[w].update_opinion(X_new[w])

            # update data
            self.X_prev = self.X.copy()
            self.X = X_new

            if self.full_time_series:
                self.X_data[time + 1, :] = X_new
            elif (time % 250 == 0):
                t_prime = int(time / 250)
                self.X_data[t_prime + 1] = X_new
        
            # 计算当前的情绪分布
            current_emotions = [self.opinion_to_emotion(node.opinion) for node in self.nodes]
            self.emotion_history.append(Counter(current_emotions))

        def check_convergence():
            state_change = np.sum(np.abs(self.X - self.X_prev))
            self.stationary_counter = self.stationary_counter + 1 if state_change < self.tolerance else 0
            self.stationary_flag = 1 if self.stationary_counter >= 100 else 0

        # run model
        while time < self.max_steps - 1 and self.stationary_flag != 1:
            if self.rewiring: rewire()
            dw_step()
            check_convergence()
            time += 1

        print(f'Model finished. \nConvergence time: {time}')

        self.convergence_time = time

        # calculate assortativy after running model
        self.end_assortativity = nx.degree_assortativity_coefficient(nx.Graph(self.edges))

        if not test: self.save_model()

    def get_edges(self, time: int = None) -> list:
        if time == None or time >= self.convergence_time or self.full_time_series == False:
            return self.edges.copy()
        elif time == 0:
            return self.initial_edges.copy()
        else:
            edges = self.initial_edges.copy()
            # find all edge changes up until highest t, where t < time
            edge_changes = [(t, e1, e2) for (t, e1, e2) in self.edge_changes if t < time]
            # iteratively make changes to network
            for (_, old_edge, new_edge) in edge_changes:
                edges.remove(old_edge)
                edges.append(new_edge)

            return edges


    def get_network(self, time: int = None) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.N))

        edges = self.get_edges(time)
        G.add_edges_from(edges)
        return G

    def get_opinions(self, time: int = None):
        if time == None or time >= self.convergence_time:
            return self.X_data.copy()
        else:
            return self.X_data[:time + 1, :]

    def save_model(self, filename=None):
        if self.full_time_series:
            self.X_data = self.X_data[:self.convergence_time, :]
        else:
            self.X_data = self.X_data[:int(self.convergence_time / 250) + 1, :]

        self.num_discordant_edges = self.num_discordant_edges[:self.convergence_time - 1]
        self.num_discordant_edges = np.trim_zeros(self.num_discordant_edges)

        if not filename:
            # C = f'{self.C:.2f}'.replace('.','')
            beta = f'{self.beta:.2f}'.replace('.','')
            filename = f'data/adaptive-bc-beta_{beta}_trial_{self.trial}_spk_{self.spawn_key}.pbz2'

        print(f'saving model to {filename}')
        with bz2.BZ2File(filename, 'w') as f:
            pickle.dump(self, f)

    def info(self):
        print(f'Seed sequeunce: {self.seed_sequence}')
        print(f'Trial number: {self.trial}')
        print(f'Bailout time: {self.max_steps}')
        print(f'Number of nodes: {self.N}')
        print(f'Edge creation probability: {self.p}')
        print(f'Convergence tolerance: {self.tolerance}')
        # print(f'Convergence parameter: {self.alphas}')
        # print(f'Confidence bounds: {self.C}')
        print(f'Rewiring threshold: {self.beta}')
        print(f'Edges to rewire at each time step, M: {self.M}')
        print(f'Node pairs to update opinions, K: {self.K}')
        print(f'Save opinion time series: {self.full_time_series}')
