'''
Author: Kamenrider 1161949421@qq.com
Date: 2024-04-02 10:25:06
LastEditors: Kamenrider 1161949421@qq.com
LastEditTime: 2024-04-03 17:49:40
FilePath: \opinion\adaptive-bc\node.py
Description: 
这个文件设置了节点的属性，由于我们现在使用的是经典的DW模型，beta=1，不用考虑每次交互后的节点重连概率，这个文件暂时不用管
Copyright (c) 2024 by 1161949421@qq.com, All Rights Reserved. 
'''
import numpy as np

class Node:
    def __init__(self, id: int, initial_opinion: float,alpha: float = None, neighbors: list = None, confidence_bound: float = None) -> None:
        self.id = id
        self.alpha = alpha
        self.initial_opinion = initial_opinion
        self.neighbors = neighbors if neighbors is not None else []
        self.confidence_bound = confidence_bound

        self.current_opinion = initial_opinion
        self.total_opinion_change = 0

    def add_neighbor(self, id: int) -> None:
        self.neighbors.append(id)

    def erase_neighbor(self, id: int) -> None:
        self.neighbors.remove(id)

    def check_neighbor(self, id: int) -> bool:
        return id in self.neighbors

    def update_opinion(self, new_opinion: float) -> None:
        self.total_opinion_change += abs(self.current_opinion - new_opinion)
        self.current_opinion = new_opinion

    def rewire(self, opinions, RNG):
        rewire_prob = self.rewire_probability(opinions)
        new_neighbor = RNG.choice(range(len(rewire_prob)), p=rewire_prob)
        self.add_neighbor(new_neighbor)
        return new_neighbor

    def rewire_probability(self, X):
        # compute distance of opinions using L2 metric
        distances = np.array([(self.current_opinion - x) ** 2 for x in X])

        # compute rewire probability distribution
        rewiring_distribution = 1 - distances
        rewiring_distribution[self.id] = 0          # set probability of current id to 0 to prevent self edges
        rewiring_distribution[self.neighbors] = 0   # set probability of current neighbors to 0 to prevent duplicate edges

        # normalize distribution
        A = np.sum(rewiring_distribution)
        rewiring_distribution = (1 / A) * rewiring_distribution

        return rewiring_distribution

    def info(self) -> None:
        print(f'id: {self.id}')
        print(f'alpha: {self.alpha}')
        print(f'init opinion: {self.initial_opinion}')
        print(f'current opinion {self.current_opinion}')
        print(f'neighbors {self.neighbors}')
