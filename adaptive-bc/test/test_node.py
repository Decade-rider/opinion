from node import Node
import numpy as np
import pytest

def test_init():
    initial_opinion = 0.5
    neighbors = [2, 3]
    C = 0.3
    node_1 = Node(1, initial_opinion, neighbors, C)
    node_2 = Node(2, initial_opinion)

    assert node_1.id == 1
    assert node_1.initial_opinion == initial_opinion
    assert node_1.current_opinion == initial_opinion
    assert node_1.neighbors == neighbors
    assert len(node_1.neighbors) == 2
    assert len(node_2.neighbors) == 0
    assert node_2.neighbors == []

    assert node_1.confidence_bound == C
    assert node_2.confidence_bound == None

@pytest.fixture
def test_node():
    id = 1
    initial_opinion = 0.5
    neighbors = [2, 3]
    return Node(id, initial_opinion, neighbors)

def test_add_neighbor(test_node: Node):
    test_node.add_neighbor(4)
    assert test_node.neighbors == [2, 3, 4]

def test_erase_neighbor(test_node: Node):
    test_node.erase_neighbor(2)
    assert test_node.neighbors == [3]

def test_check_neighbor(test_node: Node):
    assert test_node.check_neighbor(2) == True
    assert test_node.check_neighbor(4) == False

def test_rewire_probability(test_node: Node):
    opinions = [1.0, 1.0, 1.0, 1.0, 1.0]
    opinions[test_node.id] = test_node.current_opinion
    expected_distribution = [0.5, 0, 0, 0, 0.5]
    test_distribution = test_node.rewire_probability(opinions)
    assert np.all([test_distribution[i] == expected_distribution[i] for i in range(5)])

def test_rewire(test_node: Node):
    opinions = [1.0, 1.0, 1.0, 1.0, 1.0]
    opinions[test_node.id] = test_node.current_opinion
    neighbors_before_rewiring = test_node.neighbors.copy()
    RNG = np.random.default_rng()
    new_neighbor = test_node.rewire(opinions, RNG)
    assert new_neighbor not in neighbors_before_rewiring
    assert ((new_neighbor == 0 or new_neighbor == 4))
    assert len(neighbors_before_rewiring) < len(test_node.neighbors)
