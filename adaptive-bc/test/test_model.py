import networkx as nx
from node import Node
from model import Model
import numpy as np
import random
import pytest
from copy import deepcopy

import pickle
import bz2

# fix test seed
seed = 123456789

@pytest.fixture
def model_params():
    confidence_interval = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    N = 15
    C = np.random.default_rng(seed=seed).choice(confidence_interval, N)
    params = {
        "trial" : 1,
        "max_steps" : 10000,
        "N" : N,
        "p" : 0.1,
        "tolerance" : 1e-5,
        "alpha" : 0.1,
        "C" : C,
        "beta" : 0.25,
        "M" : 1,
        "K" : 1,
        "full_time_series": True
    }
    return params

@pytest.fixture
def seed_sequence():
    return seed

def __equivalent_nodes(node_1: Node, node_2: Node):
    node_attributes = ['id', 'initial_opinion', 'current_opinion', 'neighbors']
    # return True if node_1, node_2 attributes are equivalent
    return np.all([ getattr(node_1, attr) == getattr(node_2, attr) for attr in node_attributes])

def test_model_params(seed_sequence: int, model_params: dict):
    model = Model(seed_sequence, **model_params)
    for k, v in model_params.items():
        assert np.all(getattr(model, k) == v)

def test_initialize_system(seed_sequence: int, model_params: dict):
    RNG = np.random.default_rng(seed_sequence)
    X = RNG.random(model_params['N'])
    G = nx.fast_gnp_random_graph(n=model_params['N'], p=model_params['p'], seed=seed_sequence, directed=False)

    nodes = []
    for i in range(model_params['N']):
        node_neighbors = list(G[i])
        node = Node(id=i, initial_opinion=X[i], neighbors=node_neighbors)
        nodes.append(node)

    # initialize model instance
    model = Model(seed_sequence=seed_sequence, **model_params)

    # test edges
    edges = [(u, v) for u, v in G.edges()]
    assert edges == model.edges

    # test nodes
    for i in range(len(nodes)):
        assert __equivalent_nodes(nodes[i], model.nodes[i])

    assert np.all(model.C == model_params['C'])

def test_rewire(seed_sequence: int, model_params: dict):
    model = Model(seed_sequence, **model_params)
    original_edges = model.edges.copy()

    num_discordant_start = len([(i, j) for i, j in model.edges if abs(model.X[i] - model.X[j]) >= model.beta])
    model.run(test=True)
    num_discordant_end = len([(i, j) for i, j in model.edges if abs(model.X[i] - model.X[j]) >= model.beta])

    print(f'model edges {model.edges}')
    print(f'model edge changes {model.edge_changes}')

    assert model.edge_changes != []
    assert model.edges != original_edges
    assert num_discordant_start > num_discordant_end

def test_dw_step(seed_sequence: int, model_params: dict):
    model = Model(seed_sequence=seed_sequence, **model_params)
    model.run(test=True)
    assert model.stationary_flag == 1
    CT = model.convergence_time

    assert np.sum(np.abs(model.X_data[CT - 50,:] - model.X_data[CT - 51,:])) < model.tolerance
    assert np.sum(np.abs(model.X_data[CT - 2,:] - model.X_data[CT - 1,:])) < model.tolerance
    assert not np.array_equal(model.X_data[0, :], model.X_data[CT - 1, :])

    print(f'initial opinions: {model.X_data[0, :]}')
    print(f'final opinions: {model.X_data[CT - 1, :]}')

# assuming beta < 1 for this test, if its 1, then the test will fail
def test_get_network(seed_sequence: int, model_params: dict):
    model = Model(seed_sequence, **model_params)
    model.run(test=True)
    assert list(model.get_network(time=0).edges) == model.initial_edges

    # order is wrong, but edges are the same
    # assert list(model.get_network(time=1).edges) == list(nx.Graph([(2, 6), (3, 4), (3, 5), (8, 14), (10, 11), (13, 14)]).edges)

def test_get_opinions(seed_sequence: int, model_params: dict):
    model = Model(seed_sequence, **model_params)
    model.run(test=True)
    CT = model.convergence_time
    assert np.all(model.get_opinions(time=0) == model.X_data[0, :])
    assert np.all(model.get_opinions(time=1)[-1] == model.X_data[1, :])
    assert np.all(model.get_opinions(time=CT-1)[-1] == model.X_data[CT-1, :])


def test_compressed_pickle(seed_sequence: int, model_params: dict):
    RNG = np.random.default_rng(seed_sequence)
    X = RNG.random(model_params['N'])
    G = nx.fast_gnp_random_graph(n=model_params['N'], p=model_params['p'], seed=seed_sequence, directed=False)

    nodes = []
    for i in range(model_params['N']):
        node_neighbors = list(G[i])
        node = Node(id=i, initial_opinion=X[i], neighbors=node_neighbors)
        nodes.append(node)

    # initialize model instance
    model = Model(seed_sequence=seed_sequence, **model_params)

    # pickle object
    with bz2.BZ2File('test/test_compressed_pickle.pbz2', 'w') as f:
        pickle.dump(model, f)

    # load pickle object
    loaded_model = bz2.BZ2File('test/test_compressed_pickle.pbz2', 'rb')
    loaded_model = pickle.load(loaded_model)

    # model info
    loaded_model.info()

    # test edges
    edges = [(u, v) for u, v in G.edges()]
    assert edges == loaded_model.edges

    # test nodes
    for i in range(len(nodes)):
        assert __equivalent_nodes(nodes[i], loaded_model.nodes[i])

def test_save_model(seed_sequence: int, model_params: dict):
    model = Model(seed_sequence, **model_params)
    model.run(test=True)
    filename = 'test/test_save_model.pbz2'
    model.save_model(filename)

    loaded_model = bz2.BZ2File(filename, 'rb')
    loaded_model = pickle.load(loaded_model)

    X_data = model.X_data.copy()
    X_data[~np.all(X_data == 0, axis=1)]

    assert (X_data == loaded_model.X_data).all()
    assert model.edges == loaded_model.edges

    for i in range(len(model.nodes)):
        assert __equivalent_nodes(model.nodes[i], loaded_model.nodes[i])
