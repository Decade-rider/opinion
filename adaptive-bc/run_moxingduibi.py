import numpy as np
from numpy.random import SeedSequence
from model import Model
import multiprocessing
import pandas as pd
import pickle
import networkx as nx

def load_network_and_opinions():
    # 加载网络结构
    with open('adaptive-bc/Graph/network_structure.pickle', 'rb') as f:
        G = pickle.load(f)
    
    # 加载初始观点数据
    opinions_df = pd.read_excel('gdtj_qdr.xlsx', sheet_name='Sheet2', usecols=[0, 1], header=None)
    opinions_df.columns = ['NodeID', 'Opinion']
    
    # 创建一个字典，将个体编号与观点对应起来
    opinions = dict(zip(opinions_df['NodeID'], opinions_df['Opinion']))
    
    return G, opinions

def kwparams(N, C, beta, trial, K, alphas):
    return {
        "trial": trial,
        "max_steps": 1000000,
        "N": N,
        "p": 0.1,
        "tolerance": 1e-5,
        "alphas": alphas,
        "beta": beta,
        "C": C,
        "M": 1,
        "K": K,
        "full_time_series": False
    }

def run_single_experiment(seed_sequence, model_params, filename, G, opinions):
    model = Model(seed_sequence, G, opinions, **model_params)
    model.run(test=False)
    model.save_model(filename=f'adaptive-bc/data/emotion/moxingduibi/{filename}.pbz2')

def multiple_experiments(num_experiments, N, C, beta, K, alphas_list):
    G, opinions = load_network_and_opinions()
    for trial in range(1, num_experiments + 1):
        seed = 123456789
        RNG = np.random.default_rng(seed=seed)
        alphas = RNG.choice(alphas_list, N)
        model_params = kwparams(N, C, beta, trial, K, alphas)
        filename = f'baseline-ABC-K_{K}-C_1-beta_{beta}-trial_{trial}'
        run_single_experiment(SeedSequence(seed), model_params, filename, G, opinions)
        print(f'Experiment {trial} completed.')

if __name__ == "__main__":
    num_experiments = 10
    N = 100
    C = 0.5
    beta = 0.3
    K = 2
    alphas_list = np.linspace(0.01, 0.1, 10)
    multiple_experiments(num_experiments, N, C, beta, K, alphas_list)
