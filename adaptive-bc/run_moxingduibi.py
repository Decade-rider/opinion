'''
Author: Kamenrider 1161949421@qq.com
Date: 2024-04-18 00:10:09
LastEditors: Kamenrider 1161949421@qq.com
LastEditTime: 2024-04-23 15:53:08
FilePath: \opinion\adaptive-bc\run_moxingduibi.py
Description: 
用到的观点数据就是师兄的程序中用到的的gdtj的数据，sheet1是ztpd.mat生成的，sheet2则是个体的编号、初始观点以及各自的情绪值。
采用了将初始情绪值作为个体的观点输入进行仿真，原因是二元观点的数据在模型可视化的结果出来很难分析，体现不了多次交互。
Copyright (c) 2024 by 1161949421@qq.com, All Rights Reserved. 
'''
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
    opinions_df = pd.read_excel('adaptive-bc/gdtj_qdr.xlsx', sheet_name='Sheet2', usecols=[0, 2], header=None)
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
        "full_time_series": True
    }

def run_single_experiment(seed, model_params, filename, G, opinions):
    model = Model(G, opinions, seed, **model_params)  # 传递整数种子
    model.run(test=False)
    model.save_model(filename=f'adaptive-bc/data/emotion/snownlp/固定seed，alpha和C都随机/{filename}.pbz2')


def multiple_experiments(num_experiments, N, C, beta, K, alphas_list):
    G, opinions = load_network_and_opinions()  # 加载网络和观点数据
    for trial in range(1, num_experiments + 1):
        seed = 123456789  # 固定种子或使用 np.random.randint(0, 1000000) 生成随机种子
        # seed = np.random.randint(0, 1000000)
        RNG = np.random.default_rng(seed=seed)  # 创建随机数生成器
        alphas = RNG.choice(alphas_list, N)  # 从范围中选择 alpha 值
        model_params = kwparams(N, C, beta, trial, K, alphas)
        filename = f'baseline-ABC-K_{K}-C_1-beta_{beta}-trial_{trial}'
        run_single_experiment(seed, model_params, filename, G, opinions)  # 直接传递整数种子
        print(f'Experiment {trial} completed.')

if __name__ == "__main__":
    num_experiments = 10
    N = 100
    C = 0.5
    beta = 0.3
    K = 2
    alphas_list = np.linspace(0.01, 0.1, 10)
    multiple_experiments(num_experiments, N, C, beta, K, alphas_list)
