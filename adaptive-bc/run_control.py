'''
Author: Kamenrider 1161949421@qq.com
Date: 2024-04-05 20:21:14
LastEditors: Kamenrider 1161949421@qq.com
LastEditTime: 2024-04-23 14:36:39
FilePath: \opinion\adaptive-bc\run_control.py
Description: 

Copyright (c) 2024 by 1161949421@qq.com, All Rights Reserved. 
'''
import multiprocessing
from model import Model
import numpy as np
from numpy.random import SeedSequence
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

def run_single_experiment(seed_sequence, model_params, filename):
    model = Model(seed_sequence, **model_params)
    model.run(test=False)  # 运行模型
    model.save_model(filename=f'adaptive-bc/data/emotion/{filename}.pbz2')




def multiple_experiments(num_experiments, N, C, beta, K, alphas_list):
    for trial in range(1, num_experiments + 1):
        # seed = np.random.randint(0, 1000000) # 生成随机种子, 用于生成随机数,保证实验的随机性。如果要复现实验结果，可以将seed设置为固定值，如seed=123456789
        seed = 123456789
        RNG = np.random.default_rng(seed=seed)
        alphas = RNG.choice(alphas_list, N)
        model_params = kwparams(N, C, beta, trial, K, alphas)
        filename = f'baseline-ABC-K_{K}-C_1-beta_{beta}-trial_{trial}'
        run_single_experiment(seed_sequence=seed, model_params=model_params, filename=filename)
        print(f'Experiment {trial} completed.')


if __name__ == '__main__':
    
    '''
    设置实验所需的参数。
    如果新建一个文件，可以将这部分代码复制到新文件中，并修改参数。而不用复制if __name__ == '__main__'，
    不过记得使用from run_control import multiple_experiments 从而使用该方法来进行多次实验。
    '''
    num_experiments = 10  # 设置要运行的实验次数
    N = 100
    confidence_intervals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    C = np.random.choice(confidence_intervals, N)
    alphas_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    beta = 1
    K = 5

    multiple_experiments(num_experiments, N, C, beta, K, alphas_list) # 调用函数运行实验
