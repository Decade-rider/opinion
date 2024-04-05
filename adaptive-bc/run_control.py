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
    model.run(test=False)
    model.save_model(f'adaptive-bc/data/emotion/{filename}.pbz2')

def multiple_experiments(num_experiments, N, C, beta, K, alphas_list):
    for trial in range(1, num_experiments + 1):
        seed = np.random.randint(0, 1000000)
        RNG = np.random.default_rng(seed=seed)
        alphas = RNG.choice(alphas_list, N)
        model_params = kwparams(N, C, beta, trial, K, alphas)
        filename = f'baseline-ABC-K_{K}-C_1-beta_{beta}-trial_{trial}'
        run_single_experiment(seed_sequence=seed, model_params=model_params, filename=filename)
        print(f'Experiment {trial} completed.')

if __name__ == '__main__':
    num_experiments = 10  # 设置要运行的实验次数
    N = 1000
    confidence_intervals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    C = np.random.choice(confidence_intervals, N)
    alphas_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    beta = 1
    K = 5

    multiple_experiments(num_experiments, N, C, beta, K, alphas_list)
