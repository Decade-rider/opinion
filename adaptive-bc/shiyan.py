'''
Author: Kamenrider 1161949421@qq.com
Date: 2024-04-05 22:50:16
LastEditors: Kamenrider 1161949421@qq.com
LastEditTime: 2024-04-18 15:35:52
FilePath: \opinion\adaptive-bc\shiyan.py
Description: 

Copyright (c) 2024 by 1161949421@qq.com, All Rights Reserved. 
'''
# 假设 run.py 和 visualize.py 在同一目录中
import numpy as np
# from run_control import multiple_experiments 
from run_moxingduibi import multiple_experiments
from visualize import load_model,plot_opinion_evolution,plot_emotion_evolution,plot_emotion_extreme_evolution,plot_emotion_extreme_total_evolution

# 设置实验参数
num_experiments = 5  # 设置要运行的实验次数
N = 51
C = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], N) # 一般设置
# C = np.random.choice([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], N)
# C = np.random.choice([0.6, 0.6, 0.6, 0.6, 0.6], N)
# C = np.random.choice([0.5, 0.5, 0.5, 0.5, 0.5], N)
# C = np.random.choice([0.4, 0.4, 0.4, 0.4, 0.4], N)
# C = np.random.choice([0.3, 0.3, 0.3, 0.3, 0.3], N)
# C = np.random.choice([0.2, 0.2, 0.2, 0.2, 0.2], N)
# C = np.random.choice([0.1, 0.1, 0.1, 0.1, 0.1], N)
alphas_list = [0.1, 0.2, 0.3, 0.4, 0.5] # 一般设置
# alphas_list = [0.5, 0.5, 0.5, 0.5, 0.5]
# alphas_list = [0.4, 0.4, 0.4, 0.4, 0.4]
# alphas_list = [0.3, 0.3, 0.3, 0.3, 0.3]
# alphas_list = [0.2, 0.2, 0.2, 0.2, 0.2]
# alphas_list = [0.1, 0.1, 0.1, 0.1, 0.1]
beta = 1
K = 5

# 运行多次实验
multiple_experiments(num_experiments, N, C, beta, K, alphas_list)

# 加载模型并绘制结果
for trial in range(1, num_experiments + 1):
    filename = f'adaptive-bc/data/emotion/snownlp/固定seed，alpha和C都随机/baseline-ABC-K_{K}-C_1-beta_{beta}-trial_{trial}.pbz2'
    model = load_model(filename)
    
    # 绘制每次实验观点的变化
    plot_opinion_evolution(model,f'baseline-ABC-K_{K}-C_1-beta_{beta}-trial_{trial}')
    
    # 绘制每次实验的情绪演变
    plot_emotion_evolution(model, f'emotion_evolution_trial_{trial}')
    
    # 绘制每次实验的两种极端情绪演变
    plot_emotion_extreme_evolution(model, f'emotion_extreme_evolution_trial_{trial}')
    
    # 绘制每次实验的极端情绪演变
    plot_emotion_extreme_total_evolution(model, f'emotion_extreme_total_evolution_trial_{trial}')
    
