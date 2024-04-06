# 假设 run.py 和 visualize.py 在同一目录中
import numpy as np
from run_control import multiple_experiments 
from visualize import load_model,plot_opinion_evolution,plot_emotion_evolution

# 设置实验参数
num_experiments = 10
N = 500
C = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], N)
alphas_list = [0.1, 0.2, 0.3, 0.4, 0.5]
beta = 1
K = 5

# 运行多次实验
multiple_experiments(num_experiments, N, C, beta, K, alphas_list)

# 加载模型并绘制结果
for trial in range(1, num_experiments + 1):
    filename = f'adaptive-bc/data/emotion/baseline-ABC-K_{K}-C_1-beta_{beta}-trial_{trial}.pbz2'
    model = load_model(filename)
    
    # 绘制每次实验观点的变化
    plot_opinion_evolution(model,f'baseline-ABC-K_{K}-C_1-beta_{beta}-trial_{trial}')
    
    # 绘制每次实验的情绪演变
    plot_emotion_evolution(model, f'emotion_evolution_trial_{trial}')
