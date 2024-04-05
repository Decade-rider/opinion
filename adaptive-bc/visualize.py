'''
Author: Kamenrider 1161949421@qq.com
Date: 2024-04-02 10:25:06
LastEditors: Kamenrider 1161949421@qq.com
LastEditTime: 2024-04-05 18:20:07
FilePath: \opinion\adaptive-bc\visualize.py
Description: 

Copyright (c) 2024 by 1161949421@qq.com, All Rights Reserved. 
'''
import pickle
import bz2
import matplotlib.pyplot as plt

def load_model(filename):
    with bz2.BZ2File(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def plot_opinion_evolution(model, filename):
    plt.figure(figsize=(12, 8))
    plt.plot(model.X_data)
    plt.xlabel('Time')
    plt.ylabel('Opinion')
    plt.title('Opinion Evolution: Adaptive-BC')
    plt.savefig(f'adaptive-bc/data/{filename}.eps')
    plt.close()

def plot_emotion_evolution(model, filename):
    for step, emotions in enumerate(model.emotion_history):
        plt.figure()
        plt.bar(emotions.keys(), emotions.values())
        plt.xlabel('情绪类别')
        plt.ylabel('数量')
        plt.title(f'情绪分布 - 时间步 {step}')
        plt.savefig(f'adaptive-bc/data/emotion/{filename}_step_{step}.png')
        plt.close()

if __name__ == '__main__':
    file = 'baseline-ABC-K_5-C_-alpha_-beta_1-trial_1-emotion'
    model_file = f'adaptive-bc/data/emotion/{file}.pbz2'
    
    model = load_model(model_file)

    model.info()
    print(f'Convergence time: {model.convergence_time}')
    print('Opinion evolution: \n', model.X_data)

    if model.beta != 1:
        print(f'Rewiring threshold: {model.beta}')
        print('Edge changes: \n', model.edge_changes)
    else:
        print('No rewiring, beta is 1')

    plot_opinion_evolution(model, file)
    plot_emotion_evolution(model, file)

    