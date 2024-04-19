'''
Author: Kamenrider 1161949421@qq.com
Date: 2024-04-02 10:25:06
LastEditors: Kamenrider 1161949421@qq.com
LastEditTime: 2024-04-12 21:29:52
FilePath: \opinion\adaptive-bc\visualize.py
Description: 

Copyright (c) 2024 by 1161949421@qq.com, All Rights Reserved. 
'''
import pickle
import bz2
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import Counter

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

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
    plt.savefig(f'adaptive-bc/data/emotion/snownlp/固定seed，alpha和C都随机/{filename}.png')
    plt.close()

def plot_emotion_evolution(model, filename):
    # 准备数据
    negative_emotions = []
    neutral_emotions = []
    positive_emotions = []

    for emotions in model.emotion_history:
        emotion_count = Counter(emotions)
        negative_emotions.append(emotion_count['负面情绪'])
        neutral_emotions.append(emotion_count['中性情绪'])
        positive_emotions.append(emotion_count['正面情绪'])

    # 绘制情绪随时间变化的折线图
    plt.figure(figsize=(12, 8))
    plt.plot(negative_emotions, label='负面情绪')
    plt.plot(neutral_emotions, label='中性情绪')
    plt.plot(positive_emotions, label='正面情绪')
    plt.title('情绪随时间的变化')
    plt.xlabel('时间步')
    plt.ylabel('数量')
    plt.legend()
    plt.savefig(f'adaptive-bc/data/emotion/snownlp/固定seed，alpha和C都随机/{filename}_emotion_evolution.png')
    plt.close()

def plot_emotion_extreme_evolution(model, filename):
    # 准备数据
    negative_emotions_extreme = []
    positive_emotions_extreme = []
    for emotions in model.extreme_emotion_history:
        emotion_count = Counter(emotions)
        negative_emotions_extreme.append(emotion_count['极端负面情绪'])
        positive_emotions_extreme.append(emotion_count['极端正面情绪'])
    
    # 绘制两种极端情绪随时间变化的折线图
    plt.figure(figsize=(12, 8))
    plt.plot(negative_emotions_extreme, label='极端负面情绪')
    plt.plot(positive_emotions_extreme, label='极端正面情绪')
    plt.title('两种极端情绪随时间的变化')
    plt.xlabel('时间步')
    plt.ylabel('数量')
    plt.legend()
    plt.savefig(f'adaptive-bc/data/emotion/snownlp/固定seed，alpha和C都随机/{filename}_emotion_extreme_evolution.png')
    plt.close()
        
    
def plot_emotion_extreme_total_evolution(model, filename):
    # 准备数据
    emotions_extreme_total = []
    for emotions in model.extreme_emotion_history:
        emotion_count = Counter(emotions)
        emotions_extreme_total.append(emotion_count['极端负面情绪'] + emotion_count['极端正面情绪'])

    # 绘制极端情绪随时间变化的折线图
    plt.figure(figsize=(12, 8))
    plt.plot(emotions_extreme_total, label='极端情绪')
    plt.title('极端情绪随时间的变化')
    plt.xlabel('时间步')
    plt.ylabel('数量')
    plt.legend()
    plt.savefig(f'adaptive-bc/data/emotion/snownlp/固定seed，alpha和C都随机/{filename}_emotion_total_evolution.png')
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


    