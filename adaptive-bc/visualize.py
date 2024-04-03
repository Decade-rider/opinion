'''
Author: Kamenrider 1161949421@qq.com
Date: 2024-04-02 10:25:06
LastEditors: Kamenrider 1161949421@qq.com
LastEditTime: 2024-04-03 21:17:08
FilePath: \opinion\adaptive-bc\visualize.py
Description: 

Copyright (c) 2024 by 1161949421@qq.com, All Rights Reserved. 
'''
import numpy as np
import pickle
import bz2

import matplotlib.pyplot as plt

import networkx as nx

if __name__ == '__main__':

    # load the model
    file = 'baseline-ABC-K_5-C_1-beta_1'

    loaded_model = bz2.BZ2File(f'adaptive-bc/data/{file}.pbz2', 'rb')
    loaded_model = pickle.load(loaded_model)

    loaded_model.info()

    # convergence time
    print(f'Convergence time: {loaded_model.convergence_time}')

    # see opinion history
    print(f'Opinion evolution: \n{loaded_model.X_data}')

    # see all edge changes, if beta is 1, no wiring since all opinions are within threshold
    if loaded_model.beta == 1:
        print('No rewiring, beta is 1')
    else:
        print(f'Rewiring threshold: {loaded_model.beta}')
        print(f'Edge changes: \n{loaded_model.edge_changes}')

    

    # Plot opinion evolution
    plt.figure(figsize=(12, 8))
    plt.plot(loaded_model.X_data)   # convergence time is too 
    plt.xlabel('Time')
    plt.ylabel('Opinion')
    # plt.legend(['Nodes in Network'], bbox_to_anchor=(1.3, 1), loc='upper right')
    # plt.annotate(f'$confidence$ $\epsilon = {loaded_model.C}$', xy=(1.05,.8), xycoords='axes fraction', fontsize=12)
    plt.title(f'Opinion Evolution: Adaptive-BC')
    plt.show()
    plt.savefig(f'adaptive-bc/data/{file}.png')