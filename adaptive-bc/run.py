'''
Author: Kamenrider 1161949421@qq.com
Date: 2024-04-02 10:25:06
LastEditors: Kamenrider 1161949421@qq.com
LastEditTime: 2024-04-23 15:36:51
FilePath: \opinion\adaptive-bc\run.py
Description: 
整个项目的运行我是在conda环境中创建了一个py3.11的版本，需要的库也就是numpy、matplotlib、pandas、networkx、pickle、pz2和openpyxl.由于环境中我还安装了一个ndlib的库，因此我如果导出成requirment.txt的话，安装的库可能非常多，所以我就不导出了，还有对于python的版本要求最低是py3.8。

该项目的原始文件一共有四个，分别是run.py、model.py、node.py和visualize.py。其中run.py是主文件，用于运行模型；model.py是模型文件，包含了模型的定义和运行；node.py是节点文件，包含了节点的定义；visualize.py是可视化文件，包含了可视化的函数。
整个项目原来只能进行单次实验和使用生成的数据进行可视化，现在增加了多次实验的功能，可以运行多次实验，并且可以读取外部传入的观点和网络结构。
以'run_'开头的文件是用于运行模型的文件，run_control.py相对于原来的运行文件run.py，增加了多次实验的功能，可以运行多次实验；run_moxingduibi.py在run_control.py的基础上增加了读取外部传入观点和网络结构的功能。所以还额外增加了G_real.py用来生成网络结构。

而创建了这么多控制脚本，很繁琐，所以又建了了一个shiyan.py文件，用来调用这些控制脚本，实现一键运行多次实验并保存实验结果和可视化结果。
需要调用哪个文件使用'import'调用即可。不过由于脚本扩展的逐步进行只需要分别根据使用模拟和真实数据来调用run_control.py和run_moxingduibi.py即可。

整个项目的核心是model.py文件，包含了模型的定义和运行，其中的Model类是模型的核心类，包含了模型的定义和运行的方法。
model中需要传入的参数有seed_sequence, N, C, beta, trial, K, alphas。
其中seed_sequence是随机数种子，N是节点数，C是置信度，beta是模型参数，trial是实验次数，K是每次交互更新观点的节点对数，alphas是alpha值。还有一个full_time_series参数，用于控制是否保存完整的时间序列，意思是如果数据量太大的话，会导致内存溢出，因此将其设置位False会每250次作为一个时间步长进行记录。

如果需要每次实验的结果都一样需要固定随机种子，否则每次实验的结果都不一样。

C是置信度，是一个列表，每个节点的置信度是不同的，可以是随机的，也可以是固定的。alphas是alpha值，是一个列表，每个节点的alpha值是不同的，可以是随机的，也可以是固定的。一般C的取值都是[0,1],alpha的取值是[0,0.5]。为什么采用列表的形式赋值，是出于个体差异性的考虑，这样处理可以给每个个体赋予相同或不同的置信阈值和观点接受度，也是我做的一个拓展。

还有beta值得特别说明一下，beta是模型的原作者对DW模型的一个拓展参数，当beta=1时，模型就是DW模型，当beta!=1时，模型就是拓展的模型。所以在实验中，我们可以设置beta=1来进行DW模型的实验，也可以设置beta!=1来进行拓展模型的实验。

visualize.py主是四个结果的可视化，分别是观点演变、情绪演变、两种极端情绪演变和极端情绪总数演变。


Copyright (c) 2024 by 1161949421@qq.com, All Rights Reserved. 
'''
import multiprocessing
from model import Model
from node import Node
import numpy as np
from numpy.random import SeedSequence


# record data for baseline results
def kwparams(N, C, beta, trial, K,alphas):
    params = {
        "trial" : trial,
        "max_steps" : 10000000,
        "N" : N,
        "p" : 0.1,
        "tolerance" : 1e-5,
        # "alpha" : 0.1, # 0.5 is consensus parameter
        "alphas": alphas, # 0.5 is consensus parameter，添加alpha为列表
        "beta" : beta,
        "C" : C,
        "M" : 1,
        "K" : K,
        "full_time_series": False
    }
    return params

def run_model(seed_sequence, model_params, filename=None):
    model = Model(seed_sequence, **model_params)
    model.run(test=True)
    # 获取情绪历史记录
    emotion_history = model.emotion_history
    
    model.save_model(f'adaptive-bc/data/emotion/{filename}.pbz2')

    if model.beta != 1:
        print(f'Network assortativity: {model.start_assortativity}')
        print(f'End assortativity: {model.end_assortativity}')

if __name__ == '__main__':
    seed = 123456789

    N = 100
    RNG = np.random.default_rng(seed=seed)

    confidence_intervals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    C = RNG.choice(confidence_intervals, N)
    # C = 1

    alphas_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    alphas = RNG.choice(alphas_list, N)

    # beta = 0.25
    beta = 1
    trial = 1

    # params,     
    # K_list = {1, 5, 10, 20}
    K=5

    model_params=kwparams(N, C, beta, trial,K,alphas=alphas)
    run_model(seed_sequence=seed, model_params=model_params, filename=f'baseline-ABC-K_5-C_-alpha_-beta_1-trial_1-emotion')

    # print('Running model...')
    # processes = []
    # for K in K_list:
    #     print(f'testing K={K}...')
    #     process = multiprocessing.Process(target=run_model, args=(seed, kwparams(N, C, beta, trial, K), f'baseline-ABC-K_{K}-C_1-beta_1'))
    #     processes.append(process)
    #     process.start()

    # for process in processes:
    #     process.join()
    