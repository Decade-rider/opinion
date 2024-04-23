'''
Author: Kamenrider 1161949421@qq.com
Date: 2024-04-17 21:52:47
LastEditors: Kamenrider 1161949421@qq.com
LastEditTime: 2024-04-23 16:04:12
FilePath: \opinion\adaptive-bc\G_real.py
Description: 
这个程序使用的数据就是师兄的程序中用到的的qx的数据，sheet1是ztpd.mat生成的用于生成网络结构，具体地说就是根据每次交互来创建，两个个存在一次
交互即建立联系，为了体现交流是双向的建立的是无向图。
Copyright (c) 2024 by 1161949421@qq.com, All Rights Reserved. 
'''
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# 加载数据，假设 Excel 文件中现在包含列名
df = pd.read_excel('adaptive-bc\gdtj_qdr.xlsx')

# 创建无向图
G = nx.Graph()

# 添加边和节点属性
for idx, row in df.iterrows():
    # 使用新的列名
    G.add_edge(row['听的个体'], row['说的个体'])
    # 为两个节点分别设置观点属性
    G.nodes[row['听的个体']]['听的观点'] = row['听的观点']
    G.nodes[row['听的个体']]['说的观点'] = row['说的观点']
    G.nodes[row['听的个体']]['子群'] = row['子群']
    G.nodes[row['说的个体']]['听的观点'] = row['听的观点']
    G.nodes[row['说的个体']]['说的观点'] = row['说的观点']
    G.nodes[row['说的个体']]['子群'] = row['子群']

# # 输出网络基本信息
# print(nx.info(G))

# 可视化网络
nx.draw(G, with_labels=True, node_color='lightblue')
plt.show()

# 保存生成的网络结构以便导入到model.py中
with open(r'adaptive-bc\Graph\network_structure.pickle', 'wb') as f:
    pickle.dump(G, f)