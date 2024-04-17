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