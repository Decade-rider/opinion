clc,clear;
% 参数设置
n = 100; % 节点数量
m0 = 5; % 初始放置的节点数量
m = 3; % 每个新增节点连接到现有节点的数量
tMax = 100; % 最大模拟轮数
c_eps = 1e-6; % 收敛限制

% 生成无标度网络
A = scalefree(n, m0, m);

% 生成观点向量，假设服从均匀分布
lower_bound = 0; % 均匀分布的下界
upper_bound = 1; % 均匀分布的上界
s = unifrnd(lower_bound, upper_bound, n, 1);

% 运行 Friedkin-Johnsen 模型
[equilibrium, opinions] = friedkinJohnsen(A, s, tMax, c_eps, 'plot');
