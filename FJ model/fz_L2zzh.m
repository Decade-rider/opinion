clc,clear;
% 参数设置
n = 100; % 节点数量
m0 = 5; % 初始放置的节点数量
m = 3; % 每个新增节点连接到现有节点的数量
tMax = 100; % 最大模拟轮数
c_eps = 1e-6; % 收敛限制
lambda = 0.001; % 正则化参数

% 生成无标度网络
A = scalefree(n, m0, m);

% 生成观点向量，假设服从均值为0.5，标准差为0.1的正态分布
mu = 1;
sigma = 0.5;
s = normrnd(mu, sigma, n, 1);

% 正则化观点向量
s_norm = norm(s);
s_regularized = s / (s_norm / sqrt(n)) * sqrt(n - lambda); % L2 正则化

% 运行 Friedkin-Johnsen 模型
[equilibrium, opinions] = friedkinJohnsen(A, s_regularized, tMax, c_eps, 'plot');
