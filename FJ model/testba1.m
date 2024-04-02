clc;
clear;

% 参数设置
n = 1000;     % 节点数量
m0 = 5;       % 初始放置的节点数量
m = 1;        % 每个新增节点连接到现有节点的数量

% 生成无标度网络
A = scalefree(n, m0, m);

% 计算节点度
degree = sum(A);

% 计算节点度分布
degree_counts = hist(degree, 1:max(degree));

% 绘制度分布的对数-对数图
figure;
loglog(1:max(degree), degree_counts, 'o');
title('Degree Distribution of Generated Graph (Log-Log Scale)');
xlabel('Degree (k)');
ylabel('Frequency (P(k))');
