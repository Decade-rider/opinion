clc;
clear;

% 参数设置
n = 100;       % 节点数量
m0 = 5;        % 初始放置的节点数量
m = 1;         % 每个新增节点连接到现有节点的数量

% 生成无标度网络
A = scalefree(n, m0, m);

% 计算节点度分布
degree_distribution = full(sum(A, 2));

% 绘制节点度分布直方图
figure;
histogram(degree_distribution, 'Normalization', 'probability');
title('Degree Distribution of Generated Graph');
xlabel('Degree');
ylabel('Probability');

% 绘制生成的图
figure;
spy(A);  % 绘制稀疏矩阵的图形
title('Generated Graph');

% 打印生成的图的基本信息
fprintf('Number of nodes: %d\n', n);
fprintf('Number of edges: %d\n', nnz(A)/2); % 因为邻接矩阵是对称的，所以边的数量是非零元素数的一半

% 假设 degree_distribution 是节点度分布的数据
figure;
histogram(degree_distribution, 'Normalization', 'probability');
title('Degree Distribution of Generated Graph');
xlabel('Degree');
ylabel('Probability');
