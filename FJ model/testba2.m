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

% 将节点度分布保存到文本文件中
fileID = fopen('degree_distribution.txt', 'w');
fprintf(fileID, 'Degree Frequency\n');
for i = 1:length(degree_counts)
    fprintf(fileID, '%d %d\n', i, degree_counts(i));
end
fclose(fileID);

disp('Degree distribution saved to "degree_distribution.txt" file.');
