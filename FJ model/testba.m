clc;
clear;

% ��������
n = 100;       % �ڵ�����
m0 = 5;        % ��ʼ���õĽڵ�����
m = 1;         % ÿ�������ڵ����ӵ����нڵ������

% �����ޱ������
A = scalefree(n, m0, m);

% ����ڵ�ȷֲ�
degree_distribution = full(sum(A, 2));

% ���ƽڵ�ȷֲ�ֱ��ͼ
figure;
histogram(degree_distribution, 'Normalization', 'probability');
title('Degree Distribution of Generated Graph');
xlabel('Degree');
ylabel('Probability');

% �������ɵ�ͼ
figure;
spy(A);  % ����ϡ������ͼ��
title('Generated Graph');

% ��ӡ���ɵ�ͼ�Ļ�����Ϣ
fprintf('Number of nodes: %d\n', n);
fprintf('Number of edges: %d\n', nnz(A)/2); % ��Ϊ�ڽӾ����ǶԳƵģ����Աߵ������Ƿ���Ԫ������һ��

% ���� degree_distribution �ǽڵ�ȷֲ�������
figure;
histogram(degree_distribution, 'Normalization', 'probability');
title('Degree Distribution of Generated Graph');
xlabel('Degree');
ylabel('Probability');
