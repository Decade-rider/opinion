clc;
clear;

% ��������
n = 1000;     % �ڵ�����
m0 = 5;       % ��ʼ���õĽڵ�����
m = 1;        % ÿ�������ڵ����ӵ����нڵ������

% �����ޱ������
A = scalefree(n, m0, m);

% ����ڵ��
degree = sum(A);

% ����ڵ�ȷֲ�
degree_counts = hist(degree, 1:max(degree));

% ���ƶȷֲ��Ķ���-����ͼ
figure;
loglog(1:max(degree), degree_counts, 'o');
title('Degree Distribution of Generated Graph (Log-Log Scale)');
xlabel('Degree (k)');
ylabel('Frequency (P(k))');
