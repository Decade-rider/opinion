clc,clear;
% ��������
n = 100; % �ڵ�����
m0 = 5; % ��ʼ���õĽڵ�����
m = 3; % ÿ�������ڵ����ӵ����нڵ������
tMax = 100; % ���ģ������
c_eps = 1e-6; % ��������

% �����ޱ������
A = scalefree(n, m0, m);

% ���ɹ۵�������������Ӿ��ȷֲ�
lower_bound = 0; % ���ȷֲ����½�
upper_bound = 1; % ���ȷֲ����Ͻ�
s = unifrnd(lower_bound, upper_bound, n, 1);

% ���� Friedkin-Johnsen ģ��
[equilibrium, opinions] = friedkinJohnsen(A, s, tMax, c_eps, 'plot');
