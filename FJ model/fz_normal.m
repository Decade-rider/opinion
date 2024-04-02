clc,clear;
% ��������
n = 10; % �ڵ�����
m0 = 5; % ��ʼ���õĽڵ�����
m = 3; % ÿ�������ڵ����ӵ����нڵ������
tMax = 100; % ���ģ������
c_eps = 1e-6; % ��������

% �����ޱ������
A = scalefree(n, m0, m);

% ���ɹ۵�������������Ӿ�ֵΪ0.5����׼��Ϊ0.1����̬�ֲ�
mu = 1;
sigma = 0.5;
s = normrnd(mu, sigma, n, 1);

% ���� Friedkin-Johnsen ģ��
[equilibrium, opinions] = friedkinJohnsen(A, s, tMax, c_eps, 'plot');