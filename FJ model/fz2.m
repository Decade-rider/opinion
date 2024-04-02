clc, clear;
% ��������
n = 10;       % �ڵ�����
m0 = 5;       % ��ʼ���õĽڵ�����
m = 1;        % ÿ�������ڵ����ӵ����нڵ������
tMax = 100;   % ���ģ������
c_eps = 1e-6; % ��������

% �����ޱ������
A = scalefree(n, m0, m);

% ���ɹ۵�������������Ӿ�ֵΪ1����׼��Ϊ0.5����̬�ֲ�
mu = 1;
sigma = 0.5;
s = normrnd(mu, sigma, n, 1);

% ���� Friedkin-Johnsen ģ�ͣ�ʹ�õ���������
[equilibrium, opinions] = friedkinJohnsenIterative(A, s, tMax, c_eps, 'plot');
