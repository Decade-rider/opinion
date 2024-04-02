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

% ���ڵ�ȷֲ����浽�ı��ļ���
fileID = fopen('degree_distribution.txt', 'w');
fprintf(fileID, 'Degree Frequency\n');
for i = 1:length(degree_counts)
    fprintf(fileID, '%d %d\n', i, degree_counts(i));
end
fclose(fileID);

disp('Degree distribution saved to "degree_distribution.txt" file.');
