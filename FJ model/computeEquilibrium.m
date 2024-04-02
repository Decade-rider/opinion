function equilibrium = computeEquilibrium(A, B, s, maxIterations, tolerance)
    % computeEquilibrium - ʹ�õ�����������ƽ��״̬
    %
    %   ����:
    %       [A] = NxN �ڽӾ���
    %       [B] = NxN ��̶Ⱦ���
    %       [s] = Nx1 �۵�����
    %       [maxIterations] = ����������
    %       [tolerance] = ��������
    %
    %   ���:
    %       [equilibrium] = ������ƽ��״̬

    % ʹ�ó�ʼ�۵�������ʼ�� x
    x = s;
    
    % �������¹۵�
    for iter = 1:maxIterations
        % ʹ�� Friedkin-Johnsen ���̸��¹۵�
        x_new = A * x + B * s;
        
        % ����Ƿ�����
        if max(abs(x_new - x)) < tolerance
            equilibrium = x_new;
            disp(['Converged after ' num2str(iter) ' iterations']);
            return;
        end
        
        % ���¹۵�����
        x = x_new;
    end
    
    % ���û�������������������ֵ
    equilibrium = x;
    disp('Maximum iterations reached without convergence');
end
