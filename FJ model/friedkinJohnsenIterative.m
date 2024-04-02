function [equilibrium, opinions] = friedkinJohnsenIterative(A, s, tMax, c_eps, varargin)
    % friedkinJohnsenIterative - ʹ�õ�������ģ�� Friedkin-Johnsen ģ��
    %
    %   ����:
    %       [A] = NxN �ڽӾ���
    %       [s] = Nx1 �۵�����
    %       [tMax] = ����������
    %       [c_eps] = ��������
    %       ��������:
    %           'plot' = ���ƹ۵� vs ʱ��ͼ
    %
    %   ���:
    %       [equilibrium] = ������ƽ��״̬
    %       [opinions] = Nxt ÿ�ֵĹ۵����

    wantPlot = false;

    % ��������
    if (~isempty(varargin))
        for c=1:length(varargin)
            switch varargin{c}
                case {'plot'}
                    wantPlot = true;
                otherwise
                    error(['Invalid optional argument, ', varargin{c}]);
            end % switch
        end % for
    end % if

    % ��ʼ��
    [N, ~, ~] = size(A);
    B = diag(diag(A)); % Stubborness matrix
    equilibrium = computeEquilibrium(A, B, s, tMax, c_eps);

    % �������¹۵�
    opinions = zeros(N, tMax);
    x = s;
    for t = 1:tMax
        % ������һ�ֹ۵�
        x_next = A * x + B * s;

        % ���¹۵����
        opinions(:, t) = x;

        % ����Ƿ�ﵽ��������
        if max(abs(x_next - x)) < c_eps
            disp(['[Friedkin-Johnsen] �� ' num2str(t) ' �ֺ�ﵽƽ�⡣']);
            equilibrium = x_next;
            opinions(:, t+1:end) = [];
            break;
        end

        % ���¹۵�����
        x = x_next;
    end

    % ���ƹ۵� vs ʱ��ͼ
    if (wantPlot)
        plotOpinions2(opinions);
    end

end
