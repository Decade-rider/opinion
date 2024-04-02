function [equilibrium, opinions] = friedkinJohnsenIterative(A, s, tMax, c_eps, varargin)
    % friedkinJohnsenIterative - 使用迭代方法模拟 Friedkin-Johnsen 模型
    %
    %   输入:
    %       [A] = NxN 邻接矩阵
    %       [s] = Nx1 观点向量
    %       [tMax] = 最大迭代次数
    %       [c_eps] = 收敛限制
    %       其他参数:
    %           'plot' = 绘制观点 vs 时间图
    %
    %   输出:
    %       [equilibrium] = 期望的平衡状态
    %       [opinions] = Nxt 每轮的观点矩阵

    wantPlot = false;

    % 解析输入
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

    % 初始化
    [N, ~, ~] = size(A);
    B = diag(diag(A)); % Stubborness matrix
    equilibrium = computeEquilibrium(A, B, s, tMax, c_eps);

    % 迭代更新观点
    opinions = zeros(N, tMax);
    x = s;
    for t = 1:tMax
        % 计算下一轮观点
        x_next = A * x + B * s;

        % 更新观点矩阵
        opinions(:, t) = x;

        % 检查是否达到收敛条件
        if max(abs(x_next - x)) < c_eps
            disp(['[Friedkin-Johnsen] 在 ' num2str(t) ' 轮后达到平衡。']);
            equilibrium = x_next;
            opinions(:, t+1:end) = [];
            break;
        end

        % 更新观点向量
        x = x_next;
    end

    % 绘制观点 vs 时间图
    if (wantPlot)
        plotOpinions2(opinions);
    end

end
