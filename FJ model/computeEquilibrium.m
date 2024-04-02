function equilibrium = computeEquilibrium(A, B, s, maxIterations, tolerance)
    % computeEquilibrium - 使用迭代方法计算平衡状态
    %
    %   输入:
    %       [A] = NxN 邻接矩阵
    %       [B] = NxN 顽固度矩阵
    %       [s] = Nx1 观点向量
    %       [maxIterations] = 最大迭代次数
    %       [tolerance] = 收敛容限
    %
    %   输出:
    %       [equilibrium] = 期望的平衡状态

    % 使用初始观点向量初始化 x
    x = s;
    
    % 迭代更新观点
    for iter = 1:maxIterations
        % 使用 Friedkin-Johnsen 方程更新观点
        x_new = A * x + B * s;
        
        % 检查是否收敛
        if max(abs(x_new - x)) < tolerance
            equilibrium = x_new;
            disp(['Converged after ' num2str(iter) ' iterations']);
            return;
        end
        
        % 更新观点向量
        x = x_new;
    end
    
    % 如果没有收敛，返回最后计算的值
    equilibrium = x;
    disp('Maximum iterations reached without convergence');
end
