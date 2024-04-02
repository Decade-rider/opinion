function [N, A, B, equilibrium, cond_eye_A_B] = preprocessGraph(A, s)
%PREPROCESSGRAPH Run checks and get basic values from social graph

N = size(A,1);
if (size(s,1) ~= N)
    error('Wrong opinion vector size');
end

% Balance preconditioning
D = diag(1 ./ max(abs(A), [], 2));  % Compute diagonal matrix D
C = eye(N);  % Choose C as the identity matrix

% Compute balanced system
tilde_A = D * A;
tilde_b = D * s;

% Update A and B with balanced system
A = tilde_A;
B = diag(diag(A)); % Stubbornness matrix

% Compute the condition number of (eye(N) - (A - B))
cond_eye_A_B = cond(eye(N) - (A - B));

% Expected equilibrium from Kleinberg Model
equilibrium = (eye(N) - (A - B)) \ (B * tilde_b);

end
