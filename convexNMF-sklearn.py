function [W, H, G, cost] = convexnmf(V, num_basis_elems, config)
% convexnmf Decompose a matrix V into VGH using Convex-NMF [1] by
% minimizing the Euclidean distance between V and VGH. W = VG is a basis
% matrix, where the columns of G form convex combinations of the data
% points in V, and H is the encoding matrix that encodes the input V in
% terms of the basis W. Unlike NMF, V can have mixed sign. The columns of
% W can be interpreted as cluster centroids (there is a connection between
% Convex-NMF and K-means clustering), while H shows the soft membership of
% each data point to the clusters.
%
% Inputs:
%   V: [matrix]
%       m-by-n matrix containing data to be decomposed.
%   num_basis_elems: [positive scalar]
%       number of basis elements (columns of G/rows of H) for 1 source.
%   config: [structure] (optional)
%       structure containing configuration parameters.
%       config.G_init: [non-negative matrix] (default: use sklearn NMF to initialize)
%           initialize 1 convex combination matrix with a n-by-num_basis_elems
%           matrix.
%       config.H_init: [non-negative matrix] (default: use sklearn NMF to initialize)
%           initialize 1 encoding matrix with a num_basis_elems-by-n
%           non-negative matrix.
%       config.G_fixed: [boolean] (default: false)
%           indicate if the convex combination matrix is fixed during the
%           update equations.
%       config.H_fixed: [boolean] (default: false)
%           indicate if the encoding matrix is fixed during the update
%           equations.
%       config.G_sparsity: [non-negative scalar] (default: 0)
%           sparsity level for the convex combination matrix.
%       config.maxiter: [positive scalar] (default: 100)
%           maximum number of update iterations.
%       config.tolerance: [positive scalar] (default: 1e-3)
%           maximum change in the cost function between iterations before
%           the algorithm is considered to have converged.
%
% Outputs:
%   W: [matrix]
%       m-by-num_basis_elems basis matrix. W = V*G.
%   H: [non-negative matrix]
%       num_basis_elems-by-n non-negative encoding matrix.
%   G: [non-negative matrix]
%       n-by-num_basis_elems matrix of convex combinations of the columns
%       of V.
%   cost: [vector]
%       value of the cost function after each iteration.
%
% References:
%   [1] C. Ding, T. Li, and M. I. Jordan, "Convex and Semi-Nonnegative
%       Matrix Factorizations," IEEE Trans. Pattern Analysis Machine
%       Intelligence, vol. 32, no. 1, pp. 45-55, Jan. 2010.
%
% NMF Toolbox
% Colin Vaz - cvaz@usc.edu
% Signal Analysis and Interpretation Lab (SAIL) - http://sail.usc.edu
% University of Southern California
% 2015

% Check if configuration structure is given.
if nargin < 3
    config = struct;
end

config = ValidateParameters('convexnmf', config, V, num_basis_elems);

if ~isfield(config, 'G_init') || isempty(config.G_init)
    % 使用 sklearn 训练 NMF 模型并得到初始的 G 和 H
    model = fit_transform(V, num_basis_elems, config);
    G_init = model.components_.T;  % 转置得到 n-by-num_basis_elems 的矩阵
    H_init = model.transform(V).T;  % 转置得到 num_basis_elems-by-n 的矩阵
else
    G_init = config.G_init;
    H_init = config.H_init;
end

if ~isfield(config, 'G_fixed')
    config.G_fixed = false;
end

if ~isfield(config, 'G_sparsity') || isempty(config.G_sparsity)
    config.G_sparsity = 0;
end

G = G_init;
H = H_init;
G = G * diag(1 ./ sum(G, 1));
W = V * G;

V_V_pos = 0.5 * (abs(V' * V) + V' * V);
V_V_neg = 0.5 * (abs(V' * V) - V' * V);

cost =zeros(config.maxiter, 1);

for iter = 1 : config.maxiter
    % Update convex combination matrix
    if ~config.G_fixed
        G = G .* sqrt(((V_V_pos + V_V_neg * G * H) * H') ./ ((V_V_neg + V_V_pos * G * H) * H' + config.G_sparsity));
        G = G * diag(1 ./ sum(G, 1));
    end
    W = V * G;
    
    % Update encoding matrix
    if ~config.H_fixed
        H = H .* sqrt((G' * (V_V_pos + V_V_neg * G * H)) ./ (G' * (V_V_neg + V_V_pos * G * H)));
    end

    % Calculate cost for this iteration
    V_hat = ReconstructFromDecomposition(W, H);
    cost(iter) = 0.5 * sum(sum((V - V_hat).^2));
    
    % Stop iterations if change in cost function less than the tolerance
    if iter > 1 && cost(iter) < cost(iter-1) && cost(iter-1) - cost(iter) < config.tolerance
        cost = cost(1 : iter);  % trim vector
        break;
    end
end

end

function model = fit_transform(V, num_basis_elems, config)
    % 使用 sklearn 的 NMF 模型拟合数据并返回模型对象
    from sklearn.decomposition import NMF
    model = NMF(n_components=num_basis_elems, init='custom', solver='mu', max_iter=config.maxiter, tol=config.tolerance)
    model.fit(V, W=config.G_init, H=config.H_init)
    return model
end

function V_hat = ReconstructFromDecomposition(W, H)
    % 重构 V_hat
    V_hat = W * H;
end