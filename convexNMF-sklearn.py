import numpy as np
from sklearn.decomposition import NMF

def convexnmf(V, num_basis_elems, config=None):
    def validate_parameters(config, V, num_basis_elems):
        # 验证参数并设置默认值
        if config is None:
            config = {}
        
        if 'G_init' not in config or config['G_init'] is None:
            # 使用 sklearn 的 NMF 模型训练并得到 G_init 和 H_init 的初始值
            model = fit_transform(V, num_basis_elems, config)
            G_init = model.components_.T  # 转置得到 n-by-num_basis_elems 的矩阵
            H_init = model.transform(V).T  # 转置得到 num_basis_elems-by-n 的矩阵
        else:
            G_init = config['G_init']
            H_init = config['H_init']
        
        if 'G_fixed' not in config:
            config['G_fixed'] = False
        
        if 'G_sparsity' not in config or config['G_sparsity'] is None:
            config['G_sparsity'] = 0
        
        if 'maxiter' not in config:
            config['maxiter'] = 100
        
        if 'tolerance' not in config:
            config['tolerance'] = 1e-3
        
        return config, G_init, H_init
    
    def fit_transform(V, num_basis_elems, config):
        # 使用 sklearn 的 NMF 模型拟合数据并返回模型对象
        model = NMF(n_components=num_basis_elems, init='custom', solver='mu', max_iter=config['maxiter'], tol=config['tolerance'])
        model.fit(V, W=config['G_init'], H=config['H_init'])
        return model
    
    def reconstruct_from_decomposition(W, H):
        # 重构 V_hat
        V_hat = np.dot(W, H)
        return V_hat
    
    config, G_init, H_init = validate_parameters(config, V, num_basis_elems)
    
    G = G_init
    H = H_init
    G = G * np.diag(1.0 / np.sum(G, axis=0))
    W = np.dot(V, G)
    
    V_V_pos = 0.5 * (np.abs(np.dot(V.T, V)) + np.dot(V.T, V))
    V_V_neg = 0.5 * (np.abs(np.dot(V.T, V)) - np.dot(V.T, V))
    
    cost = np.zeros(config['maxiter'])
    
    for iteration in range(config['maxiter']):
        # 更新凸组合矩阵
        if not config['G_fixed']:
            G = G * np.sqrt(np.dot((V_V_pos + np.dot(V_V_neg, np.dot(G, H))), H.T) / (np.dot((V_V_neg + np.dot(V_V_pos, np.dot(G, H))), H.T) + config['G_sparsity']))
            G = G * np.diag(1.0 / np.sum(G, axis=0))
        W = np.dot(V, G)
        
        # 更新编码矩阵
        if not config['H_fixed']:
            H = H * np.sqrt(np.dot(G.T, (V_V_pos + np.dot(V_V_neg, np.dot(G, H)))) / np.dot(G.T, (V_V_neg + np.dot(V_V_pos, np.dot(G, H)))))
        
        # 计算当前迭代的损失函数值
        V_hat = reconstruct_from_decomposition(W, H)
        cost[iteration] = 0.5 * np.sum((V - V_hat)**2)
        
        # 如果损失函数变化小于容差值，停止迭代
        if iteration > 0 and cost[iteration] < cost[iteration-1] and cost[iteration-1] - cost[iteration] < config['tolerance']:
            cost = cost[:iteration]  # 截断向量
            break
    
    return W, H, G, cost