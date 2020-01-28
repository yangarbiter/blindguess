import cvxpy as cp
import numpy as np

def rbf_kernel(x, X, gamma):
    dist = np.linalg.norm(x - X, ord=2, axis=1)
    return np.exp(- gamma * dist)

#def soft_rbf_ap(X, y, eps, sep_measure, gamma):
#    c = cp.Variable(len(X))
#
#    constraints = [c>=0, c<=1]
#    for i in range(len(X)):
#        oppX = X[y!=y[i]]
#        constraints.append(cp.sum(c * rbf_kernel(X[i], oppX, gamma)))
#    
#    objective = cp.Maximize(cp.sum(c))
#    prob = cp.Problem(objective, constraints)
#    result = prob.solve()
#
#    return c.value

def soft_rbf_ap(X, y, eps, sep_measure, gamma):
    weights = np.zeros(len(X))
    for i in np.unique(y):
        idx = np.where(y == i)[0]
        others = np.where(y != i)[0]
        for j in idx:
            weights[j] = rbf_kernel(X[j], X[others], gamma)

    return weights