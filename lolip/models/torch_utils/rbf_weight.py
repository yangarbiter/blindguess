
import torch
from torch.nn import PairwiseDistance

def rbf(alpha, gamma):
    phi = torch.exp(-gamma * alpha.pow(2))
    return phi

def pairwise_weighting(X, y, gamma, norm):
    flat_X = X.flatten(start_dim=1)
    pdist = PairwiseDistance(p=norm)
    ret = []
    for i, x in enumerate(flat_X):
        ret.append(((y[i] != y) * rbf(pdist(x, flat_X), gamma=gamma)).sum())

    return 
