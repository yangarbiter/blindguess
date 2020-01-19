import numpy as np
import torch
from torch.nn import PairwiseDistance
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

def rbf(alpha, gamma):
    phi = torch.exp(-gamma * alpha.pow(2))
    return phi

def pairwise_weighting(X, y, gamma, norm):
    flat_X = X.flatten(start_dim=1)
    pdist = PairwiseDistance(p=norm)
    ret = []
    for i, x in enumerate(flat_X):
        ret.append(((y[i] != y) / rbf(pdist(x, flat_X), gamma=gamma)).sum())
    return torch.stack(ret)

def rbfw_loss(model, loss_fn, x, y, gamma, norm):
    gamma = torch.abs(gamma)
    outputs = model(x)
    loss = loss_fn(outputs, y)
    weights = pairwise_weighting(X=x, y=y, gamma=gamma, norm=norm)
    weights = weights / len(x)

    loss = (weights * loss).sum()

    return outputs, loss

def rbf_weight(xi, yi, X, y, gamma, norm):
    dif = np.linalg.norm((X - xi[:, np.newaxis]), ord=norm, axis=0)
    return (y == yi) * np.exp(-gamma * dif)

def neighbor_weights(X: np.array, y: np.array, radius: float, gamma, norm):
    nn = NearestNeighbors(radius=radius)
    X = X.reshape((len(X), -1))
    nn.fit(X)
    neibs = nn.radius_neighbors(X)

    ret = []
    for i, idxs in enumerate(neibs):
        ret.append(rbf_weight(X[i], y[i], X[idxs], y[idxs], gamma, norm).sum())
    
    ret = 1. / np.asarray(ret)
    ret = ret / np.sum(ret) * len(X) # normalize
    return ret
