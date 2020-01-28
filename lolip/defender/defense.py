"""
"""
from __future__ import division
import logging
from functools import partial
from copy import deepcopy, copy

import tensorflow as tf
import networkx as nx
from networkx.algorithms.approximation.vertex_cover import min_weighted_vertex_cover
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

from .robust_nn.eps_separation import find_eps_separated_set, find_num_collision
from ..variables import auto_var

logger = logging.getLogger(__name__)

def adversarial_pruning(X, y, eps, sep_measure):
    if len(np.unique(y)) != 2:
        raise ValueError("Can only deal with number of classes = 2"
                         "got %d", len(np.unique(y)))
    y = y.astype(int)*2-1
    augX, augy, _ = find_eps_separated_set(X, eps, y, ord=sep_measure)
    augy = (augy+1)//2
    return augX, augy

def approx_ap(X, y, eps, sep_measure):
    ori_shape = list(X.shape)
    X = X.reshape((len(X), -1))
    nn = NearestNeighbors(n_jobs=-1, p=sep_measure).fit(X)
    ind = nn.radius_neighbors(X, radius=eps, return_distance=False)
    G = nx.Graph()
    G.add_nodes_from(list(range(len(X))))
    for i in range(len(X)):
        for j in ind[i]:
            if y[i] != y[j]:
                G.add_edge(i, j)
    idx = min_weighted_vertex_cover(G)
    idx = list(set(range(len(X))) - idx)
    X = X.reshape([len(X)] + ori_shape[1:])
    return X[idx], y[idx]

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
