
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
