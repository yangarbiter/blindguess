from functools import partial

import numpy as np
import torch

from ..base import AttackModel
from .projected_gradient_descent import projected_gradient_descent

class MultiTarget(AttackModel):

  def __init__(self, model_fn, eps, eps_iter, nb_iter, norm, n_classes,
               loss_fn=None, clip_min=None, clip_max=None, y=None,
               batch_size=128, rand_init=True, rand_minmax=None):
    self.n_classes = n_classes
    self.model_fn = model_fn
    self.eps = eps
    self.batch_size = batch_size
    self.attack_fn = partial(projected_gradient_descent, model_fn=model_fn,
      eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=norm, loss_fn=loss_fn,
      clip_min=clip_min, clip_max=clip_max, targeted=True, rand_init=rand_init,
      rand_minmax=rand_minmax)

  def _preprocess_x(self, X):
    return torch.from_numpy(X.transpose(0, 3, 1, 2)).float()

  def perturb(self, X, y=None, eps=None):
    """
    y: correct: label
    """
    #self.model_fn.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = torch.utils.data.TensorDataset(
      self._preprocess_x(X), torch.from_numpy(y).long())
    loader = torch.utils.data.DataLoader(dataset,
      batch_size=self.batch_size, shuffle=False, num_workers=1)

    ret = []
    for [x, y] in loader:
      x, y = x.to(device), y.to(device)

      pred = self.model_fn(x)[:, y]

      r = []
      scores = []
      for j in range(self.n_classes):
        yp = j * torch.ones(self.batch_size).long().to(device)
        adv_x = self.attack_fn(x=x, y=yp).detach()
        scores.append((self.model_fn(adv_x)[:, yp] - pred).cpu().numpy())
        r.append(adv_x.cpu().numpy())
      scores = np.array(scores)
      r = np.array(r)

      idx = scores.argmax(axis=0)

      for i in range(self.batch_size):
        ret.append(r[i, idx[i]])

    return np.concatenate(ret, axis=0).transpose(0, 2, 3, 1)