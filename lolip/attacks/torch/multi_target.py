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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = torch.utils.data.TensorDataset(self._preprocess_x(X))
    loader = torch.utils.data.DataLoader(dataset,
        batch_size=self.batch_size, shuffle=False, num_workers=1)

    ret = []
    for [x] in loader:
      x = x.to(device)

      r = []
      for j in range(self.n_classes):
        yp = j * torch.ones(self.batch_size).long().to(device)
        r.append(self.attack_fn(x=x, y=yp).detach().cpu().numpy())
      r = np.array(r).T
      import ipdb; ipdb.set_trace()

      for i in range(self.batch_size):
        pred = self.model_fn(torch.Tensor(r[i]).to(device)).argmax(1).cpu().numpy()
        ret.append(x[np.where(pred != y[i])[0]].cpu().numpy())

    return np.concatenate(ret, axis=0).transpose(0, 2, 3, 1)