import copy
from functools import partial

import numpy as np
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..base import AttackModel


class HongPGD(AttackModel):

  def __init__(self, model_fn, eps, eps_iter, nb_iter, norm, n_classes,
               loss_fn=None, y=None, batch_size=128):
    self.n_classes = n_classes
    self.model_fn = model_fn
    self.eps = eps
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.batch_size = batch_size

  def _preprocess_x(self, X):
    return torch.from_numpy(X.transpose(0, 3, 1, 2)).float()

  def perturb(self, X, y=None, eps=None):
    """
    y: correct label
    """
    self.model_fn.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = torch.utils.data.TensorDataset(
      self._preprocess_x(X), torch.from_numpy(y).long())
    loader = torch.utils.data.DataLoader(dataset,
      batch_size=self.batch_size, shuffle=False, num_workers=1)

    ret = []
    for [x, y] in loader:
      x, y = x.to(device), y.to(device)

      advx = _pgd_whitebox(
          self.model_fn, x, y,
          epsilon=self.eps, num_steps=self.nb_iter, step_size=self.eps_iter)

      ret.append(advx)

    return np.concatenate(ret, axis=0).transpose(0, 2, 3, 1)


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=0.031,
                  num_steps=20,
                  step_size=0.01):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to("cuda")
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    #print('err pgd (white-box): ', err_pgd)
    return X_pgd.detach().cpu().numpy()