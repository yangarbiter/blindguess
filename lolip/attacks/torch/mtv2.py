import copy
from functools import partial

import numpy as np
import torch
from torch.autograd import Variable

from ..base import AttackModel

class MultiTargetV2(AttackModel):

  def __init__(self, model_fn, eps, eps_iter, nb_iter, norm, n_classes,
          clip_min=None, clip_max=None, loss_fn=None, y=None, batch_size=128):
    self.n_classes = n_classes
    self.model_fn = model_fn
    self.eps = eps
    self.eps_iter = eps_iter
    self.clip_min = clip_min
    self.clip_max = clip_max
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

      advx = _mt_whitebox(
          self.model_fn, x, y, n_classes=self.n_classes,
          epsilon=self.eps, num_steps=self.nb_iter, step_size=self.eps_iter,
          clip_min=self.clip_min, clip_max=self.clip_max
      )

      ret.append(advx)

    return np.concatenate(ret, axis=0).transpose(0, 2, 3, 1)



def _mt_whitebox(model,
                X,
                y,
                n_classes=10,
                epsilon=0.031,
                num_steps=20,
                step_size=0.003,
                clip_min=0.,
                clip_max=1.,):
    model.eval()
    #out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    batch_size = len(X)

    # define adversarial example
    X_adv_list = [torch.zeros(X.size())] * n_classes
    loss_list = []

    for idx in range(n_classes):
        x_adv = X.detach()
        for _ in range(num_steps):
            x_adv.requires_grad_()
            output = model(x_adv)
            with torch.enable_grad():
                loss_temp = 0.0
                for j in range(batch_size):
                    loss_temp += output[j][idx] - output[j][y[j]]
            grad = torch.autograd.grad(loss_temp, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
            if clip_min is not None and clip_max is not None:
                x_adv = torch.clamp(x_adv, clip_min, clip_max)
        if clip_min is not None and clip_max is not None:
            x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
        else:
            x_adv = Variable(x_adv, requires_grad=False)

        with torch.no_grad():
            X_adv_list[idx] = copy.deepcopy(x_adv.detach())
            output = model(x_adv)
            loss_temp = torch.zeros((batch_size,))
            for j in range(batch_size):
                loss_temp[j] = output[j][idx] - output[j][y[j]]
            loss_list.append(loss_temp)

    # calculate the max
    loss_matrix = torch.stack(loss_list)
    _, index_top2 = torch.topk(loss_matrix, 2, dim=0)

    # select x_adv multi-target
    x_adv_mt = torch.zeros(X.size())
    for idx_batch in range(batch_size):
        if index_top2[0][idx_batch].cuda() == y[idx_batch]:
            x_adv_mt[idx_batch] = copy.deepcopy(X_adv_list[index_top2[1][idx_batch]][idx_batch])
        else:
            x_adv_mt[idx_batch] = copy.deepcopy(X_adv_list[index_top2[0][idx_batch]][idx_batch])
    x_adv_mt = x_adv_mt.cuda()
    if clip_min is not None and clip_max is not None:
        x_adv_mt = Variable(torch.clamp(x_adv_mt, clip_min, clip_max), requires_grad=False)
    else:
        x_adv_mt = Variable(x_adv_mt, requires_grad=False)
    #err_mt = (model(x_adv_mt).data.max(1)[1] != y.data).float().sum()
    # print('err mt (white-box): ', err_mt)

    return x_adv_mt.detach().cpu().numpy()
