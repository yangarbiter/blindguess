import copy
from functools import partial

import numpy as np
import torch
from torch.autograd import Variable

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
    self.model_fn.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = torch.utils.data.TensorDataset(
      self._preprocess_x(X), torch.from_numpy(y).long())
    loader = torch.utils.data.DataLoader(dataset,
      batch_size=self.batch_size, shuffle=False, num_workers=1)

    ret = []
    for [x, y] in loader:
      x = x.to(device)

      pred = self.model_fn(x).detach().cpu().numpy()
      pred = np.array([pred[i, yi] for i, yi in enumerate(y)])

      r = []
      scores = []
      for j in range(self.n_classes):
        yp = j * torch.ones(len(x)).long().to(device)
        adv_x = self.attack_fn(x=x, y=yp).detach()
        scores.append(self.model_fn(adv_x)[:, j].detach().cpu().numpy() - pred)
        r.append(adv_x.cpu().numpy())
      scores = np.array(scores)
      idx = scores.argmax(axis=0)

      r = np.array(r)
      for i in range(len(x)):
        ret.append(r[idx[i], i])

    return np.array(ret).transpose(0, 2, 3, 1)


def _mt_whitebox(model,
                X,
                y,
                epsilon=args.epsilon,
                num_steps=args.num_steps,
                step_size=args.step_size):
    model.eval()
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    batch_size = len(X)

    # define adversarial example
    X_adv_list = [torch.zeros(X.size())] * 10
    loss_list = []

    for idx in range(10):
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
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
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
    x_adv_mt = Variable(torch.clamp(x_adv_mt, 0.0, 1.0), requires_grad=False)
    err_mt = (model(x_adv_mt).data.max(1)[1] != y.data).float().sum()
    # print('err mt (white-box): ', err_mt)

    return err, err_mt