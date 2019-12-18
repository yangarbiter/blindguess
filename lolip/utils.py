import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import _Loss


class LocalLip(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(LocalLip, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return -F.mse_loss(input, target, reduction=self.reduction)

def local_lip(model, x, xp):
    return torch.norm(model(x) - model(xp)) / torch.norm(x - xp) 


def estimate_local_lip(model, x, norm, perturb_steps=10, step_size=0.003, epsilon=0.01):
    model.eval()
    batch_size = len(x)
    # generate adversarial example
    if norm == np.inf:
        x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = local_lip(model, x, x_adv)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif norm == 2:
        delta = 0.001 * torch.randn(x.shape).cuda().detach()
        delta = torch.autograd.Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            x_adv = x + delta

            # optimize
            optimizer.zero_grad()
            with torch.enable_grad():
                loss = (-1) * LocalLip(model(x), model(x_adv))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer.step()

            # projection
            delta.data.add_(x)
            delta.data.clamp_(0, 1).sub_(x)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = x + delta
    else:
        raise ValueError(f"Unsupported norm {norm}")
    return x_adv
