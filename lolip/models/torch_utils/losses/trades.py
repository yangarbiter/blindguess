"""
https://raw.githubusercontent.com/yaodongyu/TRADES/master/trades.py
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.modules.loss import _Loss


def trades_loss(model,
                loss_fn,
                x_natural,
                y,
                norm,
                optimizer,
                clip_min=None,
                clip_max=None,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                reduction='none',
                version=None,
                device="gpu"):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    if norm == np.inf:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
                loss = loss.mean()
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            if clip_min is not None and clip_max is not None:
                x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif norm == 2:
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
                loss = loss.mean()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            if clip_min is not None and clip_max is not None:
                delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
            else:
                delta.data.sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f"Not supported Norm {norm}")
    model.train()

    if clip_min is not None and clip_max is not None:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    x_adv = Variable(x_adv, requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    outputs = model(x_natural)
    loss_natural = loss_fn(outputs, y)
    loss_robust = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                               F.softmax(model(x_natural), dim=1))
    loss_robust = loss_robust

    if version is None:
        loss = loss_natural + beta * loss_robust
    return outputs, loss
