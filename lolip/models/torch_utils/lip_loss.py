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


def one_hot(y, nb_classes, smoothing=1e-6):
    y_onehot = torch.zeros((len(y), nb_classes)) + smoothing
    y_onehot = y_onehot.scatter_(1, y.unsqueeze(1), 1 - (smoothing*(nb_classes-1)))
    return y_onehot

def lip_loss(model,
                loss_fn,
                x_natural,
                y,
                norm,
                optimizer,
                nb_classes=10,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                version=None,
                device="gpu"):
    # define KL-loss
    #criterion_kl = nn.KLDivLoss(size_average=False)
    #if version == "plus":
    #    criterion_kl = nn.KLDivLoss(reduction='none')
    #else:
    #    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    if norm == np.inf:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                #loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                #                       one_hot(y, nb_classes))
                loss_kl = loss_fn(model(x_adv), y)
                #loss_kl = torch.sum(loss_kl, dim=1) \
                #        / torch.norm(torch.flatten(x_adv - x_natural, start_dim=1), p=norm, dim=1)
                if len(loss_kl.shape) == 2:
                    loss_kl = torch.sum(loss_kl, dim=1) \
                            / torch.norm(torch.flatten(x_adv - x_natural, start_dim=1), p=norm, dim=1)
                else:
                    loss_kl = torch.sum(loss_kl) \
                            / torch.norm(torch.flatten(x_adv - x_natural, start_dim=1), p=norm, dim=1)
                loss_kl = torch.sum(loss_kl)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif norm == 2:
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            x_adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                #loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                #                           one_hot(y, nb_classes))
                loss_kl = (-1) * loss_fn(model(x_adv), y)
                if len(loss_kl.shape) == 2:
                    loss_kl = torch.sum(loss_kl, dim=1) \
                            / torch.norm(torch.flatten(x_adv - x_natural, start_dim=1), p=norm, dim=1)
                else:
                    loss_kl = torch.sum(loss_kl) \
                            / torch.norm(torch.flatten(x_adv - x_natural, start_dim=1), p=norm, dim=1)
                loss = torch.sum(loss_kl)
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
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss

    #loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
    #                       one_hot(y, nb_classes))
    loss_kl = loss_fn(model(x_adv), y)
    #loss_kl = torch.sum(loss_kl, dim=1) \
    #        / torch.norm(torch.flatten(x_adv - x_natural, start_dim=1), p=norm, dim=1)
    if len(loss_kl.shape) == 2: # kld loss
        loss_kl = torch.sum(loss_kl, dim=1) \
                / torch.norm(torch.flatten(x_adv - x_natural, start_dim=1), p=norm, dim=1)
    else:
        loss_kl = torch.sum(loss_kl) \
                / torch.norm(torch.flatten(x_adv - x_natural, start_dim=1), p=norm, dim=1)
    loss_robust = torch.sum(loss_kl)

    return loss_robust
