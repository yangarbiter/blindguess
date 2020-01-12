"""
Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing their Input Gradients
"""

import torch
from torch.autograd import grad


def gradient_regularization(model, loss_fn, optimizer, x, y, lambd):


    def model_grad(x):
        x.requires_grad_(True)
        ret = grad(loss_fn, x, retain_graph=True)[0]
        x.requires_grad_(False)
        return ret

    optimizer.zero_grad()
    x.requires_grad_(True)
    lx = loss_fn(model(x), y)
    x_grad = grad(lx, x, retain_graph=True)[0]
    lx.backward()
    regularization = torch.norm(x_grad.flatten(start_dim=1), dim=1, p=2)**2
    regularization = torch.sum(regularization)
    #x.grad.zero_()
    x.requires_grad_(False)

    outputs = model(x)
    loss_natural = loss_fn(outputs, y)

    loss = loss_natural + lambd * regularization

    return outputs, loss
