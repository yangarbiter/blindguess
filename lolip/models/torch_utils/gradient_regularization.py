"""
Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing their Input Gradients
"""

import torch

def gradient_regularization(model, loss_fn, optimizer, x, y, lambd):

    def model_grad(x):
        x.requires_grad_(True)
        lx = loss_fn(model(x), y)
        lx.backward()
        ret = x.grad
        x.grad.zero_()
        x.requires_grad_(False)
        return ret

    regularization = lambd * torch.norm(model_grad(x), p=2) ** 2

    optimizer.zero_grad()
    outputs = model(x)
    loss_natural = loss_fn(outputs, y)

    loss = loss_natural + lambd * regularization

    return outputs, loss
