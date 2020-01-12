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
        #x.grad.zero_()
        x.requires_grad_(False)
        return ret

    optimizer.zero_grad()
    regularization = torch.norm(model_grad(x).flatten(start_dim=1), dim=1, p=2)**2
    regularization = torch.sum(regularization)

    outputs = model(x)
    loss_natural = loss_fn(outputs, y)

    loss = loss_natural + lambd * regularization

    return outputs, loss
