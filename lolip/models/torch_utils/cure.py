"""
https://github.com/F-Salehi/CURE_robustness/blob/master/CURE/CURE.py
"""

import torch
from torch.autograd.gradcheck import zero_gradients


def find_z(model, loss_fn, inputs, targets, h, device):
    '''
    Finding the direction in the regularizer
    '''
    inputs.requires_grad_()
    outputs = model.eval()(inputs)
    loss_z = loss_fn(outputs, targets)
    loss_z.backward(torch.ones(targets.size()).to(device))
    grad = inputs.grad.data + 0.0
    #norm_grad = grad.norm().item()
    z = torch.sign(grad).detach() + 0.
    z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)
    zero_gradients(inputs)
    model.zero_grad()

    return z#, norm_grad


def cure_loss(model, loss_fn, inputs, targets, h=3., lambda_=4, device="cuda"):
    z = find_z(model, loss_fn, inputs, targets, h, device)

    inputs.requires_grad_()
    outputs_pos = model.eval()(inputs + z)
    outputs_orig = model.eval()(inputs)

    loss_pos = loss_fn(outputs_pos, targets)
    loss_orig = loss_fn(outputs_orig, targets)
    grad_diff = torch.autograd.grad(
            loss_pos-loss_orig, inputs,
            grad_outputs=torch.ones(targets.size()).to(device),
            create_graph=True)[0]
    reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
    model.zero_grad()

    return outputs_orig, loss_orig + torch.sum(lambda_ * reg) / float(inputs.size(0))