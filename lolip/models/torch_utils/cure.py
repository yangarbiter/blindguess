"""
https://github.com/F-Salehi/CURE_robustness/blob/master/CURE/CURE.py
"""
from bisect import bisect_right

import torch
from torch.autograd.gradcheck import zero_gradients
from torch.optim.lr_scheduler import _LRScheduler

class CureMultiStepLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma once the number of epoch reaches one of the milestones. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        super(CureMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [1e-5 for base_lr in self.base_lrs]
        elif self.last_epoch == 1:
            return [1e-4 for base_lr in self.base_lrs]
        return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]


def find_z(model, loss_fn, inputs, targets, h):
    '''
    Finding the direction in the regularizer
    '''
    inputs.requires_grad_()
    outputs = model.eval()(inputs)
    loss_z = loss_fn(outputs, targets)
    #loss_z.backward(torch.ones(targets.size()).to(device))
    loss_z.backward()
    grad = inputs.grad.data + 0.0
    #norm_grad = grad.norm().item()
    z = torch.sign(grad).detach() + 0.
    z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)
    zero_gradients(inputs)
    model.zero_grad()

    return z#, norm_grad


def cure_loss(model, loss_fn, inputs, targets, h=3., lambda_=4, version=None):
    z = find_z(model, loss_fn, inputs, targets, h)

    inputs.requires_grad_()
    outputs_pos = model.eval()(inputs + z)
    outputs_orig = model.eval()(inputs)

    loss_pos = loss_fn(outputs_pos, targets)
    loss_orig = loss_fn(outputs_orig, targets)
    grad_diff = torch.autograd.grad(
            loss_pos-loss_orig, inputs,
            #grad_outputs=torch.ones(targets.size()).to(device),
            create_graph=True)[0]
    reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
    model.zero_grad()

    if version == "sum":
        return outputs_orig, loss_orig + torch.sum(lambda_ * reg)
    else:
        return outputs_orig, loss_orig + torch.sum(lambda_ * reg) / float(inputs.size(0))
