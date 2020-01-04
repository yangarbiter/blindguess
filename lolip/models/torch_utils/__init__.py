import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from .optimizer_nadam import Nadam

def get_optimizer(model, optimizer: str, learning_rate: float, momentum):
    if optimizer == 'nadam':
        ret = Nadam(model.parameters(), lr=learning_rate)
    elif optimizer == 'adam':
        ret = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        ret = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer == 'adagrad':
        ret = optim.Adagrad(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer == 'rms':
        ret = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise ValueError(f"Not supported optimizer {optimizer}")
    return ret


class CustomKLDivLoss(nn.KLDivLoss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CustomKLDivLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs, target):
        nb_classes = inputs.shape[1]
        y_onehot = torch.zeros_like(inputs) + 1e-6
        y_onehot = y_onehot.scatter_(1, target.unsqueeze(1), 1 - (1e-6*(nb_classes-1)))

        return F.kl_div(F.log_softmax(inputs, dim=1),
                        y_onehot, reduction=self.reduction)

def get_loss(loss_name: str, reduction='sum'):
    if 'ce' in loss_name:
        ret = nn.CrossEntropyLoss(reduction=reduction)
    elif 'mse' in loss_name:
        ret = nn.MSELoss(reduction=reduction)
    elif 'kld' in loss_name:
        ret = CustomKLDivLoss(reduction=reduction)
    else:
        raise ValueError(f"Not supported loss {loss_name}")
    return ret

def get_scheduler(optimizer, n_epochs: int):
    if n_epochs <= 60:
        scheduler = MultiStepLR(optimizer, milestones=[20, 40, 50], gamma=0.1)
    elif n_epochs <= 80:
        scheduler = MultiStepLR(optimizer, milestones=[30, 50, 70], gamma=0.1)
    elif n_epochs <= 120:
        scheduler = MultiStepLR(optimizer, milestones=[40, 80, 100], gamma=0.1)
    elif n_epochs <= 160:
        scheduler = MultiStepLR(optimizer, milestones=[40, 80, 120, 140], gamma=0.1)
    else:
        scheduler = MultiStepLR(optimizer, milestones=[60, 100, 140, 180], gamma=0.1)
    return scheduler