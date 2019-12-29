from torch import optim
import torch.nn as nn
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

def get_loss(loss_name: str):
    if 'ce' in loss_name:
        ret = nn.CrossEntropyLoss(reduction='sum')
    elif 'mse' in loss_name:
        ret = nn.MSELoss(reduction='sum')
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