from torch import optim
import torch.nn as nn

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
    if loss_name == 'ce':
        ret = nn.CrossEntropyLoss(reduction='mean')
    elif loss_name == 'mse':
        ret = nn.MSELoss()
    else:
        raise ValueError(f"Not supported loss {loss_name}")
    return ret