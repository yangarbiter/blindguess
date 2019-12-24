import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.nn.modules.loss import _Loss


class LocalLip(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(LocalLip, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return -F.mse_loss(input, target, reduction=self.reduction)

def local_lip(model, x, xp, top_norm, btm_norm):
    top = torch.flatten(model(x), start_dim=1) - torch.flatten(model(xp), start_dim=1)
    down = torch.flatten(x - xp, start_dim=1)
    return torch.mean(
        torch.norm(top, dim=1, p=top_norm) / torch.norm(down + 1e-6, dim=1, p=btm_norm))

def preprocess_x(x):
    return torch.from_numpy(x.transpose(0, 3, 1, 2)).float()

def estimate_local_lip(model, X, top_norm, btm_norm,
        batch_size=128, perturb_steps=10, step_size=0.003, epsilon=0.01):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = data_utils.TensorDataset(preprocess_x(X))
    loader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size, shuffle=False, num_workers=2)
    
    ret = []
    for [x] in loader:
        x = x.to(device)
        # generate adversarial example
        if btm_norm in [2, np.inf]:
            delta = 0.001 * torch.randn(x.shape).cuda().detach()
            delta = torch.autograd.Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer = optim.SGD([delta], lr=step_size)

            for _ in range(perturb_steps):
                x_adv = x + delta

                # optimize
                optimizer.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * local_lip(model, x, x_adv, top_norm, btm_norm)
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(len(x), -1).norm(p=btm_norm, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer.step()

                # projection
                delta.data.add_(x)
                delta.data.clamp_(0, 1).sub_(x)
                delta.data.renorm_(p=btm_norm, dim=0, maxnorm=epsilon)
            x_adv = x + delta
        else:
            raise ValueError(f"Unsupported norm {btm_norm}")
        ret.append(x_adv.detach().cpu().numpy().transpose(0, 2, 3, 1))
    return np.concatenate(ret, axis=0)
