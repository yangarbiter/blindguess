from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import _Loss

from .....attacks.torch.fft_epsilon import first_order_attack_fft
from . import local_lip


def local_lip_loss(model_fn, base_loss_fn, x, y, norm, optimizer, perturb_type,
                top_norm, btm_norm, clip_min=None, clip_max=None, step_size=0.003,
                epsilon=0.031, perturb_steps=10, beta=1.0, device="gpu"):
    model_fn.eval()
    if perturb_type == "fft":
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        lip_loss_fn = lambda a, b: criterion_kl(F.log_softmax(a, dim=1), F.softmax(b, dim=1))

        _, adv_x = first_order_attack_fft(
            x, model_fn(x), model_fn, lip_loss_fn, perturb_iters=perturb_steps,
            step_size=step_size, eps=epsilon, device=device
        )

    if clip_min is not None and clip_max is not None:
        x_adv = torch.clamp(adv_x, clip_min, clip_max)

    optimizer.zero_grad()
    outputs = model_fn(x)
    loss_natural = base_loss_fn(outputs, y).mean()
    loss_local_lip = lip_loss_fn(model_fn(adv_x), outputs).mean()

    loss = loss_natural + beta * loss_local_lip
    return outputs, loss
