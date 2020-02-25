import torch
import torch.nn as nn
import torch.nn.functional as F

def local_lip(x, xp, x_outputs, xp_outputs, top_norm, btm_norm, reduction='mean'):
    down = torch.flatten(x - xp, start_dim=1)
    if top_norm == "kl":
        criterion_kl = nn.KLDivLoss(reduction='none')
        top = criterion_kl(F.log_softmax(xp_outputs, dim=1),
                           F.softmax(x_outputs, dim=1))
        ret = torch.sum(top, dim=1) / torch.norm(down + 1e-6, dim=1, p=btm_norm)
    else:
        top = torch.flatten(x_outputs, start_dim=1) - torch.flatten(xp_outputs, start_dim=1)
        ret = torch.norm(top, dim=1, p=top_norm) / torch.norm(down + 1e-6, dim=1, p=btm_norm)

    if reduction == 'mean':
        return torch.mean(ret)
    elif reduction == 'sum':
        return torch.sum(ret)
    else:
        raise ValueError(f"Not supported reduction: {reduction}")