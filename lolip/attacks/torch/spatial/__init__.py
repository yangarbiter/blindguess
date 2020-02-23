
import numpy as np
import torch

from ...base import AttackModel
from .spatial_attack import AttackerModel

class SpatialAttackModel(AttackModel):

    def __init__(self, model_fn, loss_fn=None, batch_size=128, device=None):
        self.model_fn = model_fn
        self.batch_size = batch_size
        if loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            self.loss_fn = loss_fn

        self.attacker = AttackerModel(model_fn)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def _preprocess_x(self, X):
        return torch.from_numpy(X.transpose(0, 3, 1, 2)).float()

    def perturb(self, X, y=None, eps=None):
        """
        y: correct label
        """
        self.model_fn.eval()
        dataset = torch.utils.data.TensorDataset(
            self._preprocess_x(X), torch.from_numpy(y).long())
        loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=False, num_workers=1)

        grid_attack_args = {
            'spatial_constraint': '30',
            'tries': 1,
            'use_best': True,
            'attack_type': 'grid'
        }

        ret = []
        for [x, y] in loader:
            x, y = x.to(self.device), y.to(self.device)
            advx = self.attacker(x, y, make_adv=True, **grid_attack_args).cpu().numpy()
            ret.append(advx)

        return np.concatenate(ret, axis=0).transpose(0, 2, 3, 1)