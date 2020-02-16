import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ...base import AttackModel

class SpatialAttackModel(AttackModel):
    def __init__(self, model_fn, rot_constraint, trans_constraint,
                 scale_constraint, loss_fn=None, batch_size=128, device=None):
        self.rot_constraint = rot_constraint
        self.scale_constraint = scale_constraint
        self.trans_constraint = trans_constraint
        self.model_fn = model_fn
        self.batch_size = batch_size
        if loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            self.loss_fn = loss_fn

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def _preprocess_x(self, X):
        return torch.from_numpy(X.transpose(0, 3, 1, 2)).float()

    def _get_loader(self, X, y):
        dataset = torch.utils.data.TensorDataset(
            self._preprocess_x(X), torch.from_numpy(y).long())
        loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=False, num_workers=1)
        return loader


class GridAttackModel(SpatialAttackModel):
    def __init__(self, model_fn, rot_constraint, trans_constraint,
                 scale_constraint, loss_fn=None, batch_size=128, device=None):
        super(GridAttackModel, self).__init__(
            model_fn, rot_constraint, trans_constraint, scale_constraint,
            loss_fn=loss_fn, batch_size=batch_size, device=device)

    def perturb(self, X, y=None, eps=None):
        """
        y: correct label
        """
        self.model_fn.eval()
        loader = self._get_loader(X, y)

        ret = []
        with torch.no_grad():
            for [x, y] in tqdm(loader, desc="Attacking (Grid)"):
                x, y = x.to(self.device), y.to(self.device)
                _, advx = spatial_attack(x, y, self.model_fn, self.loss_fn, self.rot_constraint,
                            self.trans_constraint, self.scale_constraint, self.device)
                ret.append(advx.cpu().numpy())

        return np.concatenate(ret, axis=0).transpose(0, 2, 3, 1)


def spatial_attack(x, y, model_fn, loss_fn, rot_constraint, trans_constraint, scale_constraint, device):
    trans_x = torch.linspace(-trans_constraint, trans_constraint, steps=5)
    trans_y = torch.linspace(-trans_constraint, trans_constraint, steps=5)

    current_loss = loss_fn(model_fn(x), y)
    current_x = x.clone()
    ori_x = x
    batch_size = len(x)
    for i in range(len(trans_x)):
        for j in range(len(trans_y)):
            matrix = torch.tensor([
                [1, 0, trans_x[i]],
                [0, 1, trans_y[j]]
            ]).float().repeat([batch_size, 1]).view(batch_size, 2, 3).to(device)
            
            #.repeat(len(x))
            grid = F.affine_grid(matrix, ori_x.size(), align_corners=False)
            x = F.grid_sample(ori_x, grid, align_corners=False)
            loss = loss_fn(model_fn(x), y)

            current_loss, mask = torch.max(torch.stack((current_loss, loss), dim=1), dim=1)
            mask = mask.bool()
            current_x[mask] = x[mask].clone()
    return current_loss, current_x
