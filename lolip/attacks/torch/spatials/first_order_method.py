import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from .grid import SpatialAttackModel

class FirstOrderAttackModel(SpatialAttackModel):
    def __init__(self, model_fn, rot_constraint, trans_constraint,
                 scale_constraint, perturb_iters=10, step_size=0.1, loss_fn=None, batch_size=128, device=None):
        super(FirstOrderAttackModel, self).__init__(
            model_fn, rot_constraint, trans_constraint, scale_constraint,
            loss_fn=loss_fn, batch_size=batch_size, device=device)
        self.perturb_iters = perturb_iters
        self.step_size = step_size

    def perturb(self, X, y=None, eps=None):
        """
        y: correct label
        """
        self.model_fn.eval()
        loader = self._get_loader(X, y)

        ret = []
        with torch.no_grad():
            for [x, y] in tqdm(loader, desc="Attacking (First Order)"):
                x, y = x.to(self.device), y.to(self.device)
                _, advx = first_order_attack(x, y, self.model_fn, self.loss_fn, self.perturb_iters,
                        self.step_size, self.rot_constraint, self.trans_constraint,
                        self.scale_constraint, self.device)
                ret.append(advx.cpu().numpy())

        return np.concatenate(ret, axis=0).transpose(0, 2, 3, 1)

def project_trans_constraint(x_v, y_v, trans_constraint):
    x_v = torch.clamp(x_v, -trans_constraint, trans_constraint)
    y_v = torch.clamp(y_v, -trans_constraint, trans_constraint)
    return x_v, y_v

def first_order_attack(x, y, model_fn, loss_fn, perturb_iters, step_size, rot_constraint, trans_constraint, scale_constraint, device):
    batch_size = len(x)
    #x_v = (torch.zeros(batch_size).float() + 0.01 * torch.randn(batch_size)).to(device)
    #y_v = (torch.zeros(batch_size).float() + 0.01 * torch.randn(batch_size)).to(device)
    final_matrix = torch.zeros((batch_size, 2, 3)).to(device)
    final_matrix[:, 0, 0] = 1
    final_matrix[:, 1, 1] = 1

    #optimizer = optim.SGD([x_v, y_v], lr=step_size)

    for i in range(batch_size):
        x_v = ((torch.zeros(1).float() + 0.05 * torch.randn(1)).to(device))
        y_v = ((torch.zeros(1).float() + 0.05 * torch.randn(1)).to(device))
        matrix = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
        ]).float().repeat([1, 1]).view(1, 2, 3).to(device)
        batch_x, batch_y = x[i: i+1], y[i: i+1]

        for _ in range(perturb_iters):
            x_v, y_v = x_v.requires_grad_(), y_v.requires_grad_()
            with torch.enable_grad():
                matrix[:, 0, 2] = x_v
                matrix[:, 1, 2] = y_v
                grid = F.affine_grid(matrix, batch_x.size(), align_corners=False)
                advx = F.grid_sample(batch_x, grid, align_corners=False)
                loss = (-1) * loss_fn(model_fn(advx), batch_y)

            loss.backward(retain_graph=True)
        
            x_v = x_v.detach() + step_size * torch.sign(x_v.grad.data)
            y_v = y_v.detach() + step_size * torch.sign(y_v.grad.data)
            x_v, y_v = project_trans_constraint(x_v, y_v, trans_constraint)

        final_matrix[i, 0, 2] = x_v
        final_matrix[i, 1, 2] = y_v

    grid = F.affine_grid(final_matrix, x.size(), align_corners=False)
    advx = F.grid_sample(x, grid, align_corners=False)
    print((model_fn(x).argmax(1) == y).float().mean())
    print((model_fn(advx).argmax(1) == y).float().mean())

    return loss, advx
