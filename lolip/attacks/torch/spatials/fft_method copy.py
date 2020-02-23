import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from .grid import SpatialAttackModel
from .first_order_method import project_trans_constraint

class FFTAttackModel(SpatialAttackModel):
    def __init__(self, model_fn, rot_constraint, trans_constraint,
                 scale_constraint, perturb_iters=10, step_size=0.1, loss_fn=None, batch_size=128, device=None):
        super(FFTAttackModel, self).__init__(
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
            for [x, y] in tqdm(loader, desc="Attacking (FFT)"):
                x, y = x.to(self.device), y.to(self.device)
                _, advx = first_order_attack_fft(x, y, self.model_fn, self.loss_fn, self.perturb_iters,
                        self.step_size, self.rot_constraint, self.trans_constraint,
                        self.scale_constraint, self.device)
                ret.append(advx.cpu().numpy())

        return np.concatenate(ret, axis=0).transpose(0, 2, 3, 1)

def torch_freq_shift_2d(f, a, b, device):
    """
    a, b is number of pixels to shift
    """
    bs, c, m, n, _ = f.shape
    m1, m2 = torch.meshgrid(torch.arange(m), torch.arange(n))
    m1, m2 = m1.to(device), m2.to(device)
    
    a = (a.view(bs, 1, 1) / m / m1.unsqueeze(0).repeat(bs, 1, 1))
    b = (b.view(bs, 1, 1) / n / m2.unsqueeze(0).repeat(bs, 1, 1))
    re = torch.cos(2 * np.pi * (a + b))
    im = torch.sin(2 * np.pi * (a + b))
    
    return torch.stack((
        (f[:, :, :, :, 0] * re - f[:, :, :, :, 1] * im),
        (f[:, :, :, :, 0] * im + f[:, :, :, :, 1] * re)), axis=4)

def fft_shift(x, a, b, device):
    """
    a, b is (batch_size, 1) here
    """
    _, _, m, n = x.shape
    freq = torch.rfft(x, signal_ndim=2, onesided=False)
    import ipdb; ipdb.set_trace()
    shift_freq = torch_freq_shift_2d(freq, -a*m/2, -b*n/2, device)
    advx = torch.irfft(shift_freq, signal_ndim=2, onesided=False)
    return advx

def first_order_attack_fft(x, y, model_fn, loss_fn, perturb_iters, step_size,
                           rot_constraint, trans_constraint, scale_constraint, device):
    batch_size = len(x)
    x_v = (torch.zeros(batch_size).float() + 0.01 * torch.randn(batch_size)).to(device)
    y_v = (torch.zeros(batch_size).float() + 0.01 * torch.randn(batch_size)).to(device)
    final_matrix = torch.zeros((batch_size, 2, 3)).to(device)
    final_matrix[:, 0, 0] = 1
    final_matrix[:, 1, 1] = 1

    for _ in range(perturb_iters):
        #optimizer.zero_grad()
        x_v, y_v = x_v.requires_grad_(), y_v.requires_grad_()
        with torch.enable_grad():
            advx = fft_shift(x, x_v, y_v, device=device)
            loss = (-1) * loss_fn(model_fn(advx), y).sum()

        loss.backward(retain_graph=True)
        #optimizer.step()
    
        x_v = x_v.detach() + step_size * torch.sign(x_v.grad.data)
        y_v = y_v.detach() + step_size * torch.sign(y_v.grad.data)
        x_v, y_v = project_trans_constraint(x_v, y_v, trans_constraint)

    fft_advx = torch.tensor(advx, requires_grad=False)
    x_v, y_v = x_v.requires_grad_(False), y_v.requires_grad_(False)
    final_matrix[:, 0, 2] = x_v
    final_matrix[:, 1, 2] = y_v

    grid = F.affine_grid(final_matrix, x.size(), align_corners=False)
    advx = F.grid_sample(x, grid, align_corners=False)
    print((model_fn(x).argmax(1) == y).float().mean())
    print((model_fn(advx).argmax(1) == y).float().mean())
    print((model_fn(torch.cat(fft_advx)).argmax(1) == y).float().mean())
    loss = loss_fn(model_fn(advx), y).mean()

    return loss, advx


    #all_advx = []
    #optimizer = optim.SGD([x_v, y_v], lr=step_size)
    #for i in range(batch_size):
    #    x_v = (torch.zeros(1).float() + 0.05 * torch.randn(1)).to(device)
    #    y_v = (torch.zeros(1).float() + 0.05 * torch.randn(1)).to(device)
    #    batch_x, batch_y = x[i: i+1], y[i: i+1]
    #    #optimizer = optim.Adam([x_v, y_v], lr=0.5)

    #    for _ in range(perturb_iters):
    #        #optimizer.zero_grad()
    #        x_v, y_v = x_v.requires_grad_(), y_v.requires_grad_()
    #        with torch.enable_grad():
    #            advx = fft_shift(batch_x, x_v, y_v, device=device)
    #            loss = (-1) * loss_fn(model_fn(advx), batch_y)

    #        loss.backward(retain_graph=True)
    #        #optimizer.step()
    #    
    #        x_v = x_v.detach() + step_size * torch.sign(x_v.grad.data)
    #        y_v = y_v.detach() + step_size * torch.sign(y_v.grad.data)
    #        x_v, y_v = project_trans_constraint(x_v, y_v, trans_constraint)
    #    all_advx.append(advx.detach().clone())
    #    final_matrix[i, 0, 2] = x_v
    #    final_matrix[i, 1, 2] = y_v
