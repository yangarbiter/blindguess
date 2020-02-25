import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from ..base import TorchAttackModel

class FFTEPSAttackModel(TorchAttackModel):
    def __init__(self, model_fn, norm, eps, loss_fn=None,  perturb_iters=10,
                 step_size=0.1, batch_size=128, device=None):
        super(FFTEPSAttackModel, self).__init__(
            model_fn=model_fn, norm=norm, batch_size=batch_size, device=device)
        self.model_fn = model_fn
        if loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.perturb_iters = perturb_iters
        self.step_size = step_size
        self.norm = norm
        self.eps = eps

    def perturb(self, X, y=None, eps=None):
        """
        y: correct label
        """
        self.model_fn.eval()
        loader = self._get_loader(X, y)

        ret = []
        with torch.no_grad():
            for [x, y] in tqdm(loader, desc="Attacking (FFT EPS)"):
                x, y = x.to(self.device), y.to(self.device)
                _, advx = first_order_attack_fft(x, y, self.model_fn, self.loss_fn, self.perturb_iters,
                        self.step_size, self.eps, self.device)
                ret.append(advx.cpu().numpy())

        return np.concatenate(ret, axis=0).transpose(0, 2, 3, 1)

def first_order_attack_fft(x, y, model_fn, loss_fn, perturb_iters, step_size,
                           eps, device):
    freq = torch.rfft(x, signal_ndim=2, onesided=False)
    pert_v = (torch.ones_like(freq) + 0.01 * torch.randn_like(freq)).to(device)
    #optimizer = optim.SGD([pert_v], lr=step_size)

    for _ in range(perturb_iters):
        #optimizer.zero_grad()
        pert_v = pert_v.requires_grad_()
        with torch.enable_grad():
            advx = torch.irfft(freq * pert_v, signal_ndim=2, onesided=False).to(device)
            loss = loss_fn(model_fn(advx), y).mean()
        loss.backward(retain_graph=True)
        #optimizer.step()
        eta = step_size * pert_v.grad.data.sign().detach()
        pert_v  = pert_v.data.detach() + eta
        pert_v = torch.clamp(pert_v, 1-eps, 1+eps)

    pert_v = pert_v.requires_grad_(False)
    advx = torch.irfft(freq * pert_v, signal_ndim=2, onesided=False)
    loss = loss_fn(model_fn(advx), y).mean()
    #print((model_fn(advx).argmax(1) == y).float().mean())
    #print(torch.norm(x-advx, p=np.inf, dim=0).mean())

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
