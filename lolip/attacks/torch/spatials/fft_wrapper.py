
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_freq_shift_2d(f, a, b):
    bs, c, m, n, _ = f.shape
    #bs, m, n, c, _ = f.shape
    m1, m2 = torch.meshgrid(torch.arange(m), torch.arange(n))
    mesh = (m1*m2).float()
    
    re = torch.cos(2*np.pi* (a/m*m1 + b/n*m2))
    im = torch.sin(2*np.pi* (a/m*m1 + b/n*m2))
    
    return torch.cat((
        (f[:, :, :, :, 0] * re - f[:, :, :, :, 1] * im).unsqueeze(4),
        (f[:, :, :, :, 0] * im + f[:, :, :, :, 1] * re).unsqueeze(4)), axis=4)


class FFTShiftWrapper(nn.Module):
    def __init__(self, model):
        self.model = model

    def forward(self, x, a, b):
        if a is not None and b is not None:
            x = torch.fft(x, signal_ndim=2)
            x = torch_freq_shift_2d(x, a, b)
            x = torch.real(torch.ifft(x))
        return self.model(x)