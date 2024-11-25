"""
This file includes the likelihood used for the imaging experiments in the paper. 
The likelihoods are implemented as sub-classes of *Likelihood*. 

These likelihood are wrappers for the forward operators defined in utils/degredations.py, 
but provide useful new functionalities, i.e., sampling and log_likelihood_grad.s

This file includes:
- Painting (Out/Inpainting)
- Superresolution
- HDR (high dynamic range, non-linear operator)
- NonLinearBlur (corrected)
- PhaseRetrieval (non-linear)
- Radon (based on https://github.com/deepinv/deepinv/blob/main/deepinv/physics/functional/radon.py)
- Blur (based on https://github.com/deepinv/deepinv/blob/main/deepinv/physics/functional/convolution.py)

Note that for some non-linear forward operator, we do not use the mathematical likelihood gradient, but 
rather cheaper approximations to facilitate faster training/sampling.

"""


import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import functional as F

from .degredations import SuperResolution as SR



class Likelihood:
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        samples = []
        for i in range(len(x)):
            samples.append(self._sample(x[i : i + 1]))
        return torch.concatenate(samples, dim=0)

    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        samples = []
        for k in range(x.shape[0]):
            e = self._sample(x[[k]])
            samples.append(e)
        return torch.cat(samples, dim=0)

    def none_like(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def plot_condition(self, x, y, ax):
        raise NotImplementedError




def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n, c, h, 1, w, 1)
    out = out.view(n, c, scale * h, scale * w)
    return out


class Superresolution(Likelihood):
    def __init__(self, scale, sigma_y, device):
        self.scale = scale
        self.sigma_y = sigma_y
        self.device = device
        # scale = round(args.forward_op.scale)
        # self.AvgPool = torch.nn.AdaptiveAvgPool2d((256 // self.scale, 256 // self.scale))
        self.forward_op = SR(channels=3, img_dim=64, ratio=self.scale, device=device)

    def loss(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return None

    def log_likelihood_grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        res = self.A(x) - y

        if self.sigma_y == 0:
            return -self.A_adjoint(res)
        return -1 / self.sigma_y**2 * self.A_adjoint(res)

    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        y = self.A(x)

        y_noise = y + self.sigma_y * torch.randn_like(y)

        return y_noise

    def A(self, x):
        return self.forward_op.H(x).reshape(
            x.shape[0], 3, 64 // self.scale, 64 // self.scale
        )

    def A_adjoint(self, y):
        return self.forward_op.Ht(y).reshape(
            y.shape[0], 3, 64, 64
        )  # MeanUpsample(y, self.scale) / (self.scale**2)

