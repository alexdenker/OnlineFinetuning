from typing import Optional, Any, Dict, Tuple, Union

import torch
import numpy as np
import torch.nn as nn

from torch import Tensor

def Euler_Maruyama_sde_predictor(
    score,
    sde,
    x: Tensor,
    time_step: Tensor,
    step_size: float,
    nloglik = None, 
    penalty = None
    ) -> Tuple[Tensor, Tensor]:
    
    if nloglik is not None:
        x.requires_grad_()
        s = score(x, time_step)
    else:
        with torch.no_grad():
            s = score(x, time_step)

    if nloglik is not None:
        div = sde.marginal_prob_mean_scale(time_step)[:, None, None, None].pow(-1)
        std_t = sde.marginal_prob_std(time_step)[:, None, None, None]
        xhat0 = div*(x + std_t**2*s)

        loss = nloglik(xhat0)
        nloglik_grad = torch.autograd.grad(outputs=loss, inputs=x)[0]
        datafitscale = loss.pow(-1)

    drift, diffusion = sde.sde(x, time_step)
    diffusion_expanded = diffusion.view(*diffusion.shape, *(1,) * (s.ndim - diffusion.ndim))

    x_mean = x - (drift - diffusion_expanded.pow(2)*s)*step_size
    noise = torch.sqrt(diffusion_expanded.pow(2)*step_size)*torch.randn_like(x)

    x = x_mean + noise 

    if nloglik is not None:
        x = x - penalty*nloglik_grad*datafitscale

    return x.detach(), x_mean.detach()

