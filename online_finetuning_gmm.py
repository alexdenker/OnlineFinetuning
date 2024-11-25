

"""

Full backpropagation through the solver
"""

import torch
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import yaml 
import time
from tqdm import tqdm 

from src import ScoreModelMLP, VPSDE, finetuning_score_based_loss_fn, FourBlobs


cfg = {
    'model': 
    { "nfeatures": 4,
     "nfeatures_out": 2,
     "nblocks": 3,
     "nunits": 64},
    'batch_size': 512,
    'time_steps_min': 200,
    'time_steps_max': 600,
    'lr': 1e-3,
    'num_steps':  1000,
    'clip_gradient': True,
    'clip_value': 10.0, 
    'device': "cuda",
}

print(cfg)

device = cfg["device"]

base_path = "model_weights/FourBlobs"

with open(os.path.join(base_path, "report.yaml"), "r") as f:
    cfg_dict = yaml.safe_load(f)

sde = VPSDE(beta_min=cfg_dict["diffusion"]["beta_min"], 
            beta_max=cfg_dict["diffusion"]["beta_max"]
            )

model = ScoreModelMLP(
           marginal_prob_std=sde.marginal_prob_std,
            nfeatures=cfg_dict["model"]["nfeatures"],
            nblocks=cfg_dict["model"]["nblocks"],
            nunits=cfg_dict["model"]["nunits"])
model.load_state_dict(torch.load(os.path.join(base_path,"fourblobs_model.pt")))
model.to(device)
model.eval() 

target = FourBlobs()
samples_target = target.sample(2000)

def r(x):
    mean = torch.tensor([1.5, 1.5], device=x.device).unsqueeze(0)
    return -((x - mean)**2/(2 * 0.1**2)).sum(dim=1, keepdims=True)


class CondSDE(torch.nn.Module):
    def __init__(self, model, sde):
        super().__init__()

        self.model = model 
        self.sde = sde 
        self.cond_model = ScoreModelMLP(
            marginal_prob_std=sde.marginal_prob_std,
            nfeatures=cfg["model"]["nfeatures"],
            nfeatures_out=cfg["model"]["nfeatures_out"],
            nblocks=cfg["model"]["nblocks"],
            nunits=cfg["model"]["nunits"])
        nn.init.zeros_(self.cond_model.outblock.weight)
        nn.init.zeros_(self.cond_model.outblock.bias)

        self.cond_model.to(device)
        self.cond_model.train() 
        
    def forward(self, ts, xT):
        """
        Implement EM solver

        """
        x_t = [xT] 
        kldiv = torch.zeros(1).to(xT.device)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            dt = t1 - t0 
            dW = torch.randn_like(xT) * torch.sqrt(dt.abs())
            ones_vec = torch.ones(xT.shape[0], device=xT.device)
            t = ones_vec * t0
        
            s_pretrained = self.model(x_t[-1], t)

            with torch.no_grad():
                marginal_prob_mean = self.sde.marginal_prob_mean_scale(t)
                marginal_prob_std = self.sde.marginal_prob_std(t)

                marginal_prob_std_expanded = marginal_prob_std.view(*marginal_prob_std.shape, *(1,) * (s_pretrained.ndim - marginal_prob_std.ndim))
                marginal_prob_mean_expanded = marginal_prob_mean.view(*marginal_prob_mean.shape, *(1,) * (s_pretrained.ndim - marginal_prob_mean.ndim))

                x0hat = (x_t[-1] + marginal_prob_std_expanded**2*s_pretrained)/marginal_prob_mean_expanded
                
            h_trans = self.cond_model(torch.cat([x_t[-1], x0hat], dim=1), t) 

            s = s_pretrained + h_trans

            drift, diffusion = self.sde.sde(x_t[-1], t) # diffusion = sqrt(beta)
            diffusion_expanded = diffusion.view(*diffusion.shape, *(1,) * (s_pretrained.ndim - diffusion.ndim))
            f_t = drift - diffusion_expanded.pow(2)*s

            sum_axes = tuple(range(1, f_t.ndim))
            f_sq = (h_trans ** 2).sum(dim=sum_axes).unsqueeze(1)
            kldiv = kldiv + 0.5*dt.abs() * f_sq * diffusion_expanded.pow(2)

            g_t = diffusion_expanded
            x_t.append(x_t[-1] + f_t * dt + g_t * dW)

        return x_t, kldiv


batch_size = cfg["batch_size"]
sde_model = CondSDE(model=model, sde=sde)

optimizer = torch.optim.Adam(sde_model.cond_model.parameters(), lr=cfg["lr"])

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.975)


for i in tqdm(range(cfg["num_steps"])):
    sde_model.cond_model.train()

    optimizer.zero_grad()

    xT = torch.randn((batch_size, 2)).to(device)
    
    if cfg["time_steps_min"] == cfg["time_steps_max"]:
        t_size = cfg["time_steps_min"]
    else:
        t_size = np.random.randint(cfg["time_steps_min"], cfg["time_steps_max"])
    ts = np.linspace(np.sqrt(1e-3), 1., t_size)[::-1].copy()
    ts = torch.from_numpy(ts).to(device)**2

    xt, kldiv = sde_model.forward(ts, xT)

    loss_data = r(xt[-1])
    loss_kldiv = kldiv
    loss = (-loss_data + loss_kldiv).mean()
    print("loss: ", loss.item())


    loss.backward()
    if cfg['clip_gradient']:
        torch.nn.utils.clip_grad_norm_(sde_model.cond_model.parameters(), cfg['clip_value'])

    optimizer.step()

    scheduler.step()

    if i % 200 == 0 and i > 0:
        plt.figure()
        plt.scatter(samples_target.cpu().numpy()[:,0], samples_target.cpu().numpy()[:,1], label="prior samples")
        plt.scatter(xt[-1].detach().cpu().numpy()[:,0], xt[-1].detach().cpu().numpy()[:,1], label="fine-tuned samples")
        plt.legend()
        plt.show()
