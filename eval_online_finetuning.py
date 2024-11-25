

"""
Implementation of the VarGrad loss. Change the direction of the SDE: going from 0 -> T
with x0 ~ N(0, I) and xT ~ data distribution
 
"""

import torch
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import yaml 
import time 
import math 

import torch 
import torch.nn as nn 

import wandb

from torchvision.utils import make_grid, save_image
from tqdm import tqdm 
from skimage.metrics import peak_signal_noise_ratio

from src import UNetModel, VPSDE, ScaleModel, Superresolution, flowers102

device = "cuda"

base_path = "model_weights"

with open(os.path.join(base_path, "report.yaml"), "r") as f:
    cfg_dict = yaml.safe_load(f)

cond_base_path = "cond_weights"
with open(os.path.join(cond_base_path, "report.yaml"), "r") as f:
    cfg = yaml.safe_load(f)

sde = VPSDE(beta_min=cfg_dict["diffusion"]["beta_min"], 
            beta_max=cfg_dict["diffusion"]["beta_max"]
            )

model = UNetModel(
           marginal_prob_std=sde.marginal_prob_std,
            model_channels=cfg_dict["model"]["model_channels"],
            max_period=cfg_dict["model"]["max_period"],
            num_res_blocks=cfg_dict["model"]["num_res_blocks"],
            in_channels=cfg_dict["model"]["in_channels"],
            out_channels=cfg_dict["model"]["out_channels"],
            attention_resolutions=cfg_dict["model"]["attention_resolutions"],
            channel_mult=cfg_dict["model"]["channel_mult"])
model.load_state_dict(torch.load(os.path.join(base_path,"model.pt")))
model.to(device)
model.eval() 




val_dataset = flowers102(root="dataset/flowers",split="val")

x_gt = val_dataset[cfg["val_img_idx"]][0].unsqueeze(0).to("cuda")


lkhd = Superresolution(scale=4, sigma_y=0.05, device="cuda")

torch.manual_seed(123)
y_noise = lkhd.sample(x_gt)

y_gt = lkhd.A(x_gt)
data_consistency_gt = torch.sum((y_gt - y_noise)**2)

Aty = lkhd.A_adjoint(y_noise)
noise_level = 0.05 

class CondSDE(torch.nn.Module):
    def __init__(self, model, sde, y_noise):
        super().__init__()

        self.model = model 
        self.sde = sde 
        self.cond_model = UNetModel(
            marginal_prob_std=sde.marginal_prob_std,
            model_channels=cfg["model"]["model_channels"],
            max_period=cfg["model"]["max_period"],
            num_res_blocks=cfg["model"]["num_res_blocks"],
            in_channels=cfg["model"]["in_channels"],
            out_channels=cfg["model"]["out_channels"],
            attention_resolutions=cfg["model"]["attention_resolutions"],
            channel_mult=cfg["model"]["channel_mult"])
        self.cond_model.load_state_dict(torch.load(os.path.join(cond_base_path,"cond_model.pt")))

        self.cond_model.to(device)
        self.cond_model.train() 

        self.time_model = ScaleModel(time_embedding_dim=cfg["time_embedding_dim"],
                                    max_period=cfg_dict["model"]["max_period"], 
                                    dim_out=1,
                                    init_scale=cfg["init_scale"])
        self.time_model.load_state_dict(torch.load(os.path.join(cond_base_path,"time_model.pt")))

        self.time_model.to(device)

        self.y_noise = y_noise

        self.k = torch.tensor([0.0], device=device, requires_grad=True)

    def forward(self, ts, x0):
        """
        Implement EM solver

        """
        x_t = torch.clone(x0)
      
        for t0, t1 in zip(ts[:-1], ts[1:]):
            dt = t1 - t0 
            dW = torch.randn_like(x0) * torch.sqrt(dt)
            ones_vec = torch.ones(x0.shape[0], device=x_gt.device)
            t = ones_vec * t0
            
            #print("Time step: ", t0, " to model: ", 1-t0)
            with torch.no_grad():
                s_pretrained = self.model(x_t, 1 - t)

            cond = torch.repeat_interleave(self.y_noise,  dim=0, repeats=x0.shape[0])

            with torch.no_grad():
                if cfg["use_tweedie"]:
                    marginal_prob_mean = self.sde.marginal_prob_mean_scale(1-t)
                    marginal_prob_std = self.sde.marginal_prob_std(1-t)

                    x0hat = (x_t+ marginal_prob_std[:,None,None,None]**2*s_pretrained)/marginal_prob_mean[:,None,None,None]
                    log_grad = -lkhd.log_likelihood_grad(x0hat, cond) #forward_op.trafo_adjoint(forward_op.trafo(x0hat) - cond)

                else:
                    log_grad = -lkhd.log_likelihood_grad(xt, cond) #forward_op.trafo_adjoint(forward_op.trafo(x_t[-1]) - cond)
            
            log_grad_scaling = self.time_model(1. - t)[:, None, None]
            #print(log_grad_scaling.shape) #.view(t.shape[0], 1, 28, 28)
            h_trans = self.cond_model(torch.cat([log_grad, x_t], dim=1), 1 - t) 
            h_trans = h_trans - log_grad_scaling*log_grad 

            s = s_pretrained + h_trans
            drift, diffusion = self.sde.sde(x_t, 1 - t) # diffusion = sqrt(beta)
            # drift = - 0.5 beta x
            f_t =  - drift + diffusion[:, None, None, None].pow(2)*s


            g_t = diffusion[:, None, None, None]
            x_t = x_t + f_t * dt + g_t * dW
            
        return x_t


sde_model = CondSDE(model=model, sde=sde, y_noise=y_noise)
sde_model.cond_model.eval()
sde_model.time_model.eval()
sde_model.eval()

t_size = 400

# look at a few samples 
from torchvision.utils import make_grid, save_image

batch_size = 8 
with torch.no_grad():
    ts_fine = np.sqrt(np.linspace(0, (1. - 1e-3)**2, t_size))
    ts_fine = torch.from_numpy(ts_fine).to(device)

    x0 = torch.randn((batch_size, 3, 64, 64)).to(device)

    xt = sde_model.forward(ts_fine, x0)

xt = (xt + 1.) / 2.
xt = torch.clamp(xt, 0, 1)

xt_grid = make_grid(xt, nrow=4)


save_image(xt_grid, "flower_samples.png")
save_image((x_gt[0] + 1.)/2., "flower_gt.png")
save_image((y_noise[0] + 1.)/2., "flower_y.png")


print(xt_grid.shape)

plt.figure()
plt.imshow(xt_grid.permute(1,2,0).cpu().numpy())
plt.axis("off")
plt.show()


num_samples = 1000
sampl_per_batch = 20 


sample_list = []

for i in tqdm(range(num_samples // sampl_per_batch)):

    with torch.no_grad():
        ts_fine = np.sqrt(np.linspace(0, (1. - 1e-3)**2, t_size))
        ts_fine = torch.from_numpy(ts_fine).to(device)

        x0 = torch.randn((sampl_per_batch, 3, 64, 64)).to(device)

        xt = sde_model.forward(ts_fine, x0)
        xt = torch.clamp(xt, -1, 1)
        sample_list.append(xt)


samples = torch.cat(sample_list)
print(samples.shape)



y_sim = lkhd.A(samples)

print(y_sim.shape, y_noise.shape)
data_consistency = torch.sum((y_sim - y_noise)**2, dim=(1,2,3))

print("data consistency: ", data_consistency.shape)

fig, (ax1) = plt.subplots(1,1, figsize=(16,4))


ax1.hist(data_consistency.cpu().numpy().ravel(), bins="auto", alpha=0.75)
ax1.set_title("|| A(x) - y_noise||")
ax1.vlines(data_consistency_gt.cpu().numpy(), 0, 10, label="|| A(x_gt) - y_noise||", colors='r')
ax1.legend()

plt.show()




mean_sample = torch.mean(samples, axis=0)


fig, (ax1, ax2) = plt.subplots(1,2)

im = ax1.imshow((x_gt[0].permute(1,2,0).cpu() + 1.)/2., cmap="gray")
ax1.axis("off")
ax1.set_title("Ground truth")

im = ax2.imshow((mean_sample.permute(1,2,0).cpu() + 1.)/2., cmap="gray")
ax2.axis("off")
ax2.set_title("Mean Sample")

plt.show()
