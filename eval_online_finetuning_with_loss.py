"""
Evaluation of the stochastic optimal control loss:
    L = E_{x_[0:T]}[ 1/2/sigma_y^2 || A(x_0) - y_noise ||^2  + int_0^T sigma(t)^2 || h(x_t, t) ||^2 dt] 


"""


import torch
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import yaml 


from torchvision.utils import make_grid, save_image
from tqdm import tqdm 

from src import UNetModel, VPSDE, ScaleModel, Superresolution, flowers102

device = "cuda"

base_path = "model_weights"

with open(os.path.join(base_path, "report.yaml"), "r") as f:
    cfg_dict = yaml.safe_load(f)

#cond_base_path = "cond_weights_var_grad"
#cond_base_path = "cond_weights_rtb"
#cond_base_path = "cond_weights_adjoint_sde"
cond_base_path = "cond_weights_dpok"


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

lkhd = Superresolution(scale=2, sigma_y=0.05, device="cuda")

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

        self.cond_model.to("cuda")
        self.cond_model.train() 

        self.time_model = ScaleModel(time_embedding_dim=cfg["time_embedding_dim"],
                                    max_period=cfg_dict["model"]["max_period"], 
                                    dim_out=1,
                                    init_scale=cfg["init_scale"])
        self.time_model.load_state_dict(torch.load(os.path.join(cond_base_path,"time_model.pt")))

        self.time_model.to(device)

        self.y_noise = y_noise
    
    def forward(self, ts, x0):
        """
        Implement EM solver

        """
        x_t = [x0] 
        #print("time steps: ", ts)
        kldiv_term1 = torch.zeros(1).to(x0.device)
        kldiv_term2 = torch.zeros(1).to(x0.device)
        kldiv_term3 = torch.zeros(1).to(x0.device)

        for t0, t1 in zip(ts[:-1], ts[1:]):
            #print(t0, t1)
            dt = t1 - t0 
            #print(dt)
            dW = torch.randn_like(x0) * torch.sqrt(dt)
            ones_vec = torch.ones(x0.shape[0], device=x_gt.device)
            t = ones_vec * t0
            #print("Time step: ", t0, " to model: ", 1-t0)
            with torch.no_grad():
                s_pretrained = self.model(x_t[-1], 1 - t)

            cond = torch.repeat_interleave(self.y_noise,  dim=0, repeats=x0.shape[0])

            with torch.no_grad():
                if cfg["use_tweedie"]:
                    marginal_prob_mean = self.sde.marginal_prob_mean_scale(1-t)
                    marginal_prob_std = self.sde.marginal_prob_std(1-t)

                    x0hat = (x_t[-1] + marginal_prob_std[:,None,None,None]**2*s_pretrained)/marginal_prob_mean[:,None,None,None]
                    log_grad = -lkhd.log_likelihood_grad(x0hat, cond) #forward_op.trafo_adjoint(forward_op.trafo(x0hat) - cond)

                else:
                    log_grad = -lkhd.log_likelihood_grad(xt[-1], cond) #forward_op.trafo_adjoint(forward_op.trafo(x_t[-1]) - cond)
            
            log_grad_scaling = self.time_model(1. - t)[:, None, None]
            #print(log_grad_scaling.shape) #.view(t.shape[0], 1, 28, 28)
            h_trans = self.cond_model(torch.cat([log_grad, x_t[-1]], dim=1), 1 - t) 
            h_trans = h_trans - log_grad_scaling*log_grad 



            s = s_pretrained + h_trans
            drift, diffusion = self.sde.sde(x_t[-1], 1 - t) # diffusion = sqrt(beta)
            # drift = - 0.5 beta x
            f_t =  - drift + diffusion[:, None, None, None].pow(2)*s

            #print("Base model norm: ", torch.norm(s_pretrained))
            #print("Cond model norm: ", torch.norm(h_trans))
            #print("Diffusion: ", diffusion.pow(2).shape)

            f_sq = (h_trans ** 2).sum(dim=(1,2,3))
            g_f = (h_trans * h_trans.detach()).sum(dim=(1,2,3))
            f_w = (h_trans * dW).sum(dim=(1,2,3))

            #print("f_sq: ", f_sq.shape, g_f.shape, f_w.shape)

            kldiv_term1 = kldiv_term1 - 0.5 * f_sq * diffusion.pow(2) * dt
            kldiv_term2 = kldiv_term2 + diffusion.pow(2) * dt * g_f 
            kldiv_term3 = kldiv_term3 + diffusion * f_w 

            g_t = diffusion[:, None, None, None]
            x_new = x_t[-1] + f_t * dt + g_t * dW
            x_t.append(x_new.detach())
            
        return x_t, kldiv_term1, kldiv_term2, kldiv_term3


sde_model = CondSDE(model=model, sde=sde, y_noise=y_noise)
sde_model.cond_model.eval()
sde_model.time_model.eval()
sde_model.eval()

t_size = 200

batch_size = 64
with torch.no_grad():
    ts_fine = np.sqrt(np.linspace(0, (1. - 1e-3)**2, t_size))

    ts_fine = torch.from_numpy(ts_fine).to(device)

    x0 = torch.randn((batch_size, 3, 64, 64)).to(device)

    xt, kldiv_term1, _, _ = sde_model.forward(ts_fine, x0)
    xt = xt[-1]

    data_fit = torch.sum((lkhd.A(xt) - y_noise)**2, dim=(1,2,3))
    loss_data = 1/2*1/noise_level**2*data_fit
    print(loss_data.shape, kldiv_term1.shape)
    print("loss_data: ", loss_data.mean().item())
    print("kldiv_term: ", kldiv_term1.mean().item())
    loss = (loss_data - kldiv_term1).mean()


print("SOC loss on samples: ", loss.item())

