"""
Implementation of online finetuning using the naive approach:

We backpropagate through the entire SDE trajectory to compute gradients.
Note, that we have to use:
    - gradient checkpointing 
    - gradient accumulation 
to fit the model in GPU memory.

This makes the training very slow, I get about 1 iteration per 30s. 

"""

import torch
import numpy as np 
import os 
import yaml 
from tqdm import tqdm 
import time 

from src import UNetModel, VPSDE, ScaleModel, Superresolution, flowers102


cfg = {
    'model': # these are parameters for the h-transform network and time-scaling network
    { "in_channels": 6,
     "out_channels": 3,
     "model_channels": 32,
     "channel_mult": [1,2],
     "num_res_blocks": 1,
     "attention_resolutions": [],
     "max_period": 0.005,
     "time_embedding_dim": 256,
     "init_scale": 5e-3},
    'batch_size': 2, 
    "grad_accum_steps": 8,
    "num_time_steps": 200,
    'lr': 1e-5,
    'num_steps':  1000,
    'use_tweedie': True,
    'clip_gradient': True,
    'clip_value': 1.0,     
    'val_img_idx': 3, 
    'device': "cuda",
}

device = cfg["device"]

base_path = "model_weights"

cond_pretrain_path = "cond_weights_rtb"

# add time stamp to save directory
cur_time = time.strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("online_finetune_naive", cur_time)
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "report.yaml"), "w") as file:
    yaml.dump(cfg, file)


with open(os.path.join(base_path, "report.yaml"), "r") as f:
    cfg_dict = yaml.safe_load(f)


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
            channel_mult=cfg_dict["model"]["channel_mult"],
            use_checkpoint=True)
model.load_state_dict(torch.load(os.path.join(base_path,"model.pt")))
model.to(device)
model.eval() 

val_dataset = flowers102(root="dataset/flowers",split="val")

x_gt = val_dataset[cfg["val_img_idx"]][0].unsqueeze(0).to(device)

noise_level = 0.05
lkhd = Superresolution(scale=2, sigma_y=noise_level, device=device)

torch.manual_seed(123)
y_noise = lkhd.sample(x_gt)


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
            channel_mult=cfg["model"]["channel_mult"],
            use_checkpoint=True
        )
        if cond_pretrain_path is not None:
            self.cond_model.load_state_dict(torch.load(os.path.join(cond_pretrain_path,"cond_model.pt")))
        self.cond_model.to("cuda")
        self.cond_model.train() 

        self.time_model = ScaleModel(time_embedding_dim=cfg["model"]["time_embedding_dim"],
                                    max_period=cfg_dict["model"]["max_period"], 
                                    dim_out=1,
                                    init_scale=cfg["model"]["init_scale"])
        if cond_pretrain_path is not None:
            self.time_model.load_state_dict(torch.load(os.path.join(cond_pretrain_path,"time_model.pt")))
        self.time_model.to(device)

        self.y_noise = y_noise
    
    def h_transform(self, x_t, t, score_pretrained, cond):
        """
        This implement our gradient-informed parametrisation of the h-transform, given has 
            NN(x_t, t) = NN1(x_t, t) - scaling(t) * grad_x0hat log p(y|x0hat(x_t))
        
        where x0hat(x_t) is the Tweedie estimate of x0 given x_t and the pretrained score model

        In particular, we treat "grad_x0hat log p(y|x0hat(x_t))" as a constant, 
        detached from the computational graph.
        """

        if cfg["use_tweedie"]:
            marginal_prob_mean = self.sde.marginal_prob_mean_scale(1-t)
            marginal_prob_std = self.sde.marginal_prob_std(1-t)

            x0hat = (x_t + marginal_prob_std[:,None,None,None]**2*score_pretrained)/marginal_prob_mean[:,None,None,None]
            log_grad = -lkhd.log_likelihood_grad(x0hat, cond).detach()

        else:
            log_grad = -lkhd.log_likelihood_grad(x_t, cond).detach()

        log_grad_scaling = self.time_model(1. - t)[:, None, None]
        h_trans = self.cond_model(torch.cat([log_grad, x_t], dim=1), 1 - t) 
        h_trans = h_trans - log_grad_scaling*log_grad

        return h_trans 
    
    def forward(self, ts, x0):
        """
        Implement EM solver. 
        We have the forward SDE as:
            dx = f(x,t)dt + g(t)dW_t
        and thus the reverse-time SDE is:
            dx = [f(x,t) - g(t)^2 * s(x,t) ] dt + g(t)dW_t
        where s(x,t) is the score function and the time runs from [T,0].
        
        However, we implement the reverse SDE in forward time, i.e., from t=0 to t=T, so we have:
            dx = [-f(x,1-t) + g(1-t)^2 * s(x,1-t) ] dt + g(1-t)dW_t
        
        Leading to the Euler-Maruyama update as:
            x_{t+dt} = x_t + [-f(x,1-t) + g(1-t)^2 * s(x,1-t) ] dt + g(1-t) * dt.sqrt() * eps
            with eps ~ N(0,I)

        The KL divergence term is computed as:
            KL = E[ int_0^T 0.5 * g(1-t)^2 * ||  h(x,1-t) ||^2 dt ]
        where h(x,1-t) is the h-transform.

        """

        x_t = torch.clone(x0)

        kldiv_term = torch.zeros(x0.shape[0]).to(x0.device)        
        
        cond = torch.repeat_interleave(self.y_noise, dim=0, repeats=x0.shape[0])

        for t0, t1 in zip(ts[:-1], ts[1:]):
            dt = t1 - t0 

            dW = torch.randn_like(x0) * torch.sqrt(dt)
            ones_vec = torch.ones(x0.shape[0], device=x_gt.device)
            t = ones_vec * t0
            
            s_pretrained = self.model(x_t, 1 - t)

            h_trans = self.h_transform(x_t, t, s_pretrained, cond)

            s = s_pretrained + h_trans
            drift, diffusion = self.sde.sde(x_t, 1 - t) # diffusion = sqrt(beta)

            f_t =  - drift + diffusion[:, None, None, None].pow(2)*s

            f_sq = (h_trans ** 2).sum(dim=(1,2,3))

            kldiv_term = kldiv_term - 0.5 * f_sq * diffusion.pow(2) * dt

            g_t = diffusion[:, None, None, None]
            x_t = x_t + f_t * dt + g_t * dW

        return x_t, kldiv_term


batch_size = cfg["batch_size"]
sde_model = CondSDE(model=model, sde=sde, y_noise=y_noise)

#t_size = cfg["time_steps"]
optimizer = torch.optim.Adam(list(sde_model.cond_model.parameters()) + list(sde_model.time_model.parameters()), lr=cfg["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["num_steps"], eta_min=cfg["lr"]/1000.)

sde_model.cond_model.train()

cond = torch.repeat_interleave(y_noise,  dim=0, repeats=batch_size)

with tqdm(total=cfg["num_steps"]) as pbar:
    for step in range(cfg["num_steps"]):
        optimizer.zero_grad()

        for _ in range(cfg["grad_accum_steps"]):

            x0 = torch.randn((batch_size, 3, 64, 64), device=device)
            
            
            ts = np.sqrt(np.linspace(0, (1. - 1e-3)**2, cfg["num_time_steps"] ) )
            ts = torch.from_numpy(ts).to(device)

            xt, kldiv_term = sde_model.forward(ts, x0)

            data_fit = torch.sum((lkhd.A(xt) - cond)**2, dim=(1,2,3))
            loss_data = 0.5*1/noise_level**2*data_fit

            soc_loss = (loss_data - kldiv_term).mean() / cfg["grad_accum_steps"]
            soc_loss.backward()

        if cfg['clip_gradient']:
            torch.nn.utils.clip_grad_norm_(sde_model.cond_model.parameters(), cfg['clip_value'])
            torch.nn.utils.clip_grad_norm_(sde_model.time_model.parameters(), cfg['clip_value'])
            
        optimizer.step()
        scheduler.step()
        
        pbar.set_description(f"SOC loss: {soc_loss.item()*cfg["grad_accum_steps"]}")
        pbar.update(1)

        
        torch.save(sde_model.cond_model.state_dict(), os.path.join(save_dir, "cond_model.pt"))
        torch.save(sde_model.time_model.state_dict(), os.path.join(save_dir, "time_model.pt"))