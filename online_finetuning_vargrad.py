"""
Minimising the SOC objective with VarGrad estimator. 

See:
    Richter et al. "VarGrad: A Low-Variance Gradient Estimator for Variational Inference" (2020)
    https://arxiv.org/pdf/2010.10436

and in the context of diffusion models:
    Denker et al. "DEFT: Efficient Fine-Tuning of Diffusion Models by Learning the Generalised h-transform"
    https://arxiv.org/abs/2406.01781
    (Appendix G.3)
    
In particular Propostion 1 in Richter et al. shows that the 
gradient of the variance of the log-density is the gradient of the KL divergence.

"""

import torch
import numpy as np 
import os 
import yaml 
import time 
from tqdm import tqdm 
import os 

from src import UNetModel, VPSDE, ScaleModel, Superresolution, flowers102

cfg = {
    'model': 
    { "in_channels": 6,
     "out_channels": 3,
     "model_channels": 32,
     "channel_mult": [1,2],
     "num_res_blocks": 1,
     "attention_resolutions": [],
     "max_period": 0.005,
     "time_embedding_dim": 256,
     "init_scale": 7e-3},
    'batch_size': 14,
    'lr': 8e-5,
    'num_time_steps': 200,
    'num_steps':  1000,
    'log_img_freq': 10,
    'use_tweedie': True,
    'clip_gradient': True,
    'clip_value': 1.0, 
    "batch_size_val": 16,
    'val_img_idx': 3,
    'device': "cuda",
    "t_size_val": 800,
}

device = cfg["device"]

base_path = "model_weights"

# add time stamp to save directory
cur_time = time.strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("online_finetune_var_grad", cur_time)
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
            channel_mult=cfg_dict["model"]["channel_mult"])
model.load_state_dict(torch.load(os.path.join(base_path,"model.pt")))
model.to(device)
model.eval() 

if os.path.isfile(os.path.join(f"model_weights/flower_idx_{cfg['val_img_idx']}.pt")):
    x_gt = torch.load(os.path.join(f"model_weights/flower_idx_{cfg['val_img_idx']}.pt")).to(device)
else:
    print("Loading flowers dataset...")
    val_dataset = flowers102(root="dataset/flowers",split="val")
    x_gt = val_dataset[cfg["val_img_idx"]][0].unsqueeze(0).to(device)
    torch.save(x_gt, os.path.join(f"model_weights/flower_idx_{cfg['val_img_idx']}.pt"))


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
            channel_mult=cfg["model"]["channel_mult"])
        self.cond_model.to(device)
        self.cond_model.train() 

        self.time_model = ScaleModel(time_embedding_dim=cfg["model"]["time_embedding_dim"],
                                    max_period=cfg_dict["model"]["max_period"], 
                                    dim_out=1,
                                    init_scale=cfg["model"]["init_scale"])
        self.time_model.to(device)

        self.y_noise = y_noise
    
    def forward(self, ts, x0):
        x_t = torch.clone(x0)
        cond = torch.repeat_interleave(self.y_noise,  dim=0, repeats=x0.shape[0])

        kldiv_term1 = torch.zeros(x0.shape[0]).to(x0.device)
        kldiv_term2 = torch.zeros(x0.shape[0]).to(x0.device)
        kldiv_term3 = torch.zeros(x0.shape[0]).to(x0.device)

        t0s = np.random.choice(ts[:-1].cpu().numpy(), size=int(0.2*len(ts[:-1])), replace=False)
        
        # always add the last time steps
        t0s = np.concatenate([t0s, np.array([ts[-1].cpu().numpy()]), np.array([ts[-2].cpu().numpy()])])

        for t0, t1 in zip(ts[:-1], ts[1:]):
            dt = t1 - t0 

            dW = torch.randn_like(x0) * torch.sqrt(dt)
            ones_vec = torch.ones(x0.shape[0], device=x_gt.device)
            t = ones_vec * t0

            with torch.no_grad():
                s_pretrained = self.model(x_t, 1 - t)

            with torch.no_grad():
                if cfg["use_tweedie"]:
                    marginal_prob_mean = self.sde.marginal_prob_mean_scale(1-t)
                    marginal_prob_std = self.sde.marginal_prob_std(1-t)

                    x0hat = (x_t + marginal_prob_std[:,None,None,None]**2*s_pretrained)/marginal_prob_mean[:,None,None,None]
                    log_grad = -lkhd.log_likelihood_grad(x0hat, cond)

                else:
                    log_grad = -lkhd.log_likelihood_grad(x_t, cond) 
            
            log_grad_scaling = self.time_model(1. - t)[:, None, None]

            h_trans = self.cond_model(torch.cat([log_grad, x_t], dim=1), 1 - t) 
            h_trans = h_trans - log_grad_scaling*log_grad 

            s = s_pretrained + h_trans
            drift, diffusion = self.sde.sde(x_t, 1 - t) 
            # drift = - 0.5 beta x
            f_t =  - drift + diffusion[:, None, None, None].pow(2)*s

            f_sq = (h_trans ** 2).sum(dim=(1,2,3))
            g_f = (h_trans * h_trans.detach()).sum(dim=(1,2,3))
            f_w = (h_trans * dW).sum(dim=(1,2,3))
            
            """
            We do not backpropagate through all of the KL terms 
            to reduce the memory consumption. 

            Note: THIS IS NOT THEORETICALLY VALIDATED, but work well in practice.
            
            See:
            Venkatraman et al. "Amortizing intractable inference in diffusion models for vision, language, and control"
            https://arxiv.org/abs/2405.20971
            (Stochastic subsampling in Appendix H.1 for relative trajectory balance)
            
            """
            if not t0.item() in t0s:
                f_sq = f_sq.detach()
                g_f = g_f.detach()
                f_w = f_w.detach()

            kldiv_term1 = kldiv_term1 - 0.5 * f_sq * diffusion.pow(2) * dt
            kldiv_term2 = kldiv_term2 + diffusion.pow(2) * dt * g_f 
            kldiv_term3 = kldiv_term3 + diffusion * f_w 

            g_t = diffusion[:, None, None, None]
            x_t = x_t + f_t * dt + g_t * dW
            x_t = x_t.detach()
            
        return x_t, kldiv_term1, kldiv_term2, kldiv_term3



batch_size = cfg["batch_size"]
sde_model = CondSDE(model=model, sde=sde, y_noise=y_noise)

optimizer = torch.optim.Adam(list(sde_model.cond_model.parameters()) + list(sde_model.time_model.parameters()), lr=cfg["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["num_steps"], eta_min=cfg["lr"]/1000.)

cond = torch.repeat_interleave(y_noise,  dim=0, repeats=batch_size)

loss_threshold = 50000.0

min_soc_loss = 1e8
with tqdm(total=cfg["num_steps"]) as pbar:
    for step in range(cfg["num_steps"]):
        optimizer.zero_grad()

        x0 = torch.randn((batch_size, 3, 64, 64)).to(device)
        
        ts = np.sqrt(np.linspace(0, (1. - 1e-3)**2, cfg["num_time_steps"] ) )
        ts = torch.from_numpy(ts).to(device)
        
        xt, kldiv_term1, kldiv_term2, kldiv_term3 = sde_model.forward(ts, x0)
        
        data_fit = torch.sum((lkhd.A(xt) - cond)**2, dim=(1,2,3))
        loss_data = 1/2*1/noise_level**2*data_fit
        loss_kl = kldiv_term1 + kldiv_term2 + kldiv_term3
        loss = loss_data + kldiv_term1 + kldiv_term2 + kldiv_term3

        print(loss)

        # filter out to large loss values to stabilize training
        # this is a common trick when using VarGrad
        # and used (as far as i know) in all implementations
        loss_filtered = loss[loss < loss_threshold]
        
        print("Filtered loss size: ", loss_filtered.shape, len(loss_filtered))
        
        if len(loss_filtered) > int(0.3*batch_size):
            loss_filtered = torch.var(loss_filtered)
            loss_filtered.backward()

            with torch.no_grad():
                original_loss = (loss_data[loss < loss_threshold] - kldiv_term1[loss < loss_threshold]).mean()

            
            if cfg['clip_gradient']:
                torch.nn.utils.clip_grad_norm_(sde_model.cond_model.parameters(), cfg['clip_value'])
                torch.nn.utils.clip_grad_norm_(sde_model.time_model.parameters(), cfg['clip_value'])
            
            optimizer.step()
            scheduler.step()
            
            pbar.set_description(f"SOC loss: {original_loss.item()}, VarGrad loss: {loss_filtered.item()}")
        else:   
            original_loss = torch.tensor(float('inf'))
            print("Too many loss values were filtered out. Draw new samples")

        torch.cuda.empty_cache()
        
        # Delete tensors explicitly
        del x0, xt, kldiv_term1, kldiv_term2, kldiv_term3, data_fit, loss_data, loss_kl, loss, loss_filtered
            
        pbar.update(1)

        if original_loss.item() < min_soc_loss:
            min_soc_loss = original_loss.item()
            torch.save(sde_model.cond_model.state_dict(), os.path.join(save_dir, "cond_model.pt"))
            torch.save(sde_model.time_model.state_dict(), os.path.join(save_dir, "time_model.pt"))