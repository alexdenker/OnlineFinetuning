

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

from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from tqdm import tqdm 
from skimage.metrics import peak_signal_noise_ratio

from src import UNetModel, VPSDE, ScaleModel, Superresolution, flowers102






cfg = {
    'model': 
    { "in_channels": 6,
     "out_channels": 3,
     "model_channels": 32,
     "channel_mult": [1,2],
     "num_res_blocks": 1,
     "attention_resolutions": [],
     "max_period": 0.005},
    'batch_size': 12,
    'time_steps_min': 200,
    'time_steps_max': 200,
    'lr': 1e-4,
    'num_steps':  300,
    'log_img_freq': 10,
    'use_tweedie': True,
    'clip_gradient': True,
    'clip_value': 1.0, 
    "batch_size_val": 12,
    "init_scale": 4e-3,
    "time_embedding_dim": 256,
    'val_img_idx': 0,
    'device': "cuda",
        "t_size_val": 800,

}

print(cfg)

device = cfg["device"]

base_path = "model_weights"

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




val_dataset = flowers102(root="dataset/flowers",split="val")

x_gt = val_dataset[cfg["val_img_idx"]][0].unsqueeze(0).to("cuda")


lkhd = Superresolution(scale=4, sigma_y=0.05, device="cuda")

y_noise = lkhd.sample(x_gt)

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
        self.cond_model.to("cuda")
        self.cond_model.train() 

        self.time_model = ScaleModel(time_embedding_dim=cfg["time_embedding_dim"],
                                    max_period=cfg_dict["model"]["max_period"], 
                                    dim_out=1,
                                    init_scale=cfg["init_scale"])
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

        t0s = np.random.choice(ts[:-1].cpu().numpy(), size=int(0.2*len(ts[:-1])), replace=False)
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


            f_sq = (h_trans ** 2).sum(dim=(1,2,3))
            g_f = (h_trans * h_trans.detach()).sum(dim=(1,2,3))
            f_w = (h_trans * dW).sum(dim=(1,2,3))

            if t0.item() in t0s:
                kldiv_term1 = kldiv_term1 - 0.5 * f_sq * diffusion.pow(2) * dt * 5 # only tak 20% of time steps
                kldiv_term2 = kldiv_term2 + diffusion.pow(2) * dt * g_f * 5
                kldiv_term3 = kldiv_term3 + diffusion * f_w

            g_t = diffusion[:, None, None, None]
            x_new = x_t[-1] + f_t * dt + g_t * dW
            x_t.append(x_new.detach())
            
        return x_t, kldiv_term1.detach(), kldiv_term2.detach(), kldiv_term3



wandb_kwargs = {
        "project": "online_finetuning",
        "entity": "alexanderdenker",
        "config": cfg,
        "name": "vargrad (flowers)",
        "mode": "online",  # "online", #"disabled", #"online" ,
        "settings": wandb.Settings(code_dir=""),
        "dir": "",
    }
with wandb.init(**wandb_kwargs) as run:

    batch_size = cfg["batch_size"]
    sde_model = CondSDE(model=model, sde=sde, y_noise=y_noise)
    #t_size = cfg["time_steps"]
    optimizer = torch.optim.Adam(list(sde_model.cond_model.parameters()) + list(sde_model.time_model.parameters()), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    x_target = x_gt.repeat(batch_size, 1, 1, 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14,6))

    ax1.set_title("Ground truth")
    ax1.imshow((x_gt[0].permute(1,2,0).cpu() + 1.)/2., cmap="gray")
    ax1.axis("off")

    ax2.set_title("y")
    ax2.imshow((y_noise[0].permute(1,2,0).cpu().numpy() + 1.)/2., cmap="gray")
    ax2.axis("off")

    ax3.set_title("Adjoint(y)")
    ax3.imshow((Aty[0].permute(1,2,0).cpu() +1.)/2., cmap="gray")
    ax3.axis("off")

    wandb.log({f"data": wandb.Image(plt)})
    plt.close()

    wandb.define_metric("custom_step")

    wandb.define_metric("val/mse_to_target_val", step_metric="custom_step")
    wandb.define_metric("val/psnr_of_mean_sample", step_metric="custom_step")
    wandb.define_metric("validation", step_metric="custom_step")
    wandb.define_metric("time_scaling", step_metric="custom_step")
    wandb.define_metric("samples", step_metric="custom_step")

    for step in tqdm(range(cfg["num_steps"])):
        sde_model.cond_model.train()
        optimizer.zero_grad()

        x0 = torch.randn((batch_size, 3, 64, 64)).to("cuda")
        if cfg["time_steps_min"] == cfg["time_steps_max"]:
            t_size = cfg["time_steps_min"]
        else:
            t_size = np.random.randint(cfg["time_steps_min"], cfg["time_steps_max"])
        ts = np.sqrt(np.linspace(0, (1. - 1e-3)**2, t_size))
        ts = torch.from_numpy(ts).to(device)
        time_start = time.time()
        
        xt, kldiv_term1, kldiv_term2, kldiv_term3 = sde_model.forward(ts, x0)


        cond = torch.repeat_interleave(y_noise,  dim=0, repeats=batch_size)
        data_fit = torch.sum((lkhd.A(xt[-1]) - cond)**2, dim=(1,2,3))
        loss_data = 1/2*1/noise_level**2*data_fit
        loss_kl = kldiv_term1 + kldiv_term2 + kldiv_term3
        loss = loss_data + kldiv_term1 + kldiv_term2 + kldiv_term3

        print(loss)
        # filter out to large loss values 
        loss_filtered = loss[loss < 50000]
        if len(loss_filtered) > int(0.3*batch_size):
            loss_filtered = torch.var(loss_filtered)
            print("var loss: ", loss_filtered)
            original_loss = (loss_data[loss < 50000] - kldiv_term1[loss < 50000]).mean()


            mse_target = torch.mean((xt[-1] - x_target)**2)
            psnrs = [] 
            for i in range(xt[-1].shape[0]):
                psnr = peak_signal_noise_ratio(x_target.detach().cpu().numpy()[i], xt[-1].detach().cpu().numpy()[i], data_range=2.)
                psnrs.append(psnr)
            print("mean PSNR (train): ", np.mean(psnrs))

            time_start = time.time() 
            loss_filtered.backward()
            #print("Calculate Gradient, adjoint SDE: ", time.time() - time_start, "s")
            if cfg['clip_gradient']:
                torch.nn.utils.clip_grad_norm_(sde_model.cond_model.parameters(), cfg['clip_value'])
                torch.nn.utils.clip_grad_norm_(sde_model.time_model.parameters(), cfg['clip_value'])
            optimizer.step()

            scheduler.step()
            wandb.log(
                        {"train/mean_psnr_of_samples": psnr,
                        "train/mse_to_target": mse_target.item(),
                        "train/vargrad_loss": loss_filtered.item(),
                        "train/original_finetuning_loss": original_loss.item(),
                        "train/loss_data_consistency": loss_data.mean().item(),
                        "train/loss_kldiv_term1": kldiv_term1.mean().item(),
                        "train/loss_kldiv_term1": kldiv_term1.mean().item(),
                        "train/loss_kldiv_term2": kldiv_term2.mean().item(),
                        "train/loss_kldiv_term3": kldiv_term3.mean().item(),
                        "train/loss_kldiv": loss_kl.mean().item(),
                        "train/learning_rate": float(scheduler.get_last_lr()[0]),
                        "step": step}
                    ) 
        else:
            print("Too many loss values were filtered out. Draw new samples")

        if step % cfg["log_img_freq"] == 0 and step > 0:
            
            val_log_dir = {}

            sde_model.cond_model.eval()

            ts_test = torch.linspace(1, 0, 100).to(device)

            scaling = sde_model.time_model(ts_test)
            plt.figure()
            plt.plot(scaling[:,0].detach().cpu().numpy())
            val_log_dir["time_scaling"] = wandb.Image(plt)
            #wandb.log({f"time_scaling": wandb.Image(plt)})
            plt.close()

            x_target_val = x_gt.repeat(cfg["batch_size_val"], 1, 1, 1)

            with torch.no_grad():
                ts_fine = np.sqrt(np.linspace(0, (1. - 1e-3)**2, cfg["t_size_val"]))
                ts_fine = torch.from_numpy(ts_fine).to(device)

                x0 = torch.randn((cfg["batch_size_val"], 3, 64, 64)).to(device)

                xt, _, _, _ = sde_model.forward(ts_fine, x0)

            mse_target = torch.mean((xt[-1] - x_target_val)**2)

            fig, axes = plt.subplots(1,8, figsize=(13,6))
            for idx, ax in enumerate(axes.ravel()):
                if idx == 0:
                    ax.imshow((x_gt[0].permute(1,2,0).cpu().numpy() + 1.)/2.)
                    ax.set_title("GT")
                elif idx == 1:
                    ax.imshow((y_noise[0].permute(1,2,0).cpu().numpy() + 1.)/2.)
                    ax.set_title("y (4x ds)")
                else:
                    ax.imshow((xt[-1][idx-1].permute(1,2,0).detach().cpu().numpy() + 1.)/2.)
                    ax.set_title(f"Sample {idx-1}")
                ax.axis("off")

            val_log_dir["samples"] = wandb.Image(plt)
            plt.close()
     
            psnrs = [] 
            for i in range(xt[-1].shape[0]):
                psnr = peak_signal_noise_ratio(x_target.detach().cpu().numpy()[i], xt[-1].detach().cpu().numpy()[i], data_range=2.)
                psnrs.append(psnr)
            print("mean PSNR (val): ", np.mean(psnrs))


            mean_sample = xt[-1].cpu().mean(dim=0)

            print(x_target_val.shape, mean_sample.shape)
            psnr = peak_signal_noise_ratio(x_target_val[0,:,:,:].cpu().numpy(), mean_sample[:,:,:].numpy(), data_range=2.)
            
            diff_to_mean = torch.mean((xt[-1].cpu() - mean_sample.unsqueeze(0))**2, dim=(1,2,3))
            diff_to_gt = torch.mean((x_target_val.cpu() - mean_sample)**2, dim=(1,2,3))
            fig, ax = plt.subplots(1,1)

            ax.hist(diff_to_mean.ravel().numpy(), bins="auto", alpha=0.75)
            ax.set_title("| x - x_mean|")
            ax.vlines(diff_to_gt.numpy(), 0, 30, label="| x_mean - x_gt|", colors='r')
            ax.legend()
            val_log_dir["diff"] = wandb.Image(plt)
            #wandb.log({f"samples": wandb.Image(plt)})
            plt.close() 

            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))

            im = ax1.imshow((x_target[0].permute(1,2,0).cpu() + 1.)/2., cmap="gray")
            ax1.axis("off")
            ax1.set_title("Ground truth")
            fig.colorbar(im, ax=ax1)

            im = ax2.imshow((mean_sample.permute(1,2,0).cpu() + 1.)/2., cmap="gray")
            ax2.axis("off")
            ax2.set_title("Mean Sample")
            fig.colorbar(im, ax=ax2)

            val_log_dir["validation"] = wandb.Image(plt)
            #wandb.log({f"validation": wandb.Image(plt)})
            plt.close()

            val_log_dir["val/psnr_of_mean_sample"] = psnr
            val_log_dir["val/mean_psnr_of_samples"] = np.mean(psnrs)
            val_log_dir["val/mse_to_target_val"] = mse_target.item()
            val_log_dir["custom_step"] = step
            wandb.log(val_log_dir) 
        