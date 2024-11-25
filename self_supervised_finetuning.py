

import torch
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import yaml 
import time
from tqdm import tqdm 

from src import UNetModel, VPSDE, ScaleModel, Superresolution, flowers102, finetuning_score_based_loss_fn

cfg = {
    'model': 
    { "in_channels": 6,
     "out_channels": 3,
     "model_channels": 32,
     "channel_mult": [1,2],
     "num_res_blocks": 1,
     "attention_resolutions": [],
     "max_period": 0.005},
    'batch_size': 64,
    'time_steps_min': 200,
    'time_steps_max': 200,
    'lr': 1e-4,
    'num_steps':  300,
    'log_img_freq': 10,
    'use_tweedie': True,
    'clip_gradient': True,
    'clip_value': 1.0, 
    "batch_size_val": 12,
    "init_scale": 5e-2,
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

print("Real noise level: ", torch.sum((lkhd.A(x_gt) - y_noise)**2, dim=(1,2,3))) # log p(y | x)

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
    
    def predict_noise(self, xt, ts):
        with torch.no_grad():
            s_pretrained = self.model(xt, ts)

            cond = torch.repeat_interleave(self.y_noise,  dim=0, repeats=x0.shape[0])

            marginal_prob_mean = self.sde.marginal_prob_mean_scale(ts)
            marginal_prob_std = self.sde.marginal_prob_std(ts)

            x0hat = (xt + marginal_prob_std[:,None,None,None]**2*s_pretrained)/marginal_prob_mean[:,None,None,None]
            log_grad = -lkhd.log_likelihood_grad(x0hat, cond) 

        log_grad_scaling = self.time_model(ts)[:, None, None]

        h_trans = self.cond_model(torch.cat([xt, log_grad], dim=1), ts) 

        h_trans = h_trans - log_grad_scaling*log_grad 

        return s_pretrained + h_trans

    def forward(self, ts, xT):
        """
        Implement EM solver

        """
        x_t = [xT] 
        kldiv_term1 = torch.zeros(1).to(xT.device)
        kldiv_term2 = torch.zeros(1).to(xT.device)

        for t0, t1 in zip(ts[:-1], ts[1:]):
            dt = t1 - t0 
            dW = torch.randn_like(xT) * torch.sqrt(dt.abs())
            ones_vec = torch.ones(xT.shape[0], device=xT.device)
            t = ones_vec * t0

            cond = torch.repeat_interleave(self.y_noise,  dim=0, repeats=x0.shape[0])
            with torch.no_grad():
                s_pretrained = self.model(x_t[-1], t)

                marginal_prob_mean = self.sde.marginal_prob_mean_scale(t)
                marginal_prob_std = self.sde.marginal_prob_std(t)

                marginal_prob_std_expanded = marginal_prob_std.view(*marginal_prob_std.shape, *(1,) * (s_pretrained.ndim - marginal_prob_std.ndim))
                marginal_prob_mean_expanded = marginal_prob_mean.view(*marginal_prob_mean.shape, *(1,) * (s_pretrained.ndim - marginal_prob_mean.ndim))

                x0hat = (x_t[-1] + marginal_prob_std_expanded**2*s_pretrained)/marginal_prob_mean_expanded
                
                log_grad = -lkhd.log_likelihood_grad(x0hat, cond) 

            log_grad_scaling = self.time_model(t)[:, None, None]

            h_trans = self.cond_model(torch.cat([x_t[-1], log_grad], dim=1), t) 
            h_trans = h_trans - log_grad_scaling*log_grad 
            s = s_pretrained + h_trans

            drift, diffusion = self.sde.sde(x_t[-1], t) # diffusion = sqrt(beta)
            diffusion_expanded = diffusion.view(*diffusion.shape, *(1,) * (s_pretrained.ndim - diffusion.ndim))
            f_t = drift - diffusion_expanded.pow(2)*s

            sum_axes = tuple(range(1, f_t.ndim))
            f_sq = (h_trans ** 2).sum(dim=sum_axes).unsqueeze(1)
            f_w = (h_trans * dW).sum(dim=sum_axes).unsqueeze(1)

            kldiv_term1 = kldiv_term1 + 0.5 * f_sq.detach() * diffusion[:,None].pow(2) * dt 
            kldiv_term2 = kldiv_term2 + diffusion[:,None] * f_w

            g_t = diffusion_expanded
            x_t.append(x_t[-1] + f_t * dt + g_t * dW)

        return x_t, kldiv_term1, kldiv_term2



batch_size = cfg["batch_size"]
sde_model = CondSDE(model=model, sde=sde, y_noise=y_noise)

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

plt.show()


for step in tqdm(range(cfg["num_steps"])):
    sde_model.cond_model.eval()
    
    x0 = torch.randn((batch_size, 3, 64, 64)).to("cuda")
    if cfg["time_steps_min"] == cfg["time_steps_max"]:
        t_size = cfg["time_steps_min"]
    else:
        t_size = np.random.randint(cfg["time_steps_min"], cfg["time_steps_max"])
    ts = np.linspace(np.sqrt(1e-3), 1., t_size)[::-1].copy()
    ts = torch.from_numpy(ts).to(device)**2
    with torch.no_grad():
        xt, kldiv_term1, kldiv_term2 = sde_model.forward(ts, x0)

    # here we compute the log likelihood
    cond = torch.repeat_interleave(y_noise,  dim=0, repeats=batch_size)
    data_fit = -torch.sum((lkhd.A(xt[-1]) - cond)**2, dim=(1,2,3)).unsqueeze(-1) # log p(y | x)
    loss_data = 1/2*1/noise_level**2*data_fit
    print("Negative log-likelihood || A x - y || = ", (-data_fit.mean()).item())

    # compute importance weights
    log_iw = -loss_data + kldiv_term1 + kldiv_term2

    log_c = torch.median(log_iw)
    compute_resampling_ratio = lambda lq, log_c: 1.0 - torch.clip(
            torch.exp(lq - log_c), 0, 1
        )

    resampling_ratio = compute_resampling_ratio(
        log_iw, log_c
            )
    resampling = torch.rand_like(resampling_ratio) <= resampling_ratio
    x0 = xt[-1][resampling.squeeze()]
    
    if step % 10 == 0 and step > 0:

        fig, axes = plt.subplots(1,8, figsize=(13,6))
        for idx, ax in enumerate(axes.ravel()):

            img_show = (x0[idx].permute(1,2,0).detach().cpu().numpy() + 1.)/2.
            ax.imshow(np.clip(img_show, 0,1))
            ax.set_title(f"Sample {idx}")
            ax.axis("off")

        plt.show()

    print(x0.shape)
    sde_model.cond_model.train()
    ## do a few steps with the supervised score matching loss 
    loss_train = []
    for train_iter in range(40):
        optimizer.zero_grad()

        loss = finetuning_score_based_loss_fn(x0, sde_model, sde)
        loss.backward()
        loss_train.append(loss.item())
        optimizer.step()

    scheduler.step()

    print("Mean loss during training: ", np.mean(loss_train))
