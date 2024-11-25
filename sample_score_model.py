

import numpy as np 
import matplotlib.pyplot as plt 
import os 
import yaml 

import torch 

from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from tqdm import tqdm 

from src import UNetModel, VPSDE, BaseSampler, Euler_Maruyama_sde_predictor, score_based_loss_fn

from torchvision.datasets import Flowers102
import torchvision.transforms as T
import torchvision.transforms.functional as F

def flowers102(root, split):
    # split = "train", "val", "test"
    image_size = 64
    transform = T.Compose(
        [
            T.Lambda(lambda img: F.center_crop(img, min(*img._size))),
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),  # normalize to [-1, 1]
        ]
    )
    dataset = Flowers102(root=root, split=split, transform=transform, download=True)
    return dataset


cfg_dict = { 
    "model":
    { "in_channels": 3,
     "out_channels": 3,
     "model_channels": 64,
     "channel_mult": [1,2,4],
     "num_res_blocks": 1,
     "attention_resolutions": [],
     "max_period": 0.005},
    "diffusion":
    {"sde": "VPSDE",
    "beta_min": 0.1,
    "beta_max": 20,
    },
    "training":
    {"num_epochs": 1000,
     "batch_size": 32,
     "lr": 5e-6},
    "sampling": 
    {"num_steps": 1000,
    "eps": 1e-5,
    "batch_size": 8}
}

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
model.load_state_dict(torch.load("model_weights/model.pt"))
model.to("cuda")


sampler = BaseSampler(
        score=model, 
        sde=sde,
        sampl_fn=Euler_Maruyama_sde_predictor,
        sample_kwargs={"batch_size": cfg_dict["sampling"]["batch_size"], 
                    "num_steps": cfg_dict["sampling"]["num_steps"],
                    "im_shape": [3,64,64],
                    "eps": cfg_dict["sampling"]["eps"] },
        device="cuda")

x_mean = sampler.sample().cpu() # [-1,1]
x_mean = (x_mean + 1. )/2.
x_mean = torch.clamp(x_mean, 0, 1)

fig, axes = plt.subplots(1, x_mean.shape[0])

for idx, ax in enumerate(axes.ravel()):
    ax.imshow(x_mean[idx].permute(1,2,0))
    ax.axis("off")

plt.show()