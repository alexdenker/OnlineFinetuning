

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

from src import UNetModel, VPSDE, BaseSampler, Euler_Maruyama_sde_predictor, score_based_loss_fn, flowers102

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

train_dataset = flowers102(root="dataset/flowers",split="train")
val_dataset = flowers102(root="dataset/flowers",split="val")
test_dataset = flowers102(root="dataset/flowers",split="test")

train_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

print("Number of images: ", len(train_dataset))
print("Number of images: ", len(val_dataset))
print("Number of images: ", len(test_dataset))


print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))

train_dl = DataLoader(train_dataset, batch_size=cfg_dict["training"]["batch_size"])

optimizer = torch.optim.Adam(model.parameters(), lr=cfg_dict["training"]["lr"])

log_dir = "model_weights/"

with open(os.path.join(log_dir, "report.yaml"), "w") as file:
    yaml.dump(cfg_dict, file)


for epoch in range(cfg_dict["training"]["num_epochs"]):
    print("Epoch: ", epoch+1)
    model.train()
    mean_loss = []
    for i, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
        optimizer.zero_grad() 
        x = batch[0].to("cuda")

        loss = score_based_loss_fn(x, model, sde)
        mean_loss.append(loss.item())
        loss.backward()

        optimizer.step()

    print("Mean loss: ", np.mean(mean_loss))
    model.eval() 

    torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))

    if epoch % 4 == 0 and epoch > 0:
        sampler = BaseSampler(
                score=model, 
                sde=sde,
                sampl_fn=Euler_Maruyama_sde_predictor,
                sample_kwargs={"batch_size": cfg_dict["sampling"]["batch_size"], 
                            "num_steps": cfg_dict["sampling"]["num_steps"],
                            "im_shape": [3,64,64],
                            "eps": cfg_dict["sampling"]["eps"] },
                device=x.device)

        x_mean = sampler.sample().cpu() # [-1,1]
        x_mean = (x_mean + 1. )/2.
        x_mean = torch.clamp(x_mean, 0, 1)

        save_image(x_mean, os.path.join(log_dir, f"sample_at_{epoch}.png"),nrow=4)
    #plt.figure()
    #plt.imshow(x_mean[0,0,:,:].cpu().numpy(), cmap="gray")
    #plt.show() 

    #img_grid = make_grid(x_mean, n_row=4)
    #print(x_mean.shape, img_grid.shape)
    #plt.figure()
    #plt.imshow(img_grid[0,:,:].numpy(), cmap="gray")
    #plt.show()