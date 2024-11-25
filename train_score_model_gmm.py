

import numpy as np 
import os 
import yaml 
import torch 
import matplotlib.pyplot as plt 
from tqdm import tqdm 

from src import ScoreModelMLP, VPSDE, BaseSampler, Euler_Maruyama_sde_predictor, score_based_loss_fn, FourBlobs

device = "cuda"

cfg_dict = { 
    "model":
    { "nfeatures": 2,
     "nblocks": 3,
     "nunits": 64 },
    "diffusion":
    {"sde": "VPSDE",
    "beta_min": 0.1,
    "beta_max": 20,
    },
    "training":
    {"num_epochs": 1000,
     "num_steps": 10, 
     "batch_size": 256,
     "lr": 1e-2},
    "sampling": 
    {"num_steps": 1000,
    "eps": 1e-5,
    "batch_size": 512}
}

sde = VPSDE(beta_min=cfg_dict["diffusion"]["beta_min"], 
            beta_max=cfg_dict["diffusion"]["beta_max"]
            )

model = ScoreModelMLP(
            marginal_prob_std=sde.marginal_prob_std,
            nfeatures=cfg_dict["model"]["nfeatures"],
            nblocks=cfg_dict["model"]["nblocks"],
            nunits=cfg_dict["model"]["nunits"])
model.to(device)

target = FourBlobs(device=device)

print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))

optimizer = torch.optim.Adam(model.parameters(), lr=cfg_dict["training"]["lr"])

log_dir = "model_weights/FourBlobs/"

with open(os.path.join(log_dir, "report.yaml"), "w") as file:
    yaml.dump(cfg_dict, file)


for epoch in range(cfg_dict["training"]["num_epochs"]):
    print("Epoch: ", epoch+1)
    model.train()
    mean_loss = []
    
    for i in range(cfg_dict["training"]["num_steps"]):
 
        optimizer.zero_grad() 
        x = target.sample(cfg_dict["training"]["batch_size"]).to("cuda")

        loss = score_based_loss_fn(x, model, sde)
        mean_loss.append(loss.item())
        loss.backward()

        optimizer.step()

    print("Mean loss: ", np.mean(mean_loss))
    model.eval() 

    torch.save(model.state_dict(), os.path.join(log_dir, "fourblobs_model.pt"))

sampler = BaseSampler(
        score=model, 
        sde=sde,
        sampl_fn=Euler_Maruyama_sde_predictor,
        sample_kwargs={"batch_size": cfg_dict["sampling"]["batch_size"], 
                    "num_steps": cfg_dict["sampling"]["num_steps"],
                    "im_shape": [2],
                    "eps": cfg_dict["sampling"]["eps"] },
        device=x.device)

x_mean = sampler.sample().cpu() 

print("Sample: ", x_mean.shape)

plt.figure()
plt.scatter(x_mean[:,0].cpu().numpy(), x_mean[:,1].cpu().numpy())
plt.show()

