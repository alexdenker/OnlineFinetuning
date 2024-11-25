



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

from src import Superresolution

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


val_dataset = flowers102(root="dataset/flowers",split="val")

x = val_dataset[0][0].unsqueeze(0).to("cuda")

print(x.shape)

lkhd = Superresolution(scale=4, sigma_y=0.00, device="cuda")

y = lkhd.sample(x)

Aty = lkhd.A_adjoint(y) 

print(y.shape)



print(y.shape)

x = (x + 1. )/2.
x = torch.clamp(x, 0, 1)[0]

y = (y + 1. )/2.
y = torch.clamp(y, 0, 1)[0]

Aty = (Aty + 1. )/2.
Aty = torch.clamp(Aty, 0, 1)[0]

print(x.mean(), y.mean(), Aty.mean())

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

ax1.imshow(x.permute(1,2,0).cpu().numpy(),vmin=0, vmax=1)

ax2.imshow(y.permute(1,2,0).cpu().numpy(),vmin=0, vmax=1)

ax3.imshow(Aty.permute(1,2,0).cpu().numpy(),vmin=0, vmax=1)


plt.show()



# test adjoint 

x = torch.rand_like(x).unsqueeze(0)
y = torch.rand_like(y).unsqueeze(0)

print(x.shape, y.shape)

Ax = lkhd.A(x)
Aty = lkhd.A_adjoint(y)

term1 = torch.dot(Ax.ravel(), y.ravel())
term2 = torch.dot(x.ravel(), Aty.ravel())

print(term1, term2, term1/term2)