
import matplotlib.pyplot as plt 
import torch 


from src import Superresolution, flowers102

val_dataset = flowers102(root="dataset/flowers",split="val")

x = val_dataset[0][0].unsqueeze(0).to("cuda")

print(x.shape)

lkhd = Superresolution(scale=4, sigma_y=0.00, device="cuda")

y = lkhd.sample(x)

Aty = lkhd.A_adjoint(y) 

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