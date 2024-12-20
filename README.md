# Online Fine-tuning of Diffusion Models

This repository implements the stochastic optimal control approach from [DEFT](https://arxiv.org/abs/2406.01781) for the fine-tuning of diffusion models.

## Inverse Problems

The goal in an inverse problem is to recover an image $x$ from indirect and noisy measurements $y$, where the image and measurements are related by a forward operator $A$. For an additive noise model we have
$$ y = A(x) + \eta, $$
with $\eta \sim \mathcal{N}(0, \sigma_\eta)$ as the measurement noise. Protoypical example include computed tomography or magnetic resoance imaging in medical imaging and super-resolution or inpainting in computer vision. In this repository we consider the problem of super-resolution. For this, let $x$ be the high-resolution image and $y$ the low-resolution image. We can model the forward operator $A$ as a downsampling operator. 

In the statistical framework, we are interesting in sampling from the posterior $p_\text{post}(y|x)$, i.e., the distribution of images $x$ given measurements $y$. Using Bayes theorem we can express the posterior using the data distribution (often also called prior distribution) and the likelihood, i.e.,

$$ p_\text{post}(y|x) = p_\text{data}(x) p_\text{lkhd}(y|x) /Z$$

In particular for the Gaussian noise model above, the negatiel log-likelihood can be written as 

$$ - \ln p_\text{lkhd}(y|x) = \frac{1}{2 \sigma_\eta^2} \| A(x) - y \|_2^2$$


## Conditioning Diffusion Models

We assume we have a forward SDE 

$$ d X_t = f_t(X_t) dt + \sigma_t dW_t, \quad X_0 \sim p_\text{data} := p_0 $$

and the corresponding (unconditional) reverse SDE

$$ d X_t = [f_t(X_t) - \sigma_t^2 s^\theta(X_t, t)]dt + \sigma_t dW_t, \quad X_T \sim p_T $$

where the score model is an approximation of the true score, i.e., $s^\theta(x_t, t) \approx \nabla{x_t} \log p_t(x_t)$. Sampling from the above reverse SDE recovers samples from the data distribution $p_\text{data}$.

Our goal is to condition the SDE such that we sample from the posterior distribution

$$ x \sim p_\text{post}(x|y)\propto p_\text{data}(x) p_\text{lkhd}(y|x) $$

given measurements $y$. Conditioning the SDE can be achieved by the h-transform, which results in a reverse SDE 

$$ d X_t = [f_t(X_t) - \sigma_t^2 (s^\theta(X_t, t) + h^\phi(X_t, t))]dt + \sigma_t dW_t, \quad X_T \sim p_T $$

where the goal is to train the additional control $h^\phi(x_t, t)$. Let the path measure of this conditional reverse SDE be defined as $\mathbb{Q}$. This can be achieved by minimising the following loss function

$$ \min_\phi  \mathbb{E}_\mathbb{Q}\left[\frac{1}{2} \int_0^T \sigma_t^2  \| h^{\phi}(x_t, t) \|_2^2  - \log p(y|x_0) \right]  $$

For the inverse problem above with the additive Gaussian noise model this reduces to 

$$\min_\phi \mathbb{E}_\mathbb{Q}\left[\frac{1}{2} \int_0^T \sigma_t^2 \| h^{\phi}(x_t, t) \|_2^2 + \frac{1}{2 \sigma _{\eta}^2} \| A(x_0) - y \|_2^2 \right]$$

Intuitively the two parts of the objective function can be understood as 
1. Training $h_\phi$ such that the final sample $x_0$ fits the measurements, i.e., $A(x_0) \approx y$
2. Regularisation of the norm of $h_\phi$ such that the reverse SDE is still close the unconditional reverse SDE.

### Scaling to higher dimension
Naive minisation of the stochastic optimal control objective is generally not possible, as the full trajectory $(x_t)_{t \in [0,T]} \sim \mathbb{Q}$ has to be kept in memory for backpropagation. 

We note that the objective can also be expressed as the KL divergence between the uncontrolled path measure $\mathbb{P}$ and the controlled path measure $\mathbb{Q}$. We can then use a log-variance loss to recover the gradients, i.e.,

$$\nabla_h D_\text{KL}(\mathbb{Q}|\mathbb{P}) =  \nabla_h \text{Var}_\mathbb{W}\left(\log \frac{d \mathbb{Q}}{d \mathbb{P}}\right) | _{\mathbb{W} = \mathbb{Q}}$$

where the RND can be expressed as 

$$\log \frac{d \mathbb{Q}}{d \mathbb{P}}(H_{0:T}) =  -\frac{1}{2} \int_0^T \sigma_t^2 \| h^\phi(H_t) \|_2^2 dt + \int_0^T \sigma_t^2 (g^\top h^\phi)(H_t) dt +\int_0^T \sigma_t h^{\phi^\top} (H_t)d W_t - \ln p _\text{lkhd}(y| H_0) $$

with $H_{0:T}$ from 

$$ d X_t = [f_t(X_t) - \sigma_t^2 (s_\theta(X_t, t) + g(X_t, t))]dt + \sigma_t dW_t, \quad X_T \sim p_T $$

in practice we choose $g_t = \text{stopgrad}(h_t)$. The log variance loss allows us to detach the gradients from the trajectory. 

## Example

We provide an example of fine-tuning an unconditional diffusion models for conditional sampling. The conditional sampling task is set to 4x super-resolution.

### Flower102

The weights for the pre-trained flowers model can be downloaded [here](https://drive.google.com/file/d/1jawOxXaToKEzoQJ3DA8uMdqNXmZIUC-Z/view?usp=sharing). The weights should be copied to `model_weights/model.pt`. We use the Flowers102 dataset, which can automatically be downloaded using `torchvision`. The original train/val/test split of Flowers102 results in 7169 training images, 1020 validation images and 6149 test images. We train the unconditional diffusion model on the joint train and test dataset and use the validation dataset of testing. This gives us a training dataset consisting of 13318 images.
