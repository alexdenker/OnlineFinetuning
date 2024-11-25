import torch 
#from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal

class FourBlobs():
    """Class of the Gaussian Mixture Model. 
    Attributes:
    ---------
   
    """

    def __init__(self, device: str = "cuda"):

        self.n_components = 4
        self.n_dim = 2 
        self.device = device 


        self.weights = torch.ones(self.n_components, device=self.device) / self.n_components

        self.means = torch.tensor([[1.5, 1.5],[1.5, -1.5],[-1.5, -1.5],[-1.5, 1.5]], device=self.device)         
        self.covs = torch.stack([torch.eye(self.n_dim, device=self.device) for _ in range(self.n_components)]) * 0.02


        categorical = Categorical(self.weights)
        gaussians = MultivariateNormal(self.means, self.covs)

        self.mixture = MixtureSameFamily(categorical, gaussians)
    

    def log_prob(self, x):

        logprob = self.mixture.log_prob(x)
        
        return logprob


    def sample(self, num_samples):

        samples = self.mixture.sample(sample_shape=(num_samples,))

        return samples

    #def pdf(self, x):
        """ Computes the probability density function at $x$ of the Gaussian Mixture Model.#

        Args:
        -----
        - `x`: torch.Tensor,
                Tensor of size `n_samples x n_dim`. Could be for example `x = np.stack([xx.ravel(), yy.ravel()]).T` with `xx, yy = np.meshgrid(np.linspace(xy_min_max[0],xy_min_max[1], 500), np.linspace(xy_min_max[0],xy_min_max[1], 500))`
        """
        #component_pdf = torch.stack([torch.exp(self.components[i].log_prob(x)) for i in range(self.n_components)]).T

        #weighted_compon_pdf = component_pdf * self.weights

        #return weighted_compon_pdf.sum(dim=1)

    
if __name__ == "__main__":
    import numpy as np 
    import matplotlib.pyplot as plt 

    n_dim = 2
    n_components = 40

    gmm = FourBlobs(device="cpu")

    samples = gmm.sample(1000)
    print(samples.shape)

    xx, yy = np.meshgrid(np.linspace(-3,3, 350), np.linspace(-3,3, 350))

    pos = np.stack([xx.ravel(), yy.ravel()])

    pos_in = torch.from_numpy(pos.T).to(gmm.device)
    print("POS IN: ", pos_in.shape)
    probs = gmm.log_prob(torch.from_numpy(pos.T).to(gmm.device))
    probs = torch.clamp(probs, -1000., None )
    print(probs)
    print(probs.shape)
    fig, ax = plt.subplots(1,1)

    im = ax.contourf(xx, yy, probs.cpu().numpy().reshape(xx.shape), levels=50)
    ax.scatter(samples[:,0], samples[:,1])
    fig.colorbar(im, ax=ax)
    plt.show()

    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, probs.cpu().numpy().reshape(xx.shape),
                       linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    """