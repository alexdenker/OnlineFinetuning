
import torch 


def score_based_loss_fn(x, model, sde, eps=1e-5):

    """
    The loss function for training score-based generative models.
    Args:
        model: A PyTorch model instance that represents a 
        time-dependent score-based model.
        x: A mini-batch of training data.
        sde: the forward sde
        eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
    z = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, random_t) # for VESDE the mean is just x

    std_expanded =  std.view(*std.shape, *(1,) * (z.ndim - std.ndim))

    perturbed_x = mean + z * std_expanded
    score  = model(perturbed_x, random_t)

    sum_axes = tuple(range(1, score.ndim))  # Exclude the first axis
    loss = torch.mean(torch.sum((score * std_expanded + z)**2, dim=sum_axes))
    
    return loss

def finetuning_score_based_loss_fn(x, model, sde, eps=1e-5):

    """
    The loss function for training score-based generative models.
    Args:
        model: A PyTorch model instance that represents a 
        time-dependent score-based model.
        x: A mini-batch of training data.
        sde: the forward sde
        eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
    z = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, random_t) # for VESDE the mean is just x

    std_expanded =  std.view(*std.shape, *(1,) * (z.ndim - std.ndim))

    perturbed_x = mean + z * std_expanded
    
    score = model.predict_noise(perturbed_x, random_t)

    sum_axes = tuple(range(1, score.ndim))  # Exclude the first axis
    loss = torch.mean(torch.sum((score * std_expanded + z)**2, dim=sum_axes))
    
    return loss