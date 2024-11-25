
import torch 
import torch.nn as nn 


class ScoreBlock(nn.Module):
    def __init__(self, nunits):
        super(ScoreBlock, self).__init__()
        self.linear = nn.Linear(nunits, nunits)
        
    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = nn.functional.relu(x)
        return x
        
    
class ScoreModelMLP(nn.Module):
    def __init__(self, marginal_prob_std, nfeatures: int = 2, nfeatures_out: int = 2, nblocks: int = 3, nunits: int = 64):
        super(ScoreModelMLP, self).__init__()
        
        self.inblock = nn.Linear(nfeatures+1, nunits)
        self.midblocks = nn.ModuleList([ScoreBlock(nunits) for _ in range(nblocks)])
        self.outblock = nn.Linear(nunits, nfeatures_out)

        self.marginal_prob_std = marginal_prob_std

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        val = torch.hstack([x, t.unsqueeze(-1)])  # Add t to inputs
        val = self.inblock(val)
        for midblock in self.midblocks:
            val = midblock(val)
        val = self.outblock(val) / self.marginal_prob_std(t)[:, None]
        return val