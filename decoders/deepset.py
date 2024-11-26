import torch
import torch.nn as nn

class DeepSet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepSet, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, normalize=False):
        phi_output = self.phi(x)        
        aggregated = phi_output.sum(dim=1)
        output = self.rho(aggregated)

        if normalize:
            output = torch.nn.functional.normalize(output, p=2, dim=1)

        return output