import torch.nn as nn

from decoders.deepset import DeepSet

class EmbProjector(nn.Module):
    def __init__(self, emb_dim, hidden_dim, output_dim):
        super(EmbProjector, self).__init__()
        self.deepset = DeepSet(emb_dim, hidden_dim, output_dim)
    
    def forward(self, x, normalize=False):
        output = self.deepset(x, normalize=normalize)
        return output