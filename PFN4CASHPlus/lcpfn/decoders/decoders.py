from torch import Tensor
import torch.nn.functional as F
import torch
from torch import nn


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)


def default_decoder(ninp, nhid, n_out, num_outputs=1):
    return  nn.Sequential(
    nn.Linear(ninp, nhid),
    GELU(),
    nn.Linear(nhid, num_outputs * n_out),
    nn.Unflatten(-1, (n_out, num_outputs)),
    )

class FeedForward(nn.Module):
    def __init__(self, ninp, nhid, n_out, num_outputs):
        super().__init__()
        self.linear0 = nn.Linear(ninp, nhid)  # First Linear Layer
        self.activation = nn.GELU()  # Activation Function
        self.linear1 = nn.Linear(nhid, 1 * n_out)  # Second Linear Layer
        self.linear2 = nn.Linear(nhid + 1 * n_out, 1 * n_out)  # Third
        self.unflatten = nn.Unflatten(-1, (n_out, 1))  # Reshape Output

    def forward(self, x):
        out0 = self.activation(self.linear0(x))  # Compute Out1
        out1 = self.linear1(out0) # Compute Out2
        feat = torch.cat([out0, self.activation(out1)], dim=-1)
        out_max = self.unflatten(self.linear2(feat))
        return torch.cat([out_max, self.unflatten(out1)], dim=-1)


def cascaded_decoder(ninp, nhid, n_out, num_outputs=2):
    return FeedForward(ninp, nhid, n_out, num_outputs)
