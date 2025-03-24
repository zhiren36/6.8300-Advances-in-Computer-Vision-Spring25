import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        is_first_layer: bool,
        omega_0: float,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights(is_first_layer, in_features)
        self.is_first_layer = is_first_layer

    @torch.no_grad()
    def init_weights(self, is_first_layer: bool, in_features: int):
        if is_first_layer:
            bound = 1 / in_features
            self.linear.weight.uniform_(-bound, bound)
        else:
            bound = torch.sqrt(torch.tensor(6 / in_features)) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #raise NotImplementedError("Not implemented!")

        return torch.sin(self.omega_0 * self.linear(x))