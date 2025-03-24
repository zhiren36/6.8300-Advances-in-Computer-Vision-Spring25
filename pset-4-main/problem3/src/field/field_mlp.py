import torch.nn as nn
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from src.components.positional_encoding import PositionalEncoding

from .field import Field


class FieldMLP(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up an MLP for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/mlp.yaml):

        - positional_encoding_octaves: The number of octaves in the positional encoding.
          If this parameter is None, do not positionally encode the input.
        - num_hidden_layers: The number of hidden linear layers.
        - d_hidden: The dimensionality of the hidden layers.

        Don't forget to add ReLU between your linear layers!
        """

        super().__init__(cfg, d_coordinate, d_out)

        # Create hidden layers
        layers: list[nn.Module] = []
        # Set up positional encoding if specified
        self.positional_encoding = None
        if cfg.positional_encoding_octaves is not None:
            self.positional_encoding = PositionalEncoding(
                cfg.positional_encoding_octaves
            )
            d_coordinate = self.positional_encoding.d_out(d_coordinate)
            layers.append(self.positional_encoding)

        for i in range(cfg.num_hidden_layers):
            layers.append(
                nn.Linear(d_coordinate if i == 0 else cfg.d_hidden, cfg.d_hidden)
            )
            layers.append(nn.ReLU())

        out_layer = nn.Linear(cfg.d_hidden, d_out)
        layers.append(out_layer)
        self.layers = nn.Sequential(*layers)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the MLP at the specified coordinates."""

        return self.layers(coordinates)
