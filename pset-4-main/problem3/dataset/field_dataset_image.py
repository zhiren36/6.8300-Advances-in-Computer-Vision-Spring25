import torch.nn.functional as F
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor
from torchvision.io import read_image

from .field_dataset import FieldDataset


class FieldDatasetImage(FieldDataset):
    def __init__(self, cfg: DictConfig) -> None:
        """Load the image in cfg.path into memory here."""

        super().__init__(cfg)
        self.image = read_image(cfg.path).float()

    def query(
        self,
        coordinates: Float[Tensor, "batch d_coordinate"],
    ) -> Float[Tensor, "batch d_out"]:
        """Sample the image at the specified coordinates and return the corresponding
        colors. Remember that the coordinates will be in the range [0, 1].

        You may find the grid_sample function from torch.nn.functional helpful here.
        Pay special attention to grid_sample's expected input range for the grid
        parameter.
        """

        return (
            F.grid_sample(
                self.image.unsqueeze(0),
                (coordinates * 2 - 1)[None][None],
                align_corners=True,
            )
            .transpose(0, 3)
            .squeeze(-1, -2)
            / 255
        )

    @property
    def d_coordinate(self) -> int:
        return 2

    @property
    def d_out(self) -> int:
        return 3

    @property
    def grid_size(self) -> tuple[int, ...]:
        """Return a grid size that corresponds to the image's shape."""

        return self.image.size()[-2:]
