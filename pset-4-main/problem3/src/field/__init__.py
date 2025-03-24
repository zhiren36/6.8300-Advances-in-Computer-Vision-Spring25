from omegaconf import DictConfig

from .field import Field
from .field_mlp import FieldMLP

FIELDS: dict[str, Field] = {
    "mlp": FieldMLP,
}


def get_field(
    cfg: DictConfig,
    d_coordinate: int,
    d_out: int,
) -> Field:
    return FIELDS[cfg.field.name](cfg.field, d_coordinate, d_out)
