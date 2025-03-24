from jaxtyping import Float
from torch import Tensor
import torch


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""

    # raise NotImplementedError("This is your homework.")

    ones = torch.ones_like(points[..., :1])
    return torch.cat((points, ones), dim=-1)


def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""

    zeros = torch.zeros_like(points[..., :1])
    return torch.cat((points, zeros), dim=-1)

    # raise NotImplementedError("This is your homework.")


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""

    # raise NotImplementedError("This is your homework.")

    return torch.einsum("...ij, ...j->...i", transform, xyz)


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """

    # raise NotImplementedError("This is your homework.")

    # need to invert cam2world
    world2cam = torch.linalg.inv(cam2world)

    return torch.einsum("...ij, ...j->...i", world2cam, xyz)


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """

    # raise NotImplementedError("This is your homework.")

    return torch.einsum("...ij, ...j->...i", cam2world, xyz)


def project(
    xyz: Float[Tensor, "*#batch 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""

    ## raise NotImplementedError("This is your homework.")
    zeros = torch.zeros_like(intrinsics[..., :1])
    new_intrinsics = torch.cat((intrinsics, zeros), dim=-1)

    projection_homogeneous = torch.einsum("...ij, ...j ->...i", new_intrinsics, xyz)

    return projection_homogeneous[..., :2] / projection_homogeneous[..., 2:3]
