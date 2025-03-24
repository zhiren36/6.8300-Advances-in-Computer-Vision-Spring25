from jaxtyping import Float
from torch import Tensor
from . import geometry
import torch


def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """

    batch_size = extrinsics.shape[0]
    rendered_images = []

    homo_vertices = geometry.homogenize_points(vertices)
    height, width = resolution
    for b in range(batch_size):
        cam2world = extrinsics[b]
        K = intrinsics[b]

        w2cam_vertices = geometry.transform_world2cam(homo_vertices, cam2world)

        project_vertices = geometry.project(w2cam_vertices, K)

        pixel_vertices = project_vertices * torch.tensor(
            [width - 1, height - 1],
            dtype=project_vertices.dtype,
            device=project_vertices.device,
        )

        #  Create a white canvas

        canvas = torch.ones((height, width), dtype=torch.float32)

        px = pixel_vertices[..., 0].round().long()
        py = pixel_vertices[..., 1].round().long()

        px = px.clamp(0, width - 1)
        py = py.clamp(0, height - 1)

        canvas[py, px] = 0.0

        rendered_images.append(canvas)

    return torch.stack(rendered_images, dim=0)
