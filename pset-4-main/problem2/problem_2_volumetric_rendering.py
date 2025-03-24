import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class SimpleImplicitModel(nn.Module):
    """
    An implicit model that contains multiple geometric shapes inside the unit cube:
    - A blue sphere
    - A red cube
    - A green sphere
    - A yellow cube
    The density inside the shapes is 100.0, and the density outside is 0.0.
    """

    def __init__(self):
        super().__init__()
        pass

    def forward(self, points_xyz):
        """
        points_xyz: [B, 3]
        Returns:
            density: [B, 1] in [0, 1]
            color:   [B, 3] in [0, 1]
        """
        device = points_xyz.device
        batch_size = points_xyz.shape[0]

        # Initialize density and color tensors
        density = torch.zeros(batch_size, 1, device=device)
        color = torch.zeros(batch_size, 3, device=device)

        # 1. Blue sphere at center (0.5, 0.5, 0.5) with radius 0.2
        sphere1_center = torch.tensor([0.5, 0.5, 0.5], device=device)
        sphere1_radius = 0.2
        sphere1_dist = (
            torch.norm(points_xyz - sphere1_center, dim=1, keepdim=True)
            - sphere1_radius
        )
        sphere1_density = (sphere1_dist <= 0).float()
        sphere1_color = torch.tensor([0.1, 0.4, 0.8], device=device)  # Soft blue

        # 2. Red cube with bounds [0.3, 0.4] x [0.3, 0.4] x [0.3, 0.7]
        cube1_min = torch.tensor([0.3, 0.3, 0.3], device=device)
        cube1_max = torch.tensor([0.4, 0.4, 0.7], device=device)
        inside_cube1 = torch.all(
            (points_xyz >= cube1_min) & (points_xyz <= cube1_max), dim=1, keepdim=True
        ).float()
        cube1_color = torch.tensor([0.9, 0.2, 0.2], device=device)  # Soft red

        # 3. Green sphere at center (0.2, 0.7, 0.3) with radius 0.2
        sphere2_center = torch.tensor([0.2, 0.7, 0.3], device=device)
        sphere2_radius = 0.2
        sphere2_dist = (
            torch.norm(points_xyz - sphere2_center, dim=1, keepdim=True)
            - sphere2_radius
        )
        sphere2_density = (sphere2_dist <= 0).float()
        sphere2_color = torch.tensor([0.2, 0.7, 0.3], device=device)  # Soft green

        # 4. Yellow cube with bounds
        cube2_min = torch.tensor([0.6, 0.1, 0.1], device=device)
        cube2_max = torch.tensor([0.8, 0.3, 0.3], device=device)
        inside_cube2 = torch.all(
            (points_xyz >= cube2_min) & (points_xyz <= cube2_max), dim=1, keepdim=True
        ).float()
        cube2_color = torch.tensor([0.9, 0.8, 0.2], device=device)  # Soft yellow

        # Calculate combined density and color
        # For each point, find which object it belongs to (if any)
        sphere1_mask = sphere1_density > 0
        cube1_mask = inside_cube1 > 0
        sphere2_mask = sphere2_density > 0
        cube2_mask = inside_cube2 > 0

        # Update density: combine all object densities
        density = (
            torch.maximum(
                torch.maximum(sphere1_density, inside_cube1),
                torch.maximum(sphere2_density, inside_cube2),
            )
            * 100.0
        )

        # Apply colors based on which object the point belongs to
        # We process in reverse order to handle overlapping objects
        # (later objects overwrite earlier ones)
        color_mask = torch.zeros_like(sphere1_mask).bool()

        # Apply colors for each object
        for mask, obj_color in [
            (sphere1_mask, sphere1_color),
            (cube1_mask, cube1_color),
            (sphere2_mask, sphere2_color),
            (cube2_mask, cube2_color),
        ]:
            # Only update colors for points that belong to this object and haven't been colored yet
            update_mask = mask & ~color_mask
            if update_mask.any():
                for i in range(3):
                    color[update_mask.squeeze(), i] = obj_color[i]
                color_mask = color_mask | update_mask

        return density, color


############################
# Part A: Camera Rays
############################


def camera_param_to_rays(c2w, intrinsics, H=128, W=128):
    """
    Given the camera parameters, generate rays for each pixel.

    Args:
        c2w: [4,4] camera-to-world transform matrix
        intrinsics: [fx, fy, cx, cy] camera intrinsic parameters
        H: Height of the image
        W: Width of the image

    Returns:
        ray_origins: [H, W, 3] origin points for rays
        ray_directions: [H, W, 3] direction vectors for rays
    """
    # TODO: Implement this function
    # 1. Create pixel coordinates using meshgrid
    # 2. Convert pixel coordinates to camera coordinates using intrinsics
    # 3. Transform rays from camera space to world space using c2w
    # 4. Return ray origins and directions of shape [H, W, 3]

    device = c2w.device

    # Ensure intrinsics is a torch tensor on the correct device.
    if not torch.is_tensor(intrinsics):
        intrinsics = torch.tensor(intrinsics, device=device, dtype=torch.float32)
    fx, fy, cx, cy = intrinsics

    # Create a meshgrid of pixel coordinates using torch.
    i = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
    j = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)

    # Compute ray directions in camera coordinates following the hint:
    # X_cam = (x - cx) / fx, Y_cam = (y - cy) / fy, Z_cam = 1
    dirs = torch.stack([(i - cx) / fx ,
                        (j - cy) / fy,
                        torch.ones_like(i)], dim=-1)  # shape: [H, W, 3]

    # Rotate the camera directions to world space using the rotation part of c2w.
    ray_directions = torch.einsum('hwc,dc->hwd', dirs, c2w[:3, :3])
    
    # Normalize the ray directions to unit length.
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

    # The ray origins are the same for every ray: the camera position given by c2w.
    ray_origins = c2w[:3, 3].expand(H, W, 3)

    return ray_origins, ray_directions



############################
# Part B: Ray Marching
############################


def sample_points_on_rays(
    ray_origins, ray_directions, num_samples=64, t_near=0.0, t_far=3.0
):
    """
    Sample points along the rays.

    Args:
        ray_origins: [H, W, 3] ray origin points
        ray_directions: [H, W, 3] ray direction vectors
        num_samples: Number of sample points along each ray
        t_near: Near plane distance
        t_far: Far plane distance

    Returns:
        points: [H, W, num_samples, 3] sampled points in 3D space
        ts: [num_samples] distances along the rays
    """
    # TODO: Implement this function
    # 1. Generate uniformly spaced samples along each ray
    # 2. Compute the 3D coordinates of each sample point
    # 3. Return an array of sample points with shape [H, W, num_samples, 3]

    t_samples = torch.linspace(t_near, t_far, num_samples, device=ray_origins.device)

    t_samples_expanded = t_samples.view(1, 1, num_samples, 1)

    # Compute the 3D coordinates of each sample point
    points = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * t_samples_expanded

    return points, t_samples



    


############################
# Part C: Volumetric Rendering
############################


def volume_rendering(densities, colors, deltas):
    """
    Perform volume rendering to compute pixel colors from densities and colors.

    Args:
        densities: [H, W, num_samples, 1] density values at each sample point
        colors: [H, W, num_samples, 3] colors at each sample point
        deltas: [num_samples] intervals between adjacent sample points

    Returns:
        image: [H, W, 3] rendered image
    """
    # TODO: Implement this function
    # 1. Initialize accumulated color and transmittance
    # 2. For each sample along the ray:
    #    - Compute alpha from density and delta
    #    - Update accumulated color and transmittance
    # 3. Return the final rendered image
    device = densities.device
    H, W, num_samples, _ = densities.shape
    sigma = densities[..., 0]

    deltas = deltas.view(1, 1, num_samples)

    alpha = 1 - torch.exp(-sigma * deltas)  # [H, W, num_samples]

    exp_term = torch.exp(-sigma * deltas)  # [H, W, num_samples]

    
    T_inclusive = torch.cumprod(exp_term, dim=2)  # [H, W, num_samples]
   
    ones = torch.ones(H, W, 1, device=device)
    T_exclusive = torch.cat([ones, T_inclusive[..., :-1]], dim=2)  # [H, W, num_samples]

   
    weights = T_exclusive * alpha  # [H, W, num_samples]

    weights = weights.unsqueeze(-1)

    
    image = torch.sum(weights * colors, dim=2)  # [H, W, 3]

    return image

    


############################
# Part D: Putting It All Together
############################


def render_implicit(model, c2w, intrinsics, H=128, W=128, num_samples=64):
    """
    Render an implicit function using volumetric rendering.

    Args:
        model: Implicit function model
        c2w: [4,4] camera-to-world transformation
        intrinsics: [fx, fy, cx, cy] camera intrinsics
        H: Image height
        W: Image width
        num_samples: Number of samples along each ray

    Returns:
        image: [H, W, 3] rendered image
    """
    device = c2w.device

    # 1. Generate rays
    ray_origins, ray_directions = camera_param_to_rays(c2w, intrinsics, H, W)

    # 2. Sample points along rays
    points, ts = sample_points_on_rays(
        ray_origins, ray_directions, num_samples=num_samples, t_near=0.0, t_far=3.0
    )

    # 3. Query model for densities and colors
    H_, W_, N, _ = points.shape
    points_flat = points.reshape(-1, 3)
    densities_flat, colors_flat = model(points_flat)
    densities = densities_flat.view(H_, W_, N, 1)
    colors = colors_flat.view(H_, W_, N, 3)

    # 4. Volume rendering
    t_far, t_near = 3.0, 0.0
    delta = (t_far - t_near) / num_samples
    deltas = delta * torch.ones(N, device=device)
    image = volume_rendering(densities, colors, deltas)

    return image


############################
# Demo
############################


def demo():
    """
    Demonstrate volumetric rendering by rendering two views of the implicit model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    model = SimpleImplicitModel().to(device)

    # Camera intrinsics
    # you can use the debugging intrinsics for debugging, it's correctly rendered images are shown in folder expected_renders_for_debug/
    # after you have finished the problem, you can use the submit intrinsics to render the image and attach it to your pdf report.
    fx, fy, cx, cy = {
        "debug": [100.0, 100.0, 64.0, 64.0],
        "submit": [75.0, 75.0, 64.0, 64.0],
    }["submit"]
    intrinsics = torch.tensor([fx, fy, cx, cy], device=device)

    # Two camera views
    # View 1: Looking at the scene from positive z-axis
    c2w_1 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 2.0],  # Looking down negative z-axis, positioned at z=2
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )

    # View 2: Looking at the scene at an angle
    theta = math.radians(30)
    c2w_2 = torch.tensor(
        [
            [math.cos(theta), 0.0, math.sin(theta), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-math.sin(theta), 0.0, math.cos(theta), -1.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )

    # Try multiple sample counts to see the effect
    sample_counts = [256]

    fig, axes = plt.subplots(
        len(sample_counts), 2, figsize=(10, 5 * len(sample_counts))
    )
    if len(sample_counts) == 1:
        axes = [axes]

    for i, num_samples in enumerate(sample_counts):
        # Render two views
        img_1 = render_implicit(
            model, c2w_1, intrinsics, H=128, W=128, num_samples=num_samples
        )
        img_2 = render_implicit(
            model, c2w_2, intrinsics, H=128, W=128, num_samples=num_samples
        )

        # Convert to numpy for plotting
        img_1_np = img_1.detach().cpu().numpy()
        img_2_np = img_2.detach().cpu().numpy()

        # Plot
        axes[i][0].imshow(img_1_np)
        axes[i][0].set_title(f"View 1 ({num_samples} samples)")
        axes[i][0].axis("off")

        axes[i][1].imshow(img_2_np)
        axes[i][1].set_title(f"View 2 ({num_samples} samples)")
        axes[i][1].axis("off")
    plt.savefig("volume_rendering.png")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    demo()
