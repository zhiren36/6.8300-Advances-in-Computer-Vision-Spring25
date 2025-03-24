from pathlib import Path
from typing import Literal, TypedDict
import torch
import json

from jaxtyping import Float
from torch import Tensor
import cv2


class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def get_kerberos() -> str:
    """Please return your kerberos ID as a string.
    This is required to match you with your specific puzzle dataset.
    """
    ## raise NotImplementedError("This is your homework.")

    return "zren"


def load_dataset(path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""

    ## raise NotImplementedError("This is your homework.")

    meta_file = path / "metadata.json"

    with open(meta_file, "r") as f:
        meta_data = json.load(f)

    extrinsics = torch.tensor(meta_data["extrinsics"])
    intrinsics = torch.tensor(meta_data["intrinsics"])

    images_dir = path / "images"
    image_paths = sorted(images_dir.glob("*.png"))

    loaded_images = []
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))

        loaded_images.append(torch.from_numpy(img_bgr))
    dataset = PuzzleDataset(
        extrinsics=extrinsics, intrinsics=intrinsics, images=loaded_images
    )

    return dataset


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    """Convert the dataset into OpenCV-style camera-to-world format. As a reminder, this
    format has the following specification:

    - The camera look vector is +Z.
    - The camera up vector is -Y.
    - The camera right vector is +X.
    - The extrinsics are in camera-to-world format, meaning that they transform points
      in camera space to points in world space.

    """
    data = dataset
    extr = data["extrinsics"]

    # Determine the transform convention by inspecting the first camera's translation.
    first_trans = extr[0, :3, 3]
    if torch.max(torch.abs(first_trans)) >= 1.95:
        print("Dataset was in world-to-camera format.")
        # Invert to obtain camera-to-world.
        cam2world_all = torch.inverse(extr)
    else:
        print("Dataset was in camera-to-world format.")
        cam2world_all = extr

    # Extract camera centers (translation parts) and compute the normalized look directions.
    centers = cam2world_all[:, :3, 3]  # Shape: [B, 3]
    look_dirs = -centers / centers.norm(
        dim=1, keepdim=True
    )  # Normalized, shape: [B, 3]

    # For the first camera, identify which rotation column aligns best with the computed look vector.
    R0 = cam2world_all[0, :3, :3]  # [3, 3]
    look_col = look_dirs[0, :3, None]  # [3, 1]
    diff_plus = R0 - look_col
    diff_minus = R0 + look_col
    mod_plus = diff_plus.abs().sum(dim=0)
    mod_minus = diff_minus.abs().sum(dim=0)
    col_errors = torch.min(mod_plus, mod_minus)
    look_idx = torch.argmin(col_errors)
    print(f"The {look_idx}-index column of the rotation matrix is the look vector.")

    # Determine the up vector from the other columns by checking alignment with world up ([0, 1, 0]).
    up_vector = None
    world_up = torch.tensor([0, 1, 0], dtype=torch.float32)
    for col in range(3):
        if col == look_idx:
            continue
        candidate = cam2world_all[:, :3, col]
        dp = torch.einsum("...i, i -> ...", candidate, world_up)
        if torch.max(torch.abs(dp)) > 0.1:
            print(
                f"The {col}-index column of the rotation matrix is identified as the up vector."
            )
            up_vector = candidate
            if dp[0] < 0:
                print("Flipping the up vector.")
                up_vector = -candidate
            break  # Use the first candidate that meets the criterion.

    # Flip the up vector to match OpenCV's convention (up = -Y).
    up_vector = -up_vector

    right_vector = torch.cross(up_vector, look_dirs, dim=1)

    R_opencv = torch.stack([right_vector, up_vector, look_dirs], dim=-1)

    extrinsics_new = torch.cat([R_opencv, centers.unsqueeze(-1)], dim=-1)

    # Append the homogeneous row [0, 0, 0, 1] for each camera.
    bottom_row = (
        torch.tensor([0, 0, 0, 1], dtype=extrinsics_new.dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(extrinsics_new.shape[0], -1, -1)
    )
    extrinsics_new = torch.cat([extrinsics_new, bottom_row], dim=1)

    new_dataset: PuzzleDataset = PuzzleDataset(
        extrinsics=extrinsics_new,
        intrinsics=data["intrinsics"],
        images=data["images"],
    )
    return new_dataset


def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""

    ## raise NotImplementedError("This is your homework.")

    return "c2w"


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""

    ## raise NotImplementedError("This is your homework.")
    return "+y"


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""

    ## raise NotImplementedError("This is your homework.")

    return "-x"


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""

    ## raise NotImplementedError("This is your homework.")

    return "+z"


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """

    ## raise NotImplementedError("This is your homework.")

    return "The idea is that we first figure out the position vector t for the camera. In c2w matrix, this t vector is just the last column with value 2 in one of its positions. Note this is the inverse to the look vector so we can find which column corresponds to it. Next, we can find the up vector because the hint says it has positive inner product with [0, 1, 0].  Finally, we can find the right vector by taking the cross product of the up and look vectors. With these three vectors, we can construct the rotation matrix and the final c2w matrix."
