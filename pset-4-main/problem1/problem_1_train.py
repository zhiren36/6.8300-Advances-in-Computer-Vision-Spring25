"""
Implement a training loop for the MLP and SIREN models.
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import ImageDataset
from typing import Dict, Any
from problem_1_mlp import MLP
from problem_1_siren import SIREN
from problem_1_gradients import gradient, laplace
from utils import plot

def psnr(model_output: torch.Tensor, ground_truth: torch.Tensor):
    mse = torch.nn.functional.mse_loss(model_output, ground_truth)
    return 10 * torch.log10(1 / mse)


def train(
    model,  # "MLP" or "SIREN"
    dataset: ImageDataset,  # Dataset of coordinates and pixels for an image
    lr: float,  # Learning rate
    total_steps: int,  # Number of gradient descent step
    steps_til_summary: int,  # Number of steps between summaries (i.e. print/plot)
    device: torch.device,  # "cuda" or "cpu"
    **kwargs: Dict[str, Any],  # Model-specific arguments
):
    """
    Train the model on the provided dataset.
    
    Given the **kwargs, initialize a neural field model and an optimizer.
    Then, train the model and log the loss and PSNR for each step. Examples
    in the notebook use MSE loss, but feel free to experiment with other
    objective functions. Additionally, in the notebook, we plot the reconstruction
    and various gradients every `steps_til_summary` steps using `utils.plot()`.

    You re allowed to change the arguments as you see fit so long as you can plot
    images of the reconstruction and the gradients/laplacian every `steps_til_summary` steps.
    Look at `should_look_like` for examples of what we would like to see. Make sure to
    also plot (MSE) loss and PSNR every `steps_til_summary` steps.

    You should train for `total_steps` gradient steps on the whole image (look at `ImageDataset` in `utils.py`)
    and visualize the results every `steps_til_summary` steps. The visualization must at least include:
    1. The MSE and PSNR
    2. The reconstructed image
    (Optionally you can also include the laplace or gradient of the image).

    PSNR is defined here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    # raise NotImplementedError("Not implemented!")

        # Convert string model name to class using a lookup dictionary
def train(
    model,               # "MLP" or "SIREN"
    dataset: ImageDataset,  # Dataset of coordinates and pixels for an image
    lr: float,           # Learning rate
    total_steps: int,    # Number of gradient descent steps
    steps_til_summary: int,  # Number of steps between summaries (print/plot)
    device: torch.device,    # "cuda" or "cpu"
    **kwargs: Dict[str, Any],  # Model-specific arguments
):
    # raise NotImplementedError("Not implemented!")

    # Convert string model name to class using a lookup dictionary
    MODEL_CLASSES = {
        "MLP": MLP,
        "SIREN": SIREN,
    }
    model_cls = MODEL_CLASSES[model]
    neural_field = model_cls(**kwargs).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(neural_field.parameters(), lr=lr)

    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    losses = []
    psnrs = []

    for step in range(1, total_steps + 1):
        for coords, ground_truth in dataloader:
            coords, ground_truth = coords.to(device), ground_truth.to(device)

            
            outputs, _ = neural_field(coords)

            loss = F.mse_loss(outputs, ground_truth)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                current_psnr = psnr(outputs, ground_truth)
                losses.append(loss.item())
                psnrs.append(current_psnr)

            if step % steps_til_summary == 0 or step == 1:
            #if step == total_steps:
                print(f"Step {step}/{total_steps}: Loss = {loss.item():.8f}, PSNR = {current_psnr:.4f}")

                with torch.no_grad():
                    coords_cpu = coords.detach().cpu()
                    outputs_cpu = outputs.detach().cpu()

                    H, W, C = dataset.img.permute(1, 2, 0).shape
                    reconstructed_img = outputs_cpu.view(H, W, C)
                coords = coords.clone().detach().requires_grad_(False)
                outputs_for_grad, coords_grad = neural_field(coords)
                gradients = gradient(outputs_for_grad, coords_grad).detach().cpu()
                laplacian = laplace(outputs_for_grad, coords_grad).detach().cpu()

                plot(
                    dataset=dataset, 
                    model_output=reconstructed_img, 
                    img_grad=gradients,
                    img_laplacian=laplacian,
                )
    return losses, psnrs
   