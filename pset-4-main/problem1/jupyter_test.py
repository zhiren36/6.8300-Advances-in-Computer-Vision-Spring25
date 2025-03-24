# %% [markdown]
# # Image fitting
# 
# After implementing all requested source code, you will fit an MLP and a SIREN to the same image.
# 
# These models train fairly fast on a CPU (~2 minutes on a M1 MacBook Pro), but feel free to train on a GPU or Colab for faster prototyping!

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import matplotlib.pyplot as plt
import torch

from problem_1_train import train
from utils import ImageDataset

# %%
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# %% [markdown]
# ## Experiment 1: MLPs vs SIRENs
# 
# First, fit the MLP.

# %%
dataset = ImageDataset(height=128)

mlp_loss, mlp_psnr = train(
    model="MLP",
    dataset=dataset,
    lr=1e-4,
    total_steps=2000,
    steps_til_summary=100,
    device=device,
    **dict(
        in_features=2,
        out_features=1,
        hidden_features=128,
        hidden_layers=3,
        activation="ReLU",
    ),
)

# %% [markdown]
# Now fit the SIREN.

# %%
dataset = ImageDataset(height=128)

siren_loss, siren_psnr = train(
    model="SIREN",
    dataset=dataset,
    lr=1e-4,
    total_steps=500,
    steps_til_summary=25,
    device=device,
    **dict(
        in_features=2,
        out_features=1,
        hidden_features=128,
        hidden_layers=3,
    ),
)

# %% [markdown]
# Compare the loss and PSNR for the two models.

# %%
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(mlp_loss, label="MLP")
plt.plot(siren_loss, label="SIREN")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.yscale("log")
plt.subplot(122)
plt.plot(mlp_psnr, label="MLP")
plt.plot(siren_psnr, label="SIREN")
plt.xlabel("Step")
plt.ylabel("PSNR")
plt.legend()
plt.show()

# %%



