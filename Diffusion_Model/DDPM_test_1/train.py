import torch
from pathlib import Path

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

DATASET_DIR = (Path(__file__).resolve().parent / "images").resolve()
print(f"Using dataset folder: {DATASET_DIR}")

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=True,
)

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,
    sampling_timesteps=250,
)

trainer = Trainer(
    diffusion,
    str(DATASET_DIR),
    train_batch_size=4,
    train_lr=8e-5,
        train_num_steps=7000,
    gradient_accumulate_every=4,
    ema_decay=0.995,
    amp=True,
    calculate_fid=True,
)

trainer.train()