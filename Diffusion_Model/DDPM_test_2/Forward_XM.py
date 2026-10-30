import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, XMWrapper

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000
)

xm = XMWrapper(
    diffusion,
    candidates = 4 # generate 4 candidates per sample and pick minimum loss
)

training_images = torch.rand(8, 3, 128, 128)
loss = xm(training_images)
loss.backward()

# sampling works as usual

sampled_images = xm.sample(batch_size = 4)