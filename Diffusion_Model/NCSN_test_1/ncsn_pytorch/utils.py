import numpy as np
import torch


def get_sigmas(n_scales=10, sigma_min=0.01, sigma_max=50.0):
    """Geometric spacing of noise levels (sigmas)."""
    return np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), n_scales))


def sample_noise(x, sigmas, device=None):
    """Sample noise for each image in batch using a random sigma from sigmas."""
    B = x.shape[0]
    idx = np.random.randint(0, len(sigmas), size=B)
    chosen = sigmas[idx]
    if device is None:
        device = x.device
    sigma_t = torch.tensor(chosen, dtype=torch.float32, device=device).view(B, 1, 1, 1)
    noise = torch.randn_like(x) * sigma_t
    return noise, torch.tensor(chosen, dtype=torch.float32, device=device)


@torch.no_grad()
def annealed_langevin_dynamics(score_model, shape, sigmas, n_steps=100, step_size=0.0002, device='cpu'):
    """Simple annealed Langevin sampling loop.

    Returns samples tensor of shape `shape`.
    """
    x = torch.randn(shape, device=device) * sigmas[0]
    for sigma in sigmas:
        sigma_t = torch.tensor(sigma, device=device, dtype=torch.float32)
        for _ in range(n_steps):
            noise = torch.randn_like(x) * np.sqrt(step_size * 2)
            score = score_model(x, sigma_t.expand(x.size(0)))
            x = x + step_size * score
            x = x + noise
    return x
