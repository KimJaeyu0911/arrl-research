import torch
import torch.nn as nn


class ScoreNet(nn.Module):
    """Simple CNN that predicts noise given a noisy image and scalar sigma.

    This is a minimal educational model, not the full NCSN architecture.
    """

    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.base = nn.Sequential(
            nn.Conv2d(in_channels + 1, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

        # small MLP to project sigma scalar to a map channel
        self.sigma_proj = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x, sigma):
        # x: (B, C, H, W), sigma: (B,) or (B,1)
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1)

        B, C, H, W = x.shape
        sig_map = self.sigma_proj(sigma.float())  # (B,1)
        sig_map = sig_map.view(B, 1, 1, 1).expand(-1, 1, H, W)

        inp = torch.cat([x, sig_map], dim=1)
        out = self.base(inp)
        return out
