"""Small VAE matching the paper's per-timestep projection and five-dimensional z."""

import torch
from torch import nn


class SpongeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame = nn.Linear(6, 5)
        self.posterior = nn.Linear(2000, 10)

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (400, 6):
            raise ValueError("Expected [batch,400,6]")
        return self.posterior(self.frame(x).flatten(1)).chunk(2, dim=-1)


class SpongeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpongeEncoder()
        self.decode_hidden = nn.Sequential(nn.Linear(5, 2000), nn.ReLU(), nn.Dropout(0.1))
        self.decode_frame = nn.Linear(5, 6)

    def decode(self, z):
        return self.decode_frame(self.decode_hidden(z).reshape(-1, 400, 5))

    def forward(self, x, sample=True):
        mu, logvar = self.encoder(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decode(z), mu, logvar


def vae_loss(reconstruction, target, mu, logvar, beta=0.06):
    mse = (reconstruction - target).square().mean()
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(dim=-1).mean()
    return mse + beta * kl, mse, kl
