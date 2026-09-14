"""Frozen encoder inference without simulation or training imports."""

import torch

from .model import SpongeEncoder
from .paths import read_path
from .preprocessing import Preprocessor


class FrozenSpongeEncoder:
    def __init__(self, path):
        checkpoint = torch.load(read_path(path), map_location="cpu", weights_only=True)
        self.preprocessor = Preprocessor.from_state(checkpoint["preprocessing"])
        self.model = SpongeEncoder().eval()
        self.model.load_state_dict(checkpoint["encoder"])
        self.model.requires_grad_(False)
        self.metadata = {key: value for key, value in checkpoint.items() if key != "encoder"}

    @torch.no_grad()
    def encode(self, raw_ft):
        x = torch.from_numpy(self.preprocessor.transform(raw_ft))
        mu, _ = self.model(x)
        return mu.numpy()
