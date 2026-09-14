"""Self-contained offline inference; deliberately no motion execution API."""

import numpy as np
import torch

from scripts.shared.encoder import FrozenSpongeEncoder
from scripts.shared.model import SpongeEncoder
from scripts.shared.paths import read_path
from scripts.shared.preprocessing import Preprocessor
from scripts.shared.real_models import HeightFeedback, XYDecoder
from scripts.shared.real_preprocessing import (
    CausalFTFilter,
    ChannelScaler,
    finite_array,
)


class EmbeddedSpongeEncoder(FrozenSpongeEncoder):
    """Use the existing encode implementation with an embedded checkpoint."""

    def __init__(self, checkpoint):
        self.preprocessor = Preprocessor.from_state(checkpoint["preprocessing"])
        self.model = SpongeEncoder().eval()
        self.model.load_state_dict(checkpoint["encoder"])
        self.model.requires_grad_(False)
        self.metadata = {key: value for key, value in checkpoint.items() if key != "encoder"}


class OfflinePolicy:
    def __init__(self, path):
        self._initialize(torch.load(read_path(path), map_location="cpu", weights_only=True))

    @classmethod
    def from_payload(cls, payload):
        obj = cls.__new__(cls)
        obj._initialize(payload)
        return obj

    def _initialize(self, payload):
        if (
            payload.get("schema_version") != 1
            or payload.get("artifact_kind") != "offline_wiping_policy"
            or payload.get("source_kind") not in ("real", "synthetic")
            or payload.get("hardware_ready") is not False
        ):
            raise ValueError("Invalid offline policy bundle")
        self.source_kind = payload["source_kind"]
        self.hardware_ready = False
        self.metadata = payload["metadata"]
        self.warnings = list(payload["warnings"])
        if self.source_kind == "synthetic" and "SYNTHETIC_SOFTWARE_TEST_ONLY" not in self.warnings:
            raise ValueError("Synthetic policy provenance is missing")
        self.encoder = EmbeddedSpongeEncoder(payload["encoder"])
        self.xy = XYDecoder().eval().requires_grad_(False)
        self.feedback = HeightFeedback().eval().requires_grad_(False)
        self.xy.load_state_dict(payload["xy"])
        self.feedback.load_state_dict(payload["feedback"])
        for model in (self.encoder.model, self.xy, self.feedback):
            if any(not torch.isfinite(x).all() for x in model.state_dict().values()):
                raise ValueError("Policy contains non-finite weights")
        self.scalers = {
            key: ChannelScaler.from_state(value) for key, value in payload["scalers"].items()
        }
        self.processing = payload["processing"]

    def encode_exploration(self, raw_ft):
        result = self.encoder.encode(raw_ft)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite exploration embedding")
        return result

    def _sponge(self, value):
        value = finite_array(value, "sponge embedding")
        if value.ndim != 2 or value.shape[1] != 5 or not len(value):
            raise ValueError("Expected nonempty sponge embedding [batch,5]")
        result = torch.from_numpy(value.astype(np.float32))
        if not torch.isfinite(result).all():
            raise ValueError("Embedding overflows float32")
        return result

    @torch.no_grad()
    def predict_xy(self, z_s):
        prediction = self.xy(self._sponge(z_s)).numpy()
        return self.scalers["xy"].inverse(prediction)

    @torch.no_grad()
    def predict_delta_h(self, z_s, filtered_ft_history):
        sponge = self._sponge(z_s)
        ft = finite_array(filtered_ft_history, "filtered FT history")
        if ft.ndim != 3 or ft.shape[0] != len(sponge) or ft.shape[2] != 6 or ft.shape[1] > 5:
            raise ValueError("Expected filtered FT history [batch,0..5,6]")
        if ft.shape[1] < 5:
            return None
        ft = torch.from_numpy(self.scalers["ft"].transform(ft))
        if not torch.isfinite(ft).all():
            raise ValueError("Normalized FT overflows float32")
        return self.scalers["delta_h"].inverse(self.feedback(sponge, ft).numpy())

    def make_ft_filter(self, sample_hz=100.0):
        return CausalFTFilter(
            sample_hz, self.processing["filter_order"], self.processing["filter_cutoff_hz"]
        )
