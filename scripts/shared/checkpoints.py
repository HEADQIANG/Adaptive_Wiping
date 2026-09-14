"""Atomic checkpoint serialization shared by offline trainers."""

from pathlib import Path

import torch

from .paths import writable_path


def save_checkpoint(path, payload):
    path = writable_path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)
