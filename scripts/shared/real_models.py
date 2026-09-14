"""Paper-sized motion and force-feedback branches; no hardware dependencies."""

import torch
from torch import nn
from torch.nn import functional as F


class XYDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(5, 50)

    def forward(self, sponge):
        if sponge.ndim != 2 or sponge.shape[1] != 5:
            raise ValueError("Expected sponge embedding [batch,5]")
        return self.linear(self.dropout(sponge)).reshape(-1, 25, 2)


class FTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 25, kernel_size=3)
        self.conv2 = nn.Conv1d(25, 25, kernel_size=3)
        self.dropout = nn.Dropout(0.1)
        self.project = nn.Linear(25, 6)

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (5, 6):
            raise ValueError("Expected FT history [batch,5,6]")
        x = history.transpose(1, 2)
        x = self.dropout(F.relu(self.conv1(F.pad(x, (2, 0)))))
        x = self.dropout(F.relu(self.conv2(F.pad(x, (2, 0)))))
        return self.project(x[:, :, -1])


class HeightFeedback(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft_encoder = FTEncoder()
        self.height = nn.Sequential(
            nn.Linear(11, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

    def forward(self, sponge, history):
        if sponge.ndim != 2 or sponge.shape != (history.shape[0], 5):
            raise ValueError("Expected matching sponge embedding [batch,5]")
        return self.height(torch.cat((sponge, self.ft_encoder(history)), dim=-1))
