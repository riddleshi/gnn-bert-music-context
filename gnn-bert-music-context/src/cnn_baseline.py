"""CNN-on-mel baseline (B2) and a tiny MLP on pooled audio (B4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MelCNN(nn.Module):
    """Two-block CNN over a downsampled log-mel heatmap (no graph, no text)."""

    def __init__(self, n_labels: int, in_h: int = 64, in_w: int = 32, channels: list[int] | None = None):
        super().__init__()
        c1, c2, c3 = channels or [16, 32, 64]
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(c3, n_labels)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        h = self.features(mel).flatten(1)
        return self.head(h)


class AudioMLP(nn.Module):
    def __init__(self, in_dim: int, n_labels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
