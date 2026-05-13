"""PyTorch TCN-MIMO model for 24-hour electricity price curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TcnMimoConfig:
    lookback_days: int = 14
    hidden_channels: int = 64
    tcn_levels: int = 3
    kernel_size: int = 3
    dropout: float = 0.15
    batch_size: int = 32
    epochs: int = 300
    patience: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    huber_weight: float = 0.55
    smape_weight: float = 0.35
    shape_weight: float = 0.10

    @classmethod
    def from_dict(cls, values: Dict) -> "TcnMimoConfig":
        base = cls()
        data = base.__dict__.copy()
        data.update({k: v for k, v in values.items() if k in data and v is not None})
        return cls(**data)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.net(x) + self.downsample(x))


class TcnEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, levels: int, kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        for level in range(levels):
            dilation = 2**level
            input_channels = in_channels if level == 0 else hidden_channels
            layers.append(TemporalBlock(input_channels, hidden_channels, kernel_size, dilation, dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TcnMimoNet(nn.Module):
    """Encode historical prices and target-day exogenous sequence, output 24 prices."""

    def __init__(self, exog_dim: int, config: TcnMimoConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_channels
        self.price_encoder = TcnEncoder(1, hidden, config.tcn_levels, config.kernel_size, config.dropout)
        self.exog_encoder = TcnEncoder(exog_dim, hidden, config.tcn_levels, config.kernel_size, config.dropout)
        self.head = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, price_history, target_exog):
        batch_size = price_history.shape[0]
        price_seq = price_history.reshape(batch_size, 1, -1)
        price_encoded = self.price_encoder(price_seq)
        price_context = F.adaptive_avg_pool1d(price_encoded, 1).repeat(1, 1, 24)

        exog_seq = target_exog.permute(0, 2, 1)
        exog_encoded = self.exog_encoder(exog_seq)
        fused = torch.cat([price_context, exog_encoded], dim=1)
        return self.head(fused).squeeze(1)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
