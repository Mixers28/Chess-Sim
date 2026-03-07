"""
chess_net.py — AlphaZero neural network with Squeeze-and-Excitation residual blocks.

Input : (B, INPUT_PLANES, 8, 8)
Output: policy_logits (B, 4096)  — unnormalised move scores
        value         (B,)       — position evaluation in [-1, 1]

SE blocks (Hu et al. 2018) add a channel attention gate after each conv pair:
  global avg pool → FC(C→C/16) → ReLU → FC(C/16→C) → Sigmoid → channel-wise scale
This lets the network dynamically re-weight feature channels per position,
improving strength for ~2% more parameters over plain ResBlocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_env import INPUT_PLANES, ACTION_SIZE


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        hidden = max(channels // ratio, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x).flatten(1)                    # (B, C)
        s = self.fc(s).unsqueeze(-1).unsqueeze(-1)     # (B, C, 1, 1)
        return x * s


class SEResBlock(nn.Module):
    """Residual block with SE channel attention gate."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se(self.bn2(self.conv2(x)))
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style network with SE residual blocks.
      channels = 256  (AlphaZero default)
      n_res    = 20   (AlphaZero default)
    """

    def __init__(self, channels: int = 256, n_res: int = 20):
        super().__init__()
        self.channels = channels
        self.n_res    = n_res

        # Input tower
        self.input_tower = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # SE Residual tower
        self.res_tower = nn.Sequential(*[SEResBlock(channels) for _ in range(n_res)])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, ACTION_SIZE),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),   # output in [-1, 1]
        )

    def forward(self, x: torch.Tensor):
        x = self.input_tower(x)
        x = self.res_tower(x)
        policy = self.policy_head(x)           # (B, 4096)
        value  = self.value_head(x).squeeze(1) # (B,)
        return policy, value
