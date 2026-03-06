"""
chess_net.py — AlphaZero neural network architecture.

Input : (B, 13, 8, 8)  — 13-plane board encoding
Output: policy_logits (B, 4096)  — unnormalised move scores
        value         (B,)       — position evaluation in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style network.
      channels  = 256  (AlphaZero default)
      n_res     = 20   (AlphaZero default)
    """

    def __init__(self, channels: int = 256, n_res: int = 20):
        super().__init__()
        self.channels = channels
        self.n_res    = n_res

        # Input tower
        self.input_tower = nn.Sequential(
            nn.Conv2d(13, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # Residual tower
        self.res_tower = nn.Sequential(*[ResBlock(channels) for _ in range(n_res)])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4096),
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
        policy = self.policy_head(x)          # (B, 4096)
        value  = self.value_head(x).squeeze(1) # (B,)
        return policy, value
