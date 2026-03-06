"""
DQN components for chess:
  ChessDQN   — small CNN mapping board state → Q-values for all 4096 moves
  ReplayBuffer — fixed-capacity circular buffer of game transitions
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class ChessDQN(nn.Module):
    """
    Input : (B, 13, 8, 8) board planes
    Output: (B, 4096)     Q-value for every possible (from, to) move index
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, 4096),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class ReplayBuffer:
    """Stores (state, action, reward, next_state, done, next_legal_mask) tuples."""

    def __init__(self, capacity: int = 60_000):
        self.buf = deque(maxlen=capacity)

    def push(self,
             state:       np.ndarray,   # (13, 8, 8)
             action:      int,
             reward:      float,
             next_state:  np.ndarray,   # (13, 8, 8)
             done:        bool,
             next_mask:   np.ndarray):  # (4096,)
        self.buf.append((state, action, reward, next_state, done, next_mask))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d, nm = zip(*batch)
        return (
            torch.tensor(np.array(s),  dtype=torch.float32),
            torch.tensor(a,            dtype=torch.long),
            torch.tensor(r,            dtype=torch.float32),
            torch.tensor(np.array(ns), dtype=torch.float32),
            torch.tensor(d,            dtype=torch.bool),
            torch.tensor(np.array(nm), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)
