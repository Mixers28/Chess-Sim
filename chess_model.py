"""
chess_model.py — Shared singleton state for the AlphaZero chess engine.

Imported by both chess_wargames.py (standalone training)
and app.py (web server + background self-play).
All mutable state lives here as module globals.
"""

import os
import random
import threading
from collections import deque

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from chess_net import AlphaZeroNet
from chess_env import INPUT_PLANES

# ── Device ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Network config ────────────────────────────────────────────────────
AZ_CHANNELS  = 256
AZ_RES_BLOCKS = 20

# ── Hyperparameters ───────────────────────────────────────────────────
REPLAY_CAPACITY   = 200_000
LR                = 1e-3
WEIGHT_DECAY      = 1e-4
ELO_DEFAULT_AI    = 800
ELO_DEFAULT_HUMAN = 1200
ELO_K             = 32

# ── Checkpoint ────────────────────────────────────────────────────────
CHECKPOINT_DIR  = os.path.join(os.path.dirname(__file__), "checkpoint")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pt")

# ── Shared model objects ──────────────────────────────────────────────
policy_net = AlphaZeroNet(AZ_CHANNELS, AZ_RES_BLOCKS).to(device)
optimizer  = optim.Adam(policy_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Cosine annealing with warm restarts: LR decays smoothly then resets.
# T_0=2000 opt steps per first cycle; each restart doubles the cycle length.
scheduler  = lr_sched.CosineAnnealingWarmRestarts(
    optimizer, T_0=2000, T_mult=2, eta_min=1e-5
)

n_params = sum(p.numel() for p in policy_net.parameters())


# ── AlphaZero replay buffer ───────────────────────────────────────────
class AZReplayBuffer:
    """Stores (state, policy_target, value_target) triples."""

    def __init__(self, capacity: int = REPLAY_CAPACITY):
        self.buf = deque(maxlen=capacity)

    def push(self,
             state:   np.ndarray,   # (13, 8, 8)
             policy:  np.ndarray,   # (4096,)  visit-count distribution
             value:   float):       # ±1 or 0
        self.buf.append((state, policy, value))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, p, v = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32).to(device),
            torch.tensor(np.array(p), dtype=torch.float32).to(device),
            torch.tensor(v,           dtype=torch.float32).to(device),
        )

    def __len__(self) -> int:
        return len(self.buf)


replay_buf = AZReplayBuffer(REPLAY_CAPACITY)

# ── Training state ────────────────────────────────────────────────────
total_games    = 0
selfplay_games = 0
human_games    = 0
human_wins     = 0      # human won
human_losses   = 0      # AI won
human_draws    = 0
ai_elo         = float(ELO_DEFAULT_AI)
elo_history: list[list] = []   # [[game_n, elo], ...] — recorded after each human game

# ── Concurrency ───────────────────────────────────────────────────────
model_lock        = threading.Lock()
human_game_active = threading.Event()
shutdown_flag     = False


# ── Elo ───────────────────────────────────────────────────────────────
def update_elo(current_ai_elo: float,
               human_elo: float,
               ai_score: float) -> float:
    expected = 1.0 / (1.0 + 10 ** ((human_elo - current_ai_elo) / 400))
    return current_ai_elo + ELO_K * (ai_score - expected)


# ── Checkpoint save ───────────────────────────────────────────────────
def record_elo() -> None:
    """Append the current (total_games, ai_elo) to elo_history."""
    global elo_history
    elo_history.append([total_games, round(ai_elo)])


def save_checkpoint() -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with model_lock:
        torch.save({
            "policy_state_dict": policy_net.state_dict(),
            "optimizer_state":   optimizer.state_dict(),
            "scheduler_state":   scheduler.state_dict(),
            "az_channels":       AZ_CHANNELS,
            "az_res_blocks":     AZ_RES_BLOCKS,
            "az_input_planes":   INPUT_PLANES,
            "total_games":       total_games,
            "selfplay_games":    selfplay_games,
            "human_games":       human_games,
            "human_wins":        human_wins,
            "human_losses":      human_losses,
            "human_draws":       human_draws,
            "ai_elo":            ai_elo,
            "elo_history":       elo_history,
        }, CHECKPOINT_PATH)
    print(f"[checkpoint] Saved — games: {total_games:,}  Elo: {ai_elo:.0f}")


# ── Checkpoint load ───────────────────────────────────────────────────
def load_checkpoint() -> bool:
    global total_games, selfplay_games, human_games
    global human_wins, human_losses, human_draws, ai_elo, elo_history

    if not os.path.exists(CHECKPOINT_PATH):
        return False

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)

    # Verify architecture matches
    if (ckpt.get("az_channels") != AZ_CHANNELS or
            ckpt.get("az_res_blocks") != AZ_RES_BLOCKS or
            ckpt.get("az_input_planes") != INPUT_PLANES):
        print("[checkpoint] Architecture mismatch — starting fresh.")
        return False

    try:
        policy_net.load_state_dict(ckpt["policy_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
    except Exception as e:
        print(f"[checkpoint] State dict mismatch ({e}) — starting fresh.")
        return False

    total_games    = ckpt.get("total_games",    0)
    selfplay_games = ckpt.get("selfplay_games", 0)
    human_games    = ckpt.get("human_games",    0)
    human_wins     = ckpt.get("human_wins",     0)
    human_losses   = ckpt.get("human_losses",   0)
    human_draws    = ckpt.get("human_draws",    0)
    ai_elo         = float(ckpt.get("ai_elo",   ELO_DEFAULT_AI))
    elo_history    = ckpt.get("elo_history", [])

    print(f"[checkpoint] Resuming from game {total_games:,}  (Elo: {ai_elo:.0f})")
    return True
