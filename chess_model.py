"""
chess_model.py — Shared singleton state for the AlphaZero chess engine.

Imported by both chess_wargames.py (standalone training)
and app.py (inference-only web server).
All mutable state lives here as module globals.

Checkpoint split:
  model.pt  — weights, optimizer, scheduler (GPU owns; pushed to Coolify)
  stats.pt  — Elo, game counts (each machine owns its own copy)

Set env var SYNC_MODEL_TARGET=user@host:/path/model.pt to auto-push
benchmarked model generations to the inference host.
"""

import os
import random
import shlex
import subprocess
import threading
from datetime import datetime, timezone
from collections import deque

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from chess_net import AlphaZeroNet
from chess_env import INPUT_PLANES

# ── Device ────────────────────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ── Network config ────────────────────────────────────────────────────
AZ_CHANNELS  = 192
AZ_RES_BLOCKS = 10

# ── Hyperparameters ───────────────────────────────────────────────────
REPLAY_CAPACITY   = 100_000
LR                = 1e-3
WEIGHT_DECAY      = 1e-4
ELO_DEFAULT_AI    = 800
ELO_DEFAULT_HUMAN = 1200
ELO_K             = 32

# ── Checkpoint paths ──────────────────────────────────────────────────
CHECKPOINT_DIR  = os.path.join(os.path.dirname(__file__), "checkpoint")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pt")  # legacy
MODEL_PATH      = os.path.join(CHECKPOINT_DIR, "model.pt")       # weights (shared)
STATS_PATH      = os.path.join(CHECKPOINT_DIR, "stats.pt")       # Elo/counts (local)
BUFFER_PATH     = os.path.join(CHECKPOINT_DIR, "replay_buffer.npz")
BUFFER_SEED_SIZE = 20_000   # max samples persisted across restarts

# Optional: set to deploy benchmarked model generations to the inference host
# e.g. export SYNC_MODEL_TARGET=user@server:/data/chess-sim/checkpoint/model.pt
SYNC_MODEL_TARGET = os.environ.get("SYNC_MODEL_TARGET", "")
SYNC_MODEL_PORT   = os.environ.get("SYNC_MODEL_PORT", "22")

# ── Shared model objects ──────────────────────────────────────────────
policy_net = AlphaZeroNet(AZ_CHANNELS, AZ_RES_BLOCKS).to(device)

# Main optimizer excludes concept head (weight decay was collapsing it)
_trunk_params   = [p for n, p in policy_net.named_parameters()
                   if "concept_bottleneck" not in n]
_concept_params = list(policy_net.concept_bottleneck.parameters())
optimizer = optim.Adam([
    {"params": _trunk_params,   "lr": LR, "weight_decay": WEIGHT_DECAY},
    {"params": _concept_params, "lr": LR, "weight_decay": 0.0},
])

# Cosine annealing with warm restarts: LR decays smoothly then resets.
# T_0=2000 opt steps per first cycle; each restart doubles the cycle length.
scheduler  = lr_sched.CosineAnnealingWarmRestarts(
    optimizer, T_0=2000, T_mult=2, eta_min=1e-5
)

n_params = sum(p.numel() for p in policy_net.parameters())


# ── AlphaZero replay buffer ───────────────────────────────────────────
class AZReplayBuffer:
    """Stores (state, policy_target, value_target, concept_labels) tuples."""

    def __init__(self, capacity: int = REPLAY_CAPACITY):
        self.buf = deque(maxlen=capacity)

    def push(self,
             state:    np.ndarray,            # (19, 8, 8)
             policy:   np.ndarray,            # (ACTION_SIZE,) visit-count distribution
             value:    float,                 # ±1 or 0
             concepts: np.ndarray | None = None):  # (N_CONCEPTS,) or None for legacy
        self.buf.append((state, policy, value, concepts))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, p, v, c = zip(*batch)
        states   = torch.tensor(np.array(s), dtype=torch.float32).to(device)
        policies = torch.tensor(np.array(p), dtype=torch.float32).to(device)
        values   = torch.tensor(v,           dtype=torch.float32).to(device)
        # Handle legacy samples that predate concept labels (concepts=None)
        if c[0] is None:
            from chess_env import N_CONCEPTS
            concepts = torch.zeros(len(c), N_CONCEPTS, dtype=torch.float32).to(device)
        else:
            concepts = torch.tensor(np.array(c), dtype=torch.float32).to(device)
        return states, policies, values, concepts

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
model_training_games = 0
model_selfplay_games = 0
model_saved_at = ""
model_version = "unversioned"

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


def record_elo() -> None:
    """Append the current (total_games, ai_elo) to elo_history."""
    global elo_history
    elo_history.append([total_games, round(ai_elo)])


def _atomic_torch_save(payload: dict, path: str) -> None:
    """Write a torch checkpoint without exposing a partially written file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _stats_payload() -> dict:
    return {
        "total_games":    total_games,
        "selfplay_games": selfplay_games,
        "human_games":    human_games,
        "human_wins":     human_wins,
        "human_losses":   human_losses,
        "human_draws":    human_draws,
        "ai_elo":         ai_elo,
        "elo_history":    elo_history,
    }


def save_stats(path: str = STATS_PATH) -> None:
    """Persist machine-local counters without writing model weights."""
    _atomic_torch_save(_stats_payload(), path)


# ── Checkpoint save ───────────────────────────────────────────────────
def save_checkpoint(*, sync_model: bool = True) -> None:
    global model_training_games, model_selfplay_games
    global model_saved_at, model_version

    saved_at = datetime.now(timezone.utc).isoformat()
    version = f"sp-{selfplay_games}-games-{total_games}"
    with model_lock:
        # model.pt is trainer-owned and shared with inference machines.
        _atomic_torch_save({
            "policy_state_dict":  policy_net.state_dict(),
            "optimizer_state":    optimizer.state_dict(),
            "scheduler_state":    scheduler.state_dict(),
            "az_channels":        AZ_CHANNELS,
            "az_res_blocks":      AZ_RES_BLOCKS,
            "az_input_planes":    INPUT_PLANES,
            "training_games":     total_games,
            "selfplay_games":     selfplay_games,
            "model_saved_at":     saved_at,
            "model_version":      version,
        }, MODEL_PATH)
        save_stats()
        model_training_games = total_games
        model_selfplay_games = selfplay_games
        model_saved_at = saved_at
        model_version = version
    print(f"[checkpoint] Saved — games: {total_games:,}  Elo: {ai_elo:.0f}")
    save_replay_buffer()
    if sync_model:
        _sync_model()


def _split_remote_target(target: str) -> tuple[str, str]:
    """Split user@host:/path into SSH host and remote path."""
    if ":" not in target:
        raise ValueError("SYNC_MODEL_TARGET must use user@host:/absolute/path format")
    host, remote_path = target.split(":", 1)
    if not host or not remote_path.startswith("/"):
        raise ValueError("SYNC_MODEL_TARGET must use user@host:/absolute/path format")
    return host, remote_path


def _sync_model() -> None:
    """Atomically push model.pt to the inference host."""
    if not SYNC_MODEL_TARGET:
        return
    try:
        host, remote_path = _split_remote_target(SYNC_MODEL_TARGET)
        remote_temp = f"{remote_path}.uploading"
        subprocess.run(
            ["scp", "-P", SYNC_MODEL_PORT, MODEL_PATH, f"{host}:{remote_temp}"],
            check=True,
        )
        activate_command = (
            f"test -s {shlex.quote(remote_temp)} && "
            f"mv -f -- {shlex.quote(remote_temp)} {shlex.quote(remote_path)}"
        )
        subprocess.run(
            ["ssh", "-p", SYNC_MODEL_PORT, host, activate_command],
            check=True,
        )
        print(f"[sync] model.pt → {SYNC_MODEL_TARGET} (atomic)")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[sync] Warning: model sync failed ({exc})")


# ── Replay buffer save/load ───────────────────────────────────────────
def save_replay_buffer() -> None:
    """Persist up to BUFFER_SEED_SIZE samples to disk (float16 to save space)."""
    if len(replay_buf) == 0:
        return
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    buf_list = list(replay_buf.buf)
    # Take the most recent BUFFER_SEED_SIZE samples
    buf_list = buf_list[-BUFFER_SEED_SIZE:]
    s, p, v, c = zip(*buf_list)
    np.savez_compressed(
        BUFFER_PATH,
        states   = np.array(s, dtype=np.float16),
        policies = np.array(p, dtype=np.float16),
        values   = np.array(v, dtype=np.float32),
        concepts = np.array(c, dtype=np.float32),
    )
    print(f"[buffer] Saved {len(buf_list):,} samples → {BUFFER_PATH}")


def load_replay_buffer() -> int:
    """Restore persisted samples into the replay buffer. Returns count loaded."""
    if not os.path.exists(BUFFER_PATH):
        return 0
    data = np.load(BUFFER_PATH)
    states   = data["states"].astype(np.float32)
    policies = data["policies"].astype(np.float32)
    values   = data["values"]
    concepts = data["concepts"]
    for i in range(len(values)):
        replay_buf.push(states[i], policies[i], float(values[i]), concepts[i])
    print(f"[buffer] Loaded {len(values):,} seed samples from disk")
    return len(values)


# ── Checkpoint load ───────────────────────────────────────────────────
def load_checkpoint(
    *,
    stats_path: str = STATS_PATH,
    fallback_stats_path: str | None = None,
    load_buffer: bool = True,
    load_training_state: bool = True,
) -> bool:
    """Load checkpoint — new split format first, legacy fallback for migration."""
    if os.path.exists(MODEL_PATH):
        return _load_split(
            stats_path=stats_path,
            fallback_stats_path=fallback_stats_path,
            load_buffer=load_buffer,
            load_training_state=load_training_state,
        )
    if os.path.exists(CHECKPOINT_PATH):
        print("[checkpoint] Migrating legacy checkpoint.pt → split format")
        return _load_legacy()
    return False


def _load_model_weights(path: str, *, load_training_state: bool = True) -> bool:
    """Load weights/optimizer/scheduler from path. Returns False on mismatch."""
    global model_training_games, model_selfplay_games
    global model_saved_at, model_version

    ckpt = torch.load(path, map_location=device, weights_only=True)
    if (ckpt.get("az_channels") != AZ_CHANNELS or
            ckpt.get("az_res_blocks") != AZ_RES_BLOCKS or
            ckpt.get("az_input_planes") != INPUT_PLANES):
        print("[checkpoint] Architecture mismatch — starting fresh.")
        return False
    try:
        policy_net.load_state_dict(ckpt["policy_state_dict"], strict=False)
        if load_training_state and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if load_training_state and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
    except Exception as e:
        print(f"[checkpoint] State dict mismatch ({e}) — starting fresh.")
        return False
    model_training_games = int(ckpt.get("training_games", 0))
    model_selfplay_games = int(ckpt.get("selfplay_games", model_training_games))
    model_saved_at = str(ckpt.get("model_saved_at", ""))
    model_version = str(ckpt.get("model_version", "unversioned"))
    return True


def _load_stats(path: str) -> None:
    """Load Elo and game counts from stats file."""
    global total_games, selfplay_games, human_games
    global human_wins, human_losses, human_draws, ai_elo, elo_history
    if not os.path.exists(path):
        print(f"[checkpoint] Warning: {path} not found — stats reset to defaults")
        return
    s = torch.load(path, map_location="cpu", weights_only=True)
    total_games    = s.get("total_games",    0)
    selfplay_games = s.get("selfplay_games", 0)
    human_games    = s.get("human_games",    0)
    human_wins     = s.get("human_wins",     0)
    human_losses   = s.get("human_losses",   0)
    human_draws    = s.get("human_draws",    0)
    ai_elo         = float(s.get("ai_elo",   ELO_DEFAULT_AI))
    elo_history    = s.get("elo_history", [])


def _load_split(
    *,
    stats_path: str,
    fallback_stats_path: str | None,
    load_buffer: bool,
    load_training_state: bool,
) -> bool:
    if not _load_model_weights(MODEL_PATH, load_training_state=load_training_state):
        return False
    if os.path.exists(stats_path):
        _load_stats(stats_path)
    elif fallback_stats_path and os.path.exists(fallback_stats_path):
        _load_stats(fallback_stats_path)
    else:
        _load_stats(stats_path)
    print(f"[checkpoint] Resuming — games: {total_games:,}  Elo: {ai_elo:.0f}")
    if load_buffer:
        load_replay_buffer()
    return True


def reload_model_weights() -> bool:
    """Reload a deployed model without importing optimizer state."""
    with model_lock:
        loaded = _load_model_weights(MODEL_PATH, load_training_state=False)
        if loaded:
            policy_net.eval()
    return loaded


def get_model_metadata() -> dict:
    return {
        "training_games": model_training_games,
        "selfplay_games": model_selfplay_games,
        "saved_at": model_saved_at,
        "version": model_version,
    }


def _load_legacy() -> bool:
    """Load old single-file checkpoint.pt and migrate to split format on next save."""
    global total_games, selfplay_games, human_games
    global human_wins, human_losses, human_draws, ai_elo, elo_history
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    if (ckpt.get("az_channels") != AZ_CHANNELS or
            ckpt.get("az_res_blocks") != AZ_RES_BLOCKS or
            ckpt.get("az_input_planes") != INPUT_PLANES):
        print("[checkpoint] Architecture mismatch — starting fresh.")
        return False
    try:
        policy_net.load_state_dict(ckpt["policy_state_dict"], strict=False)
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
    print(f"[checkpoint] Migrated — games: {total_games:,}  Elo: {ai_elo:.0f}")
    load_replay_buffer()
    return True
