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

import chess
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from chess_net import AlphaZeroNet
from chess_env import ACTION_SIZE, INPUT_PLANES

# ── Device ────────────────────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ── Network config ────────────────────────────────────────────────────
AZ_CHANNELS  = 192
AZ_RES_BLOCKS = 10
TRAINING_PIPELINE_VERSION = 2
REPLAY_SCHEMA_VERSION = 2

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
EXPERT_BUF_PATH = os.path.join(CHECKPOINT_DIR, "expert_buffer.npz")  # Stockfish-labeled
EXPERT_COMPACT_PATH = os.path.join(CHECKPOINT_DIR, "expert_buffer.compact.npz")
ENDGAME_EXPERT_PATH = os.path.join(CHECKPOINT_DIR, "endgame_expert.npz")
BUFFER_SEED_SIZE = 20_000   # max samples persisted across restarts

# Fraction of each training batch drawn from the non-evictable expert buffer.
# Stockfish evaluations provide dense non-zero value targets while elite human
# moves anchor the policy head when self-play is dominated by capped draws.
EXPERT_BATCH_FRAC = float(os.environ.get("EXPERT_BATCH_FRAC", "0.25"))
ENDGAME_WITHIN_EXPERT_FRAC = float(
    os.environ.get("ENDGAME_WITHIN_EXPERT_FRAC", "0.50")
)
EXPERT_SCHEMA_VERSION = 2
if not 0.0 <= EXPERT_BATCH_FRAC <= 1.0:
    raise ValueError("EXPERT_BATCH_FRAC must be between 0 and 1")
if not 0.0 <= ENDGAME_WITHIN_EXPERT_FRAC <= 1.0:
    raise ValueError("ENDGAME_WITHIN_EXPERT_FRAC must be between 0 and 1")

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

    def __init__(self, capacity: int | None = REPLAY_CAPACITY):
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


class ExpertReplayBuffer:
    """Compact, non-evictable expert samples with one action per position."""

    def __init__(self):
        self.states = np.empty((0, INPUT_PLANES, 8, 8), dtype=np.float16)
        self.actions = np.empty(0, dtype=np.int32)
        self.values = np.empty(0, dtype=np.float32)
        self.concepts = np.empty((0, 0), dtype=np.float32)

    def load(self, states, actions, values, concepts) -> None:
        self.states = states
        self.actions = actions
        self.values = values
        self.concepts = concepts

    def append(self, states, actions, values, concepts) -> None:
        if len(self) == 0:
            self.load(states, actions, values, concepts)
            return
        self.states = np.concatenate((self.states, states))
        self.actions = np.concatenate((self.actions, actions))
        self.values = np.concatenate((self.values, values))
        self.concepts = np.concatenate((self.concepts, concepts))

    def sample(self, n: int):
        indices = np.asarray(random.sample(range(len(self)), n), dtype=np.int64)
        policies = np.zeros((n, ACTION_SIZE), dtype=np.float32)
        policies[np.arange(n), self.actions[indices]] = 1.0
        return (
            self.states[indices],
            policies,
            self.values[indices],
            self.concepts[indices],
        )

    def __len__(self) -> int:
        return len(self.values)


expert_buf = ExpertReplayBuffer()
endgame_buf = ExpertReplayBuffer()


def _tensorize(s, p, v, c):
    """Stack raw sample tuples into device tensors (robust to legacy None concepts)."""
    states   = torch.tensor(np.array(s), dtype=torch.float32).to(device)
    policies = torch.tensor(np.array(p), dtype=torch.float32).to(device)
    values   = torch.tensor(v,           dtype=torch.float32).to(device)
    if any(x is None for x in c):
        from chess_env import N_CONCEPTS
        c = [np.zeros(N_CONCEPTS, dtype=np.float32) if x is None else x for x in c]
    concepts = torch.tensor(np.array(c), dtype=torch.float32).to(device)
    return states, policies, values, concepts


def sample_training_batch(
    selfplay_buffer: AZReplayBuffer,
    batch_size: int,
    expert_frac: float = EXPERT_BATCH_FRAC,
):
    """
    Build a training batch mixing self-play data with non-evictable expert data.

    The expert share supplies engine-evaluated value targets and elite human
    policy targets. Falls back to pure self-play when no expert data is loaded.
    Returns None until the self-play buffer can fill its share of the batch.
    """
    if not 0.0 <= expert_frac <= 1.0:
        raise ValueError("expert_frac must be between 0 and 1")
    available_expert = len(expert_buf) + len(endgame_buf)
    n_expert = min(int(batch_size * expert_frac), available_expert)
    n_self   = batch_size - n_expert
    if len(selfplay_buffer) < n_self:
        return None

    states = policies = values = concepts = None
    if n_self:
        samples = random.sample(selfplay_buffer.buf, n_self)
        s, p, v, c = zip(*samples)
        states = np.asarray(s)
        policies = np.asarray(p)
        values = np.asarray(v, dtype=np.float32)
        concepts = np.asarray(c)

    if n_expert:
        n_endgame = min(
            round(n_expert * ENDGAME_WITHIN_EXPERT_FRAC),
            len(endgame_buf),
        )
        n_general = min(n_expert - n_endgame, len(expert_buf))
        remaining = n_expert - n_endgame - n_general
        if remaining:
            extra_endgame = min(remaining, len(endgame_buf) - n_endgame)
            n_endgame += extra_endgame
            remaining -= extra_endgame
        if remaining:
            n_general += min(remaining, len(expert_buf) - n_general)

        expert_parts = []
        if n_general:
            expert_parts.append(expert_buf.sample(n_general))
        if n_endgame:
            expert_parts.append(endgame_buf.sample(n_endgame))
        for expert_states, expert_policies, expert_values, expert_concepts in expert_parts:
            if states is None:
                states = expert_states
                policies = expert_policies
                values = expert_values
                concepts = expert_concepts
            else:
                states = np.concatenate((states, expert_states))
                policies = np.concatenate((policies, expert_policies))
                values = np.concatenate((values, expert_values))
                concepts = np.concatenate((concepts, expert_concepts))

    order = np.random.permutation(batch_size)
    return _tensorize(
        states[order],
        policies[order],
        values[order],
        concepts[order],
    )


def _migrate_legacy_expert_actions(actions: np.ndarray) -> np.ndarray:
    """Convert the old 4096+from*64+to knight-promotion encoding."""
    from chess_env import move_to_idx

    migrated = actions.astype(np.int32, copy=True)
    for row in np.flatnonzero(migrated >= 4096):
        base = int(migrated[row]) - 4096
        move = chess.Move(
            base >> 6,
            base & 63,
            promotion=chess.KNIGHT,
        )
        migrated[row] = move_to_idx(move)
    return migrated


def _load_compact_expert_buffer(source_stat: os.stat_result) -> bool:
    if not os.path.exists(EXPERT_COMPACT_PATH):
        return False
    data = np.load(EXPERT_COMPACT_PATH)
    if (
        int(data.get("schema_version", 0)) != EXPERT_SCHEMA_VERSION
        or int(data.get("source_size", -1)) != source_stat.st_size
        or int(data.get("source_mtime_ns", -1)) != source_stat.st_mtime_ns
    ):
        return False
    expert_buf.load(
        data["states"],
        data["actions"].astype(np.int32),
        data["values"].astype(np.float32),
        data["concepts"].astype(np.float32),
    )
    return True


def _convert_expert_buffer(source_stat: os.stat_result) -> None:
    data = np.load(EXPERT_BUF_PATH)
    required = {"states", "policies", "values", "concepts"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError("expert buffer missing arrays: " + ", ".join(sorted(missing)))

    states = data["states"]
    dense_policies = data["policies"]
    values = data["values"].astype(np.float32)
    concepts = data["concepts"].astype(np.float32)
    n_samples = len(values)
    if (
        states.shape != (n_samples, INPUT_PLANES, 8, 8)
        or dense_policies.shape != (n_samples, ACTION_SIZE)
        or concepts.shape[0] != n_samples
    ):
        raise ValueError("expert buffer arrays have incompatible shapes")

    actions = np.empty(n_samples, dtype=np.int32)
    for start in range(0, n_samples, 2048):
        chunk = dense_policies[start:start + 2048]
        nonzero = np.count_nonzero(chunk, axis=1)
        sums = chunk.astype(np.float32).sum(axis=1)
        if np.any(nonzero != 1) or np.any(np.abs(sums - 1.0) > 1e-4):
            raise ValueError("expert policies must be one-hot distributions")
        actions[start:start + len(chunk)] = np.argmax(chunk, axis=1)

    source_schema = int(data["schema_version"]) if "schema_version" in data else 1
    if source_schema == 1:
        actions = _migrate_legacy_expert_actions(actions)
    elif source_schema != EXPERT_SCHEMA_VERSION:
        raise ValueError(f"unsupported expert schema: {source_schema}")
    if np.any(actions < 0) or np.any(actions >= ACTION_SIZE):
        raise ValueError("expert buffer contains out-of-range actions")

    del dense_policies
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    np.savez_compressed(
        EXPERT_COMPACT_PATH,
        states=states,
        actions=actions,
        values=values,
        concepts=concepts,
        schema_version=np.array(EXPERT_SCHEMA_VERSION, dtype=np.int64),
        source_size=np.array(source_stat.st_size, dtype=np.int64),
        source_mtime_ns=np.array(source_stat.st_mtime_ns, dtype=np.int64),
    )
    expert_buf.load(states, actions, values, concepts)


def load_expert_buffer() -> int:
    """Load Stockfish-labeled samples into the non-evictable expert buffer."""
    expert_buf.load(
        np.empty((0, INPUT_PLANES, 8, 8), dtype=np.float16),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float32),
        np.empty((0, 0), dtype=np.float32),
    )
    endgame_buf.load(
        np.empty((0, INPUT_PLANES, 8, 8), dtype=np.float16),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float32),
        np.empty((0, 0), dtype=np.float32),
    )
    if not os.path.exists(EXPERT_BUF_PATH):
        print("[expert] No expert buffer found — training on self-play only.")
    else:
        source_stat = os.stat(EXPERT_BUF_PATH)
        if not _load_compact_expert_buffer(source_stat):
            print("[expert] Converting legacy dense expert buffer to compact format...")
            _convert_expert_buffer(source_stat)

    if os.path.exists(ENDGAME_EXPERT_PATH):
        data = np.load(ENDGAME_EXPERT_PATH)
        if int(data.get("schema_version", 0)) != EXPERT_SCHEMA_VERSION:
            raise ValueError("endgame expert buffer has incompatible schema")
        states = data["states"]
        actions = data["actions"].astype(np.int32)
        values = data["values"].astype(np.float32)
        concepts = data["concepts"].astype(np.float32)
        if (
            states.shape[0] != len(actions)
            or len(actions) != len(values)
            or concepts.shape[0] != len(actions)
            or np.any(actions < 0)
            or np.any(actions >= ACTION_SIZE)
        ):
            raise ValueError("endgame expert buffer arrays are incompatible")
        endgame_buf.load(states, actions, values, concepts)
        print(f"[expert] Added {len(actions):,} tactical/endgame samples")

    total_expert = len(expert_buf) + len(endgame_buf)
    print(f"[expert] Loaded {total_expert:,} expert samples (non-evictable, "
          f"{EXPERT_BATCH_FRAC:.0%} of each batch)")
    return total_expert

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
            "training_pipeline_version": TRAINING_PIPELINE_VERSION,
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
        schema_version = np.array(REPLAY_SCHEMA_VERSION, dtype=np.int64),
    )
    print(f"[buffer] Saved {len(buf_list):,} samples → {BUFFER_PATH}")


def load_replay_buffer() -> int:
    """Restore persisted samples into the replay buffer. Returns count loaded."""
    if not os.path.exists(BUFFER_PATH):
        return 0
    data = np.load(BUFFER_PATH)
    schema_version = int(data["schema_version"]) if "schema_version" in data else 0
    if schema_version != REPLAY_SCHEMA_VERSION:
        print(
            "[buffer] Ignoring incompatible replay buffer "
            f"(schema {schema_version}, expected {REPLAY_SCHEMA_VERSION})"
        )
        return 0
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
    pipeline_version = int(ckpt.get("training_pipeline_version", 0))
    if load_training_state and pipeline_version != TRAINING_PIPELINE_VERSION:
        print(
            "[checkpoint] Training pipeline mismatch "
            f"(version {pipeline_version}, expected {TRAINING_PIPELINE_VERSION}) "
            "— starting a clean generation."
        )
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
