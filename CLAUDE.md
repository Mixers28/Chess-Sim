# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Hygiene

- At the start of every session, read all files in this project's Claude Code memory directory (`~/.claude/projects/<project-slug>/memory/`) to load context.
- After any significant decision, architectural change, or user correction, update or create the relevant memory file and refresh `MEMORY.md`.
- In long sessions: re-check memory files after every ~10 tool calls or whenever the topic shifts, and update stale entries before the session ends.

## Running the Project

```bash
# Web server (play vs AI at http://localhost:8000)
python app.py

# Standalone self-play training (CLI, 10k games)
python chess_wargames.py
```

**Dependencies**: Python 3.10+, PyTorch, FastAPI, Uvicorn, python-chess, numpy, pydantic. GPU strongly recommended; CPU self-play is ~1.2s/simulation.

Checkpoints are saved to `checkpoint/` (excluded from git):
- `model.pt` — trainer-owned weights and generation metadata; deployed atomically after each 500-game benchmark
- `stats.pt` — trainer game counts
- `web_stats.pt` — web-only human Elo and game counts
- `replay_buffer.npz` — up to 20k samples seeded across trainer restarts
- `human_games/*.npz` — human-game samples exported by the inference server

**Virtual env**: `./venv/` — activate with `source venv/bin/activate`.

## Architecture

This is an **AlphaZero-style chess engine** with a dedicated self-play trainer and an inference-only web UI for human play.

### Module Dependency Flow

```
chess_env.py          (standalone: board encoding, move indexing)
    ↓
chess_net.py          (AlphaZeroNet: SE-ResNet policy/value network)
    ↓
chess_model.py        (singleton shared state: model, optimizer, replay buffer, locks)
    ↓
chess_mcts.py         (batched virtual-loss MCTS, uses network + env)
    ↓
chess_wargames.py     (selfplay_game(), az_update(), train(); app imports constants only)
    ↓
app.py                (FastAPI inference server, model hot reload, human game logic)
```

### Shared Mutable State Pattern

`chess_model.py` holds all mutable globals: `policy_net`, `optimizer`, `scheduler`, `replay_buf`, `model_lock`, Elo data, and game counters. Both `app.py` (web server) and `chess_wargames.py` (standalone training) access these via `model_lock` for thread safety. Never modify these globals outside of lock context.

### Key Constants (in `chess_model.py` and `app.py`)

| Constant | Value | Notes |
|---|---|---|
| `AZ_CHANNELS` / `AZ_RES_BLOCKS` | 128 / 10 | Network size |
| `INPUT_PLANES` | 19 | Board encoding depth |
| `ACTION_SIZE` | 8192 | 4096 standard + 4096 knight underpromotions |
| `REPLAY_CAPACITY` | 100,000 | Circular training buffer |
| `MCTS_SIMS` | 200 | Trainer self-play simulations/move |
| `MCTS_SIMS_HUMAN` | 50/20 | GPU/CPU simulations for human play |
| `MAX_MOVES` | 200 | Half-move cap (self-play); cap-outs labeled -0.15, true draws 0 |
| `RESIGN_THRESHOLD` | -0.70 | Self-play resign threshold; 10% of games play with resign disabled |

### Board Encoding (`chess_env.py`)

19-plane `(19, 8, 8)` tensor:
- Planes 0–5: White pieces (P N B R Q K)
- Planes 6–11: Black pieces (P N B R Q K)
- Plane 12: Side to move
- Planes 13–16: Castling rights
- Plane 17: En passant square
- Plane 18: Repetition flag

Move indexing: `from_sq * 64 + to_sq` (0–4095), knight underpromotions offset by 4096.

### MCTS (`chess_mcts.py`)

Uses **batched virtual-loss** parallelism: multiple simulations are run with virtual losses applied, all leaf nodes are evaluated in a single GPU forward pass, then backed up together. This gives ~6× speedup over sequential simulation. Pass a previous `root` node to `get_policy()` to reuse the subtree.

### Training (`chess_wargames.py`)

- `selfplay_game()`: One full game with opening book (first 8 moves; book positions are not training samples), exponential temperature decay `τ(n) = max(0.05, exp(-n/20))`, resign mechanism (3 consecutive moves below -0.70, disabled in 10% of games for calibration).
- `_run_benchmark()`: every 500 self-play games, plays fixed-opponent matches (random + heuristic) and appends scores to `benchmark/history.jsonl` for objective strength tracking.
- Local recovery checkpoints are saved every 50 games without deployment. A completed 500-game benchmark saves and atomically deploys that generation.
- `az_update()`: Policy (cross-entropy) + value (MSE) loss, combined as `policy + 0.5 * value`, gradient clipped to norm ≤ 1.0. 5 steps per game.
- Data augmentation: `mirror_sample()` horizontally flips each position for 2× training data.

### Web Server (`app.py`)

The web service is inference-only. It never writes `model.pt`, optimizer state, or replay data. It watches for a stable replacement of `model.pt` and hot-reloads it only while no human game is active. Human outcomes update `web_stats.pt`; completed games are exported to `checkpoint/human_games/`.

### API Endpoints (`app.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve web UI |
| GET | `/api/state` | Current board state (FEN, legal moves, outcome) |
| GET | `/api/stats` | AI Elo, human counts, device, deployed model metadata |
| POST | `/api/new-game` | Start a new human vs AI game |
| POST | `/api/move` | Submit a human move (UCI format) |
| GET | `/api/ai-move` | Request the AI's move |
| POST | `/api/resign` | Cleanly terminate game (no Elo/training effect) |

### Deployment

A `Dockerfile` is included for containerized deployment (used with Coolify). It installs CPU-only PyTorch to keep image size manageable. The `checkpoint/` directory is bind-mounted in production so model weights, web stats, and human-game exports persist across container restarts.

```bash
docker build -t chess-sim .
docker run -p 8000:8000 -v ./checkpoint:/app/checkpoint chess-sim
```


## Reasoning Architecture

### ConceptBottleneck (implemented in `chess_net.py`)
- 6 concepts: material_balance, king_safety, piece_mobility,
  pawn_structure, space_control, tactical_threat
- Auxiliary head off the res_tower; does NOT bottleneck policy/value heads
- Supervised with auto-labels from `chess_env.compute_concept_labels()`
- Concept loss weight: 0.1
- Used for search-grounded move explanations in the web UI

Note: a chess → logistics transfer learning direction (Phase 2) was explored
and abandoned — the chess trunk showed no improvement over XGBoost on
logistics prediction. The project is chess-only.
