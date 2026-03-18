# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
# Web server (play vs AI at http://localhost:8000)
python app.py

# Standalone self-play training (CLI, 10k games)
python chess_wargames.py

# Tic-Tac-Toe Q-learning demo (unrelated)
python wargames.py
```

**Dependencies**: Python 3.10+, PyTorch, FastAPI, Uvicorn, python-chess, numpy, pydantic. GPU strongly recommended; CPU self-play is ~1.2s/simulation.

Checkpoints are saved to `checkpoint/checkpoint.pt` (excluded from git due to size) — automatically every 50 self-play games or every 10 human games.

## Architecture

This is an **AlphaZero-style chess engine** with a web UI for human play and continuous background self-play learning.

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
chess_wargames.py     (selfplay_game(), az_update(), train() — also imported by app.py)
    ↓
app.py                (FastAPI server, background selfplay thread, human game logic)
```

### Shared Mutable State Pattern

`chess_model.py` holds all mutable globals: `policy_net`, `optimizer`, `scheduler`, `replay_buf`, `model_lock`, Elo data, and game counters. Both `app.py` (web server) and `chess_wargames.py` (standalone training) access these via `model_lock` for thread safety. Never modify these globals outside of lock context.

### Key Constants (in `chess_model.py` and `app.py`)

| Constant | Value | Notes |
|---|---|---|
| `AZ_CHANNELS` / `AZ_RES_BLOCKS` | 256 / 20 | Network size |
| `INPUT_PLANES` | 19 | Board encoding depth |
| `ACTION_SIZE` | 8192 | 4096 standard + 4096 knight underpromotions |
| `REPLAY_CAPACITY` | 200,000 | Circular training buffer |
| `MCTS_SIMS_SP` | 100 | Self-play simulations/move |
| `MCTS_SIMS_HUMAN` | 50/10 | GPU/CPU simulations for human play |
| `MAX_MOVES` | 80 | Half-move cap before draw |
| `RESIGN_THRESHOLD` | -0.9 | Value below which engine may resign |

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

- `selfplay_game()`: One full game with opening book (first 8 moves), exponential temperature decay `τ(n) = max(0.05, exp(-n/20))`, resign mechanism (5 consecutive moves below -0.9).
- `az_update()`: Policy (cross-entropy) + value (MSE) loss, combined as `policy + 0.5 * value`, gradient clipped to norm ≤ 1.0. 5 steps per game.
- Data augmentation: `mirror_sample()` horizontally flips each position for 2× training data.

### Web Server (`app.py`)

`selfplay_loop()` runs as a background daemon thread, playing self-play games continuously and training after each. A watchdog thread monitors it and restarts on crash. Human games are tracked in `HumanGame` with MCTS tree reuse between moves. Human game outcomes feed into Elo tracking and contribute to the shared replay buffer.

### API Endpoints (`app.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve web UI |
| GET | `/api/state` | Current board state (FEN, legal moves, outcome) |
| GET | `/api/stats` | AI Elo, game counts, replay buffer size, device |
| POST | `/api/new-game` | Start a new human vs AI game |
| POST | `/api/move` | Submit a human move (UCI format) |
| GET | `/api/ai-move` | Request the AI's move |
| POST | `/api/resign` | Cleanly terminate game (no Elo/training effect) |

### Deployment

A `Dockerfile` is included for containerized deployment (used with Coolify). It installs CPU-only PyTorch to keep image size manageable. The `checkpoint/` directory is bind-mounted in production so model weights persist across container restarts.

```bash
docker build -t chess-sim .
docker run -p 8000:8000 -v ./checkpoint:/app/checkpoint chess-sim
```

### Unused Files

- `chess_dqn.py`: Early DQN approach, not used in current pipeline.
- `agent.py`, `game.py`, `wargames.py`: Tic-Tac-Toe Q-learning demo, completely separate from chess.


## Reasoning Architecture (planned)

### ConceptBottleneck
- 6 concepts: material_balance, king_safety, piece_mobility, 
  pawn_structure, space_control, tactical_threat
- Inserted between res_tower and policy/value heads
- Supervised with auto-labels from python-chess
- Concept loss weight: 0.1

### Transfer Learning Target: time:matters Logistics
- LogisticsInputAdapter maps offer feature vectors → 256×8×8 latent
- Chess res_tower trunk frozen initially, fine-tuned on offer outcomes
- Offer features: origin, destination, cargo_class, weight_kg, 
  deadline_hours, declared_value, dgr_class, service_tier
- New heads: OfferRankingHead (top-5 routes), OfferValueHead 
  (on-time probability × margin)

### Concept → Logistics Mapping
- material_balance    → margin_headroom
- king_safety         → critical_node_risk  
- piece_mobility      → route_optionality
- pawn_structure      → supply_chain_dependency
- space_control       → network_coverage
- tactical_threat     → disruption_probability