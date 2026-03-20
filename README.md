# Chess-Sim

An AlphaZero-style chess engine with a web UI. Play against the AI in your browser while it continuously learns from self-play in the background.

The longer-term goal is transfer learning: use the trained chess trunk as a feature extractor for logistics offer ranking (see `PHASE2_LOGISTICS.md`).

## How it works

- **Neural network** — 192-channel, 10-block SE-ResNet with three heads: policy (move probabilities), value (position evaluation), and concept (6 interpretable chess concepts).
- **MCTS** — batched virtual-loss Monte Carlo Tree Search. Multiple simulations run in parallel, all leaf evaluations are batched into one GPU forward pass, then backed up together (~6× faster than sequential).
- **Self-play** — the engine plays itself continuously in a background thread, generating training data and updating via gradient descent.
- **Concept bottleneck** — an auxiliary head predicts 6 chess concepts (material balance, king safety, piece mobility, pawn structure, space control, tactical threat) from fixed trunk features. Used to generate search-grounded move explanations.
- **Move explanations** — after each AI move, the top 3 MCTS candidates are compared by Q-value, visit share, and concept deltas. A deterministic sentence explains why the chosen move was preferred.
- **Elo tracking** — the AI's rating is updated after every human game using the standard Elo formula.

## Project structure

```
app.py               FastAPI web server — routes, game state, self-play thread, queue
chess_model.py       Shared singleton: network, optimizer, replay buffer, Elo, checkpointing
chess_net.py         AlphaZeroNet architecture (SE-ResBlock tower + policy/value/concept heads)
chess_mcts.py        Batched virtual-loss MCTS + explain_move_v2 (search-grounded explanation)
chess_env.py         Board encoding (19×8×8), move indexing, legal move masks, concept labels
chess_wargames.py    Self-play loop, az_update(), data augmentation (mirror)
pretrain_pgn.py      Supervised pre-training on elite PGN games before self-play
static/index.html    Web UI — board, candidates table, reasoning card, Elo chart
Dockerfile           CPU-only container for Coolify deployment
PHASE2_LOGISTICS.md  Research plan: chess trunk → logistics offer ranking transfer learning
```

## Requirements

Python 3.10+, PyTorch, FastAPI, Uvicorn, python-chess, numpy, pydantic.

**GPU install (recommended):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn python-chess numpy pydantic
```

**CPU install:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-chess numpy pydantic
```

Virtual env is at `./venv/` — activate with `source venv/bin/activate`.

## Running

### Web server (play against the AI)

```bash
python app.py
```

Open `http://localhost:8000`. You play as White; the AI plays as Black.

On startup the server loads the latest checkpoint, runs self-play in the background, and saves checkpoints every 50 self-play games or every 10 human games.

### Standalone self-play training (no web UI)

```bash
python chess_wargames.py
```

Runs self-play with CLI progress output, saves checkpoints every 50 games.

### Pre-training on PGN games

Run once after changing `AZ_CHANNELS` or starting fresh. Requires an elite PGN file (e.g. Lichess elite database).

```bash
python pretrain_pgn.py --pgn lichess_elite_2025-11.pgn --games 200000
```

Then start self-play: `python chess_wargames.py`

## Checkpoints

Saved to `checkpoint/` (excluded from git):

| File | Contents | Scope |
|------|----------|-------|
| `model.pt` | weights, optimizer, scheduler | shared across machines |
| `stats.pt` | Elo, game counts | per-machine |
| `replay_buffer.npz` | up to 20k seed samples | per-machine |

To sync `model.pt` to a remote Coolify server after each save:

```bash
export SYNC_MODEL_TARGET=user@host:/path/checkpoint/model.pt
export SYNC_MODEL_PORT=22   # optional, default 22
```

## Docker deployment

```bash
docker build -t chess-sim .
docker run -p 8000:8000 -v ./checkpoint:/app/checkpoint chess-sim
```

The image uses CPU-only PyTorch to keep image size manageable. Bind-mount `checkpoint/` so weights persist across container restarts.

## Board encoding

19-plane `(19, 8, 8)` float32 tensor:

| Planes | Content |
|--------|---------|
| 0–5 | White pieces (P N B R Q K) |
| 6–11 | Black pieces (P N B R Q K) |
| 12 | Side to move |
| 13–16 | Castling rights |
| 17 | En passant square |
| 18 | Repetition flag |

Move indexing: `from_sq * 64 + to_sq` (0–4095). Knight underpromotions use indices 4096–8191.

## Key constants

| Constant | Value | Location |
|----------|-------|----------|
| `AZ_CHANNELS` | 192 | `chess_model.py` |
| `AZ_RES_BLOCKS` | 10 | `chess_model.py` |
| `REPLAY_CAPACITY` | 100,000 | `chess_model.py` |
| `MCTS_SIMS_SP` | 100 | `app.py` |
| `MCTS_SIMS_HUMAN` | 50 (GPU) / 10 (CPU) | `app.py` |
| `MAX_MOVES` | 80 | `chess_wargames.py` |
| `RESIGN_THRESHOLD` | −0.9 | `chess_wargames.py` |

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve web UI |
| GET | `/api/state` | Current board state (FEN, legal moves, outcome) |
| GET | `/api/stats` | AI Elo, game counts, replay buffer size, device |
| GET | `/api/eval` | Current position value from the network |
| GET | `/api/elo-history` | Elo over time for chart rendering |
| POST | `/api/new-game` | Start a new human vs AI game (queued if server busy) |
| POST | `/api/move` | Submit a human move (UCI format) |
| GET | `/api/ai-move` | Request the AI's move + candidates + reasoning |
| POST | `/api/resign` | Cleanly terminate game (no Elo effect) |
| GET | `/api/selfplay-stream` | SSE stream of live self-play board positions |

## Roadmap

**Phase 1 — Chess (in progress)**
- [x] SE-ResNet 192ch / 10-block with policy, value, concept heads
- [x] Batched virtual-loss MCTS
- [x] PGN pre-training + self-play
- [x] Search-grounded move explanations (Reasoning v2)
- [ ] Session integrity — bind moves/resign to player_id
- [ ] Benchmark harness — win rate vs random, heuristic, Stockfish at low depth
- [ ] Promotion type support (all four pieces, not just queen)
- [ ] Move encoding tests

**Phase 2 — Logistics transfer learning**
- [ ] LogisticsInputAdapter: offer features → 256×8×8 latent
- [ ] Freeze chess trunk, train logistics heads on public datasets (Cargo 2000, DataCo)
- [ ] Compare: XGBoost baseline, MLP scratch, frozen trunk, fine-tuned trunk, trunk + concept supervision
- [ ] Evaluate on time:matters internal offer data (Phase 2b)

See `PHASE2_LOGISTICS.md` for the full experiment design.
