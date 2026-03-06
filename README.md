# Chess-Sim

An AlphaZero-style chess engine with a web UI. Play against the AI in your browser while it continuously learns from self-play in the background.

## How it works

The AI uses the same approach as DeepMind's AlphaZero:

- **Neural network** — a deep residual network (20 res blocks, 256 channels) with a policy head (move probabilities) and a value head (position evaluation).
- **MCTS** — Monte Carlo Tree Search guides move selection using the network's policy and value outputs. UCB exploration balances exploitation vs. discovery.
- **Self-play** — the engine continuously plays itself in a background thread, generating training data and updating the network via gradient descent.
- **Online learning** — human games are also used as training data; the network improves as you play against it.
- **Elo tracking** — the AI's rating is updated after every human game using standard Elo formula.

## Project structure

```
app.py              FastAPI web server — routes, game state, self-play thread
chess_model.py      Shared singleton: network, optimizer, replay buffer, Elo, checkpointing
chess_net.py        AlphaZeroNet architecture (ResBlock tower + policy/value heads)
chess_mcts.py       Monte Carlo Tree Search implementation
chess_env.py        Board encoding (13x8x8), move indexing, legal move masks
chess_wargames.py   Self-play loop, AlphaZero training update (az_update)
wargames.py         Standalone CLI training script
static/index.html   Web UI
checkpoint/         Saved model weights and training state
```

## Requirements

- Python 3.10+
- PyTorch
- FastAPI + Uvicorn
- python-chess

Install dependencies:

```bash
pip install torch fastapi uvicorn python-chess numpy pydantic
```

GPU is strongly recommended. On CPU, MCTS is ~1.2s/simulation, making response times slow.

## Running

### Web server (play against the AI)

```bash
python app.py
```

Then open [http://localhost:8000](http://localhost:8000) in your browser. You play as White, the AI plays as Black.

The server automatically:
- Loads the latest checkpoint on startup
- Runs self-play in the background to keep training
- Saves a checkpoint every 50 self-play games or every 10 human games

### Standalone training (no web UI)

```bash
python chess_wargames.py
```

Runs 10,000 self-play games with CLI progress output and saves checkpoints every 50 games.

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve web UI |
| GET | `/api/state` | Current board state (FEN, legal moves, outcome) |
| GET | `/api/stats` | AI Elo, game counts, replay buffer size, device |
| POST | `/api/new-game` | Start a new human vs AI game |
| POST | `/api/move` | Submit a human move (UCI format) |
| GET | `/api/ai-move` | Request the AI's move |

## Board encoding

The board is encoded as a `13 x 8 x 8` float32 tensor:

- Planes 0–5: white pieces (P N B R Q K)
- Planes 6–11: black pieces (P N B R Q K)
- Plane 12: side to move (1.0 = white, 0.0 = black)

Moves are encoded as integers `0–4095` (`from_square * 64 + to_square`). Promotions are always to queen.

## Configuration

Key constants in `chess_model.py` and `app.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `AZ_CHANNELS` | 256 | Network width |
| `AZ_RES_BLOCKS` | 20 | Residual tower depth |
| `REPLAY_CAPACITY` | 200,000 | Max replay buffer size |
| `LR` | 1e-3 | Adam learning rate |
| `MCTS_SIMS_SP` | 100 | Simulations per move (self-play) |
| `MCTS_SIMS_HUMAN` | 50 (GPU) / 10 (CPU) | Simulations per move (vs human) |
| `TRAIN_STEPS` | 5 | Gradient steps after each game |
