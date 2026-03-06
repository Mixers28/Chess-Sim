"""
app.py — Chess AlphaZero Web Server

FastAPI server that:
  • Serves a chess web UI for humans to play against the AI
  • Runs background self-play (MCTS) in a daemon thread
  • Learns from human games (stores training samples + gradient updates)
  • Tracks Elo rating
  • Persists all knowledge via checkpoints

Run:
    python3 app.py
Then open http://localhost:8000
"""

import atexit
import os
import random
import threading
import time
from contextlib import asynccontextmanager

import chess
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chess_model as M
from chess_model import load_checkpoint, save_checkpoint
from chess_env import encode, idx_to_move, legal_mask
from chess_mcts import MCTS
from chess_wargames import az_update, selfplay_game

STATIC_DIR      = os.path.join(os.path.dirname(__file__), "static")
MCTS_SIMS_SP    = 100   # simulations per move during self-play
# Human game: fewer sims for playable response time
# CPU: ~1.2s/sim → 10 sims ≈ 12s/move, 50 sims ≈ 60s/move
# GPU: ~0.05s/sim → 50 sims ≈ 2.5s/move
import torch as _t
MCTS_SIMS_HUMAN = 50 if _t.cuda.is_available() else 10
TRAIN_STEPS     = 5     # gradient steps after each game
SAVE_EVERY_SP   = 50    # self-play games between saves
SAVE_EVERY_HU   = 10    # human games between saves
MAX_MOVES       = 150


# ── Human game state ──────────────────────────────────────────────────
class HumanGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board        = chess.Board()
        self.active       = False
        self.move_history = []
        self.outcome      = None
        self.traj_w       = []    # (state, policy, color) for white (human)
        self.traj_b       = []    # (state, policy, color) for black (AI)


current_game = HumanGame()
game_lock    = threading.Lock()


# ── Finalize human game (called under game_lock) ───────────────────────
def _finalize_human_game():
    board   = current_game.board
    outcome = board.outcome()

    if outcome is None:
        result, w_r, b_r, ai_score = "draw", 0.0, 0.0, 0.5
    elif outcome.winner == chess.WHITE:
        result, w_r, b_r, ai_score = "white", 1.0, -1.0, 0.0
    elif outcome.winner == chess.BLACK:
        result, w_r, b_r, ai_score = "black", -1.0, 1.0, 1.0
    else:
        result, w_r, b_r, ai_score = "draw", 0.0, 0.0, 0.5

    current_game.outcome = result

    # Build training samples from human game trajectories
    final_enc = encode(board)
    dummy_policy = np.zeros(4096, dtype=np.float32)

    for traj, reward in [(current_game.traj_w, w_r), (current_game.traj_b, b_r)]:
        for state, policy in traj:
            M.replay_buf.push(state, policy if policy is not None else dummy_policy, reward)

    # Gradient update
    with M.model_lock:
        for _ in range(TRAIN_STEPS):
            az_update(M.policy_net, M.replay_buf, M.optimizer)

    # Elo + stats
    M.ai_elo       = M.update_elo(M.ai_elo, M.ELO_DEFAULT_HUMAN, ai_score)
    M.human_games += 1
    M.total_games += 1
    if ai_score == 1.0:
        M.human_losses += 1
    elif ai_score == 0.0:
        M.human_wins   += 1
    else:
        M.human_draws  += 1

    M.human_game_active.clear()

    if M.human_games % SAVE_EVERY_HU == 0:
        save_checkpoint()


# ── Background self-play thread ────────────────────────────────────────
def selfplay_loop():
    board = chess.Board()
    mcts  = MCTS(M.policy_net, M.device, n_sims=MCTS_SIMS_SP)

    while not M.shutdown_flag:
        if M.human_game_active.is_set():
            time.sleep(0.1)
            continue

        M.policy_net.eval()
        samples, _, _ = selfplay_game(board, mcts)

        for state, policy, value in samples:
            M.replay_buf.push(state, policy, value)

        with M.model_lock:
            for _ in range(TRAIN_STEPS):
                az_update(M.policy_net, M.replay_buf, M.optimizer)

        M.total_games    += 1
        M.selfplay_games += 1

        if M.selfplay_games % SAVE_EVERY_SP == 0:
            save_checkpoint()


# ── App lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not load_checkpoint():
        print("[app] Starting fresh — no checkpoint found.")

    M.policy_net.eval()
    atexit.register(save_checkpoint)

    sp = threading.Thread(target=selfplay_loop, daemon=True, name="selfplay")
    sp.start()

    dev_str = str(M.device).upper()
    print(f"[app] AlphaZero | {M.AZ_RES_BLOCKS} res blocks | "
          f"{M.AZ_CHANNELS} channels | {M.n_params:,} params | Device: {dev_str}")
    if M.device.type == "cpu":
        print("[app] ⚠  GPU strongly recommended — self-play will be slow on CPU")

    yield

    M.shutdown_flag = True
    save_checkpoint()


app = FastAPI(lifespan=lifespan)
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/state")
async def get_state():
    with game_lock:
        board = current_game.board
        return {
            "fen":          board.fen(),
            "turn":         "white" if board.turn == chess.WHITE else "black",
            "legal_moves":  [mv.uci() for mv in board.legal_moves],
            "outcome":      current_game.outcome,
            "move_history": current_game.move_history,
            "is_game_over": board.is_game_over(),
            "active":       current_game.active,
        }


@app.get("/api/stats")
async def get_stats():
    return {
        "ai_elo":            round(M.ai_elo),
        "total_games":       M.total_games,
        "selfplay_games":    M.selfplay_games,
        "human_games":       M.human_games,
        "human_wins":        M.human_wins,
        "human_losses":      M.human_losses,
        "human_draws":       M.human_draws,
        "replay_buffer":     len(M.replay_buf),
        "human_game_active": M.human_game_active.is_set(),
        "device":            str(M.device),
    }


@app.post("/api/new-game")
async def new_game():
    with game_lock:
        current_game.reset()
        current_game.active = True
    M.human_game_active.set()
    return {"status": "ok", "fen": chess.Board().fen()}


class MoveRequest(BaseModel):
    move: str


@app.post("/api/move")
async def human_move(req: MoveRequest):
    """Human (white) makes a move."""
    with game_lock:
        if not current_game.active:
            raise HTTPException(400, "No active game.")
        if current_game.board.turn != chess.WHITE:
            raise HTTPException(400, "Not white's turn.")
        if current_game.outcome is not None:
            raise HTTPException(400, "Game is already over.")

        try:
            mv = chess.Move.from_uci(req.move)
        except ValueError:
            raise HTTPException(400, f"Invalid UCI: {req.move!r}")

        if mv not in current_game.board.legal_moves:
            mv_q = chess.Move.from_uci(req.move + "q")
            if mv_q not in current_game.board.legal_moves:
                raise HTTPException(400, f"Illegal move: {req.move}")
            mv = mv_q

        # Record for learning (use uniform policy for human moves)
        state = encode(current_game.board)
        mask  = legal_mask(current_game.board)
        policy = mask / max(mask.sum(), 1.0)
        current_game.traj_w.append((state, policy))

        current_game.board.push(mv)
        current_game.move_history.append(mv.uci())

        if current_game.board.is_game_over():
            _finalize_human_game()
            return {"status": "game_over", "outcome": current_game.outcome,
                    "fen": current_game.board.fen()}

        return {"status": "ok", "fen": current_game.board.fen()}


@app.get("/api/ai-move")
async def ai_move():
    """AI (black) calculates and plays its move using MCTS."""
    with game_lock:
        if not current_game.active:
            raise HTTPException(400, "No active game.")
        if current_game.board.turn != chess.BLACK:
            raise HTTPException(400, "Not AI's turn.")
        if current_game.outcome is not None:
            raise HTTPException(400, "Game is already over.")

        board = current_game.board
        state = encode(board)

        # Run MCTS (greedy — temperature=0)
        mcts = MCTS(M.policy_net, M.device, n_sims=MCTS_SIMS_HUMAN)
        with M.model_lock:
            M.policy_net.eval()
            action, counts = mcts.get_policy(board, temperature=0)

        mv = idx_to_move(action, board)
        if mv is None:
            mv     = random.choice(list(board.legal_moves))
            action = mv.from_square * 64 + mv.to_square

        # Record trajectory
        total = counts.sum()
        policy = counts / total if total > 0 else counts
        current_game.traj_b.append((state, policy))

        board.push(mv)
        current_game.move_history.append(mv.uci())

        if board.is_game_over():
            _finalize_human_game()
            return {"move": mv.uci(), "status": "game_over",
                    "outcome": current_game.outcome, "fen": board.fen()}

        return {"move": mv.uci(), "status": "ok", "fen": board.fen()}


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
