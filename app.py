"""
app.py — Chess AlphaZero Web Server

FastAPI server that:
  • Serves a chess web UI for humans to play against the AI
  • Runs background self-play (MCTS) in a daemon thread
  • Broadcasts live self-play positions via SSE
  • Learns from human games (stores training samples + gradient updates)
  • Tracks Elo rating and progression history
  • Persists all knowledge via checkpoints

Run:
    python3 app.py
Then open http://localhost:8000
"""

import asyncio
import atexit
import json
import os
import random
import threading
import time
import traceback
from contextlib import asynccontextmanager

import chess
import chess.pgn
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chess_model as M
from chess_model import load_checkpoint, save_checkpoint
from chess_env import encode, idx_to_move, move_to_idx, legal_mask, mirror_sample, ACTION_SIZE
from chess_mcts import MCTS
from chess_wargames import az_update, selfplay_game

STATIC_DIR      = os.path.join(os.path.dirname(__file__), "static")
MCTS_SIMS_SP    = 100   # simulations per move during self-play
import torch as _t
MCTS_SIMS_HUMAN = 50 if _t.cuda.is_available() else 10
TRAIN_STEPS     = 5     # gradient steps after each game
SAVE_EVERY_SP   = 50    # self-play games between saves
SAVE_EVERY_HU   = 10    # human games between saves
MAX_MOVES       = 80


# ── Live self-play state (for SSE stream) ─────────────────────────────
_sp_state: dict = {}
_sp_lock        = threading.Lock()


# ── Human game state ──────────────────────────────────────────────────
class HumanGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board        = chess.Board()
        self.active       = False
        self.move_history = []
        self.outcome      = None
        self.traj_w       = []    # (state, policy) for white (human)
        self.traj_b       = []    # (state, policy) for black (AI)
        self.mcts_root    = None  # reuse MCTS tree between moves
        self.n_sims       = MCTS_SIMS_HUMAN  # difficulty (sims per move)


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
    dummy_policy = np.zeros(ACTION_SIZE, dtype=np.float32)

    for traj, reward in [(current_game.traj_w, w_r), (current_game.traj_b, b_r)]:
        for state, policy in traj:
            p = policy if policy is not None else dummy_policy
            M.replay_buf.push(state, p, reward)
            M.replay_buf.push(*mirror_sample(state, p, reward))

    # Gradient update
    with M.model_lock:
        for _ in range(TRAIN_STEPS):
            az_update(M.policy_net, M.replay_buf, M.optimizer, M.scheduler)

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

    M.record_elo()
    M.human_game_active.clear()

    if M.human_games % SAVE_EVERY_HU == 0:
        save_checkpoint()


# ── Background self-play thread ────────────────────────────────────────
_sp_thread: threading.Thread | None = None   # module-level for watchdog access


def selfplay_loop():
    board = chess.Board()
    mcts  = MCTS(M.policy_net, M.device, n_sims=MCTS_SIMS_SP)

    while not M.shutdown_flag:
        if M.human_game_active.is_set():
            time.sleep(0.1)
            continue

        try:
            M.policy_net.eval()

            def _broadcast(fen, move_uci, move_n):
                with _sp_lock:
                    _sp_state.update({
                        "fen":    fen,
                        "move":   move_uci,
                        "game":   M.selfplay_games,
                        "move_n": move_n,
                    })

            samples, _, _ = selfplay_game(board, mcts, position_cb=_broadcast)

            for state, policy, value in samples:
                M.replay_buf.push(state, policy, value)
                M.replay_buf.push(*mirror_sample(state, policy, value))

            with M.model_lock:
                for _ in range(TRAIN_STEPS):
                    az_update(M.policy_net, M.replay_buf, M.optimizer, M.scheduler)

            M.total_games    += 1
            M.selfplay_games += 1

            if M.selfplay_games % SAVE_EVERY_SP == 0:
                save_checkpoint()

        except Exception:
            print("[selfplay] Error — resetting board and continuing:", flush=True)
            traceback.print_exc()
            board.reset()
            time.sleep(1)


# ── App lifespan ──────────────────────────────────────────────────────
def _start_selfplay_thread() -> threading.Thread:
    global _sp_thread
    t = threading.Thread(target=selfplay_loop, daemon=True, name="selfplay")
    t.start()
    _sp_thread = t
    return t


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not load_checkpoint():
        print("[app] Starting fresh — no checkpoint found.")

    M.policy_net.eval()
    atexit.register(save_checkpoint)

    _start_selfplay_thread()

    dev_str = str(M.device).upper()
    print(f"[app] AlphaZero+SE | {M.AZ_RES_BLOCKS} res blocks | "
          f"{M.AZ_CHANNELS} channels | {M.n_params:,} params | Device: {dev_str}")
    if M.device.type == "cpu":
        print("[app] ⚠  GPU strongly recommended — self-play will be slow on CPU")

    async def _watchdog():
        while not M.shutdown_flag:
            await asyncio.sleep(30)
            if _sp_thread and not _sp_thread.is_alive() and not M.shutdown_flag:
                print("[app] Selfplay thread died — restarting", flush=True)
                _start_selfplay_thread()

    wd = asyncio.create_task(_watchdog())

    yield

    M.shutdown_flag = True
    wd.cancel()
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
        "selfplay_alive":    _sp_thread is not None and _sp_thread.is_alive(),
    }


@app.get("/api/elo-history")
async def get_elo_history():
    return {"history": M.elo_history}


@app.get("/api/eval")
async def get_eval():
    """Run the value head on the current position. Returns eval from White's perspective."""
    with game_lock:
        board_copy = current_game.board.copy(stack=False)

    state_t = torch.tensor(encode(board_copy), dtype=torch.float32) \
                   .unsqueeze(0).to(M.device)
    with torch.no_grad(), M.model_lock:
        _, value_t = M.policy_net(state_t)

    value = value_t.item()
    # value head is from current player's perspective; convert to White's perspective
    if board_copy.turn == chess.BLACK:
        value = -value

    return {"eval": round(value, 3), "turn": "white" if board_copy.turn == chess.WHITE else "black"}


@app.get("/api/pgn")
async def get_pgn():
    """Return the current game as a PGN string."""
    with game_lock:
        move_history = list(current_game.move_history)
        outcome      = current_game.outcome

    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"]  = "WARGAMES"
    pgn_game.headers["White"]  = "Human"
    pgn_game.headers["Black"]  = f"AlphaZero (Elo {round(M.ai_elo)})"
    pgn_game.headers["Result"] = (
        "1-0" if outcome == "white" else
        "0-1" if outcome == "black" else
        "1/2-1/2" if outcome == "draw" else "*"
    )

    board = chess.Board()
    node  = pgn_game
    for uci in move_history:
        mv   = chess.Move.from_uci(uci)
        node = node.add_variation(mv)
        board.push(mv)

    return {"pgn": str(pgn_game)}


@app.get("/api/selfplay-stream")
async def selfplay_stream():
    """SSE endpoint: streams live self-play board positions as they happen."""
    async def generate():
        last_key = None
        while True:
            with _sp_lock:
                state = dict(_sp_state)
            key = (state.get("game"), state.get("move_n"))
            if state and key != last_key:
                last_key = key
                yield f"data: {json.dumps(state)}\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/new-game")
async def new_game(sims: int = Query(default=None)):
    with game_lock:
        current_game.reset()
        current_game.active = True
        if sims is not None:
            current_game.n_sims = max(5, min(sims, 800))
    M.human_game_active.set()
    return {"status": "ok", "fen": chess.Board().fen(), "n_sims": current_game.n_sims}


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

        # Record for learning (uniform policy for human moves)
        state  = encode(current_game.board)
        mask   = legal_mask(current_game.board)
        policy = mask / max(mask.sum(), 1.0)
        current_game.traj_w.append((state, policy))

        # Advance MCTS tree to match human's move
        if current_game.mcts_root is not None:
            current_game.mcts_root = current_game.mcts_root.children.get(move_to_idx(mv))

        current_game.board.push(mv)
        current_game.move_history.append(mv.uci())

        if current_game.board.is_game_over():
            _finalize_human_game()
            return {"status": "game_over", "outcome": current_game.outcome,
                    "fen": current_game.board.fen()}

        return {"status": "ok", "fen": current_game.board.fen()}


@app.get("/api/ai-move")
async def ai_move():
    """AI (black) calculates and plays its move using MCTS with tree reuse."""
    with game_lock:
        if not current_game.active:
            raise HTTPException(400, "No active game.")
        if current_game.board.turn != chess.BLACK:
            raise HTTPException(400, "Not AI's turn.")
        if current_game.outcome is not None:
            raise HTTPException(400, "Game is already over.")
        board_snapshot = current_game.board.copy(stack=True)
        state          = encode(board_snapshot)
        n_sims         = current_game.n_sims
        prev_root      = current_game.mcts_root

    # Claim a draw if available and the position is not winning for the AI
    if board_snapshot.can_claim_draw():
        state_t = torch.tensor(encode(board_snapshot), dtype=torch.float32) \
                       .unsqueeze(0).to(M.device)
        with torch.no_grad(), M.model_lock:
            _, val_t = M.policy_net(state_t)
        # val_t is from current player's (black/AI) perspective; claim if not winning
        if val_t.item() < 0.1:
            with game_lock:
                _finalize_human_game()
            return {"move": None, "status": "game_over", "outcome": "draw",
                    "fen": board_snapshot.fen(), "pv": []}

    # Run MCTS outside the lock (slow path) — reuse tree if available
    mcts = MCTS(M.policy_net, M.device, n_sims=n_sims)
    with M.model_lock:
        M.policy_net.eval()
        action, counts, new_root = mcts.get_policy(
            board_snapshot, temperature=0, root=prev_root
        )

    pv = mcts.get_pv(new_root, board_snapshot)

    mv = idx_to_move(action, board_snapshot)
    if mv is None:
        mv     = random.choice(list(board_snapshot.legal_moves))
        action = mv.from_square * 64 + mv.to_square

    total  = counts.sum()
    policy = counts / total if total > 0 else counts

    # Re-acquire lock to push move and update game state
    with game_lock:
        if not current_game.active or current_game.outcome is not None:
            raise HTTPException(400, "Game state changed during AI thinking.")
        if mv not in current_game.board.legal_moves:
            raise HTTPException(500, "AI selected an illegal move.")

        current_game.traj_b.append((state, policy))

        # Store the subtree under AI's chosen move for next search
        current_game.mcts_root = new_root.children.get(action)

        current_game.board.push(mv)
        current_game.move_history.append(mv.uci())

        if current_game.board.is_game_over():
            _finalize_human_game()
            return {"move": mv.uci(), "status": "game_over",
                    "outcome": current_game.outcome,
                    "fen": current_game.board.fen(), "pv": pv}

        return {"move": mv.uci(), "status": "ok",
                "fen": current_game.board.fen(), "pv": pv}


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
