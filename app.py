"""
app.py — Chess AlphaZero Web Server

FastAPI server that:
  • Serves a chess web UI for humans to play against the AI
  • Loads trainer-owned model checkpoints for inference
  • Exports human games for later trainer ingestion
  • Tracks Elo rating and progression history
  • Persists web-only statistics separately from model weights

Run:
    python3 app.py
Then open http://localhost:8000
"""

import asyncio
import atexit
import os
import random
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager

import chess
import chess.pgn
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chess_model as M
from chess_model import load_checkpoint
from chess_env import encode, idx_to_move, move_to_idx, legal_mask, compute_concept_labels
from chess_mcts import MCTS
from chess_wargames import RESIGN_THRESHOLD

STATIC_DIR      = os.path.join(os.path.dirname(__file__), "static")
WEB_STATS_PATH  = os.environ.get(
    "WEB_STATS_PATH",
    os.path.join(os.path.dirname(__file__), "checkpoint", "web_stats.pt"),
)
HUMAN_GAMES_DIR = os.environ.get(
    "HUMAN_GAMES_DIR",
    os.path.join(os.path.dirname(__file__), "checkpoint", "human_games"),
)
import torch as _t
_gpu = _t.cuda.is_available() or _t.backends.mps.is_available()
MCTS_SIMS_HUMAN = 50 if _gpu else 20

# ── Human game state ──────────────────────────────────────────────────
GAME_INACTIVITY_TIMEOUT = 120   # seconds before an orphaned game is auto-abandoned
QUEUE_EXPIRY_TIMEOUT    = 300   # seconds before a queued player is silently dropped


class HumanGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board            = chess.Board()
        self.active           = False
        self.move_history     = []
        self.outcome          = None
        self.traj_w           = []    # (state, policy, concepts, board_copy) for white (human)
        self.traj_b           = []    # (state, policy, concepts, board_copy) for black (AI)
        self.mcts_root        = None  # reuse MCTS tree between moves
        self.n_sims           = MCTS_SIMS_HUMAN  # difficulty (sims per move)
        self.player_id        = None  # ID of the player currently in this game slot
        self.last_activity    = time.time()  # updated on each move; used for timeout


current_game = HumanGame()
game_lock    = threading.Lock()

# ── Player queue ───────────────────────────────────────────────────────
# Each entry: {"player_id": str, "n_sims": int|None}
_game_queue: list[dict] = []
_queue_lock = threading.Lock()


def _dequeue_next() -> None:
    """Start the next queued game using the global game->queue lock order."""
    with game_lock:
        _dequeue_next_locked()


def _dequeue_next_locked() -> None:
    """Start the next queued game. The caller must hold game_lock."""
    now = time.time()
    entry = None
    with _queue_lock:
        while _game_queue:
            candidate = _game_queue[0]
            age = now - candidate.get("queued_at", now)
            if age > QUEUE_EXPIRY_TIMEOUT:
                print(f"[queue] Expired entry for {candidate['player_id']} ({age:.0f}s old)")
                _game_queue.pop(0)
                continue
            entry = _game_queue.pop(0)
            break
    if entry is None:
        return
    current_game.reset()
    current_game.active    = True
    current_game.player_id = entry["player_id"]
    if entry["n_sims"] is not None:
        current_game.n_sims = max(5, min(entry["n_sims"], 800))
    M.human_game_active.set()


def _export_human_game(result: str, white_reward: float, black_reward: float) -> None:
    """Persist human-game samples without mutating the inference model."""
    rows = [
        (state, policy, white_reward, concepts, "white")
        for state, policy, concepts, _ in current_game.traj_w
    ]
    rows.extend(
        (state, policy, black_reward, concepts, "black")
        for state, policy, concepts, _ in current_game.traj_b
    )
    if not rows:
        return

    os.makedirs(HUMAN_GAMES_DIR, exist_ok=True)
    filename = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}.npz"
    path = os.path.join(HUMAN_GAMES_DIR, filename)
    temp_path = f"{path}.tmp.npz"
    states, policies, values, concepts, colors = zip(*rows)
    try:
        np.savez_compressed(
            temp_path,
            states=np.asarray(states, dtype=np.float16),
            policies=np.asarray(policies, dtype=np.float16),
            values=np.asarray(values, dtype=np.float32),
            concepts=np.asarray(concepts, dtype=np.float32),
            colors=np.asarray(colors),
            moves=np.asarray(current_game.move_history),
            result=np.asarray(result),
            model_version=np.asarray(M.model_version),
        )
        os.replace(temp_path, path)
        print(f"[human-game] Exported {len(rows)} samples → {path}")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# ── Finalize human game (called under game_lock) ───────────────────────
def _finalize_human_game(ai_resigned: bool = False):
    board   = current_game.board
    outcome = board.outcome()

    if ai_resigned:
        result, w_r, b_r, ai_score = "white", 1.0, -1.0, 0.0   # human wins
    elif outcome is None:
        result, w_r, b_r, ai_score = "draw", 0.0, 0.0, 0.5
    elif outcome.winner == chess.WHITE:
        result, w_r, b_r, ai_score = "white", 1.0, -1.0, 0.0
    elif outcome.winner == chess.BLACK:
        result, w_r, b_r, ai_score = "black", -1.0, 1.0, 1.0
    else:
        result, w_r, b_r, ai_score = "draw", 0.0, 0.0, 0.5

    current_game.outcome = result
    try:
        _export_human_game(result, w_r, b_r)
    except Exception:
        print("[human-game] Export failed", flush=True)
        traceback.print_exc()

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
    current_game.active = False
    M.human_game_active.clear()
    M.save_stats(WEB_STATS_PATH)
    _dequeue_next_locked()


# ── App lifespan ──────────────────────────────────────────────────────
def _model_file_signature() -> tuple[int, int, int] | None:
    try:
        stat = os.stat(M.MODEL_PATH)
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


@asynccontextmanager
async def lifespan(app: FastAPI):
    migrating_legacy_web_stats = (
        not os.path.exists(WEB_STATS_PATH) and os.path.exists(M.STATS_PATH)
    )
    if not load_checkpoint(
        stats_path=WEB_STATS_PATH,
        fallback_stats_path=M.STATS_PATH,
        load_buffer=False,
        load_training_state=False,
    ):
        print("[app] Starting fresh — no checkpoint found.")
    elif migrating_legacy_web_stats:
        # Preserve old human counters/Elo without importing trainer generations.
        M.total_games = M.human_games
        M.selfplay_games = 0
        M.save_stats(WEB_STATS_PATH)

    M.policy_net.eval()
    atexit.register(M.save_stats, WEB_STATS_PATH)

    dev_str = str(M.device).upper()
    print(f"[app] AlphaZero+SE | {M.AZ_RES_BLOCKS} res blocks | "
          f"{M.AZ_CHANNELS} channels | {M.n_params:,} params | Device: {dev_str}")
    cpu_count = os.cpu_count() or 2
    torch.set_num_threads(max(1, cpu_count - 1))
    print(f"[app] Inference-only mode | PyTorch threads: {max(1, cpu_count - 1)}")

    loaded_signature = _model_file_signature()
    failed_signature = None
    pending_signature = None

    async def _model_watcher():
        nonlocal loaded_signature, failed_signature, pending_signature
        while not M.shutdown_flag:
            await asyncio.sleep(5)
            signature = _model_file_signature()
            if signature is None or signature in (loaded_signature, failed_signature):
                pending_signature = None
                continue
            if signature != pending_signature:
                pending_signature = signature
                continue
            with game_lock:
                if current_game.active:
                    continue
                if M.reload_model_weights():
                    loaded_signature = signature
                    failed_signature = None
                    pending_signature = None
                    print(
                        f"[app] Activated model {M.model_version} "
                        f"({M.model_training_games:,} training games)"
                    )
                else:
                    failed_signature = signature
                    pending_signature = None
                    print("[app] Rejected incompatible deployed model", flush=True)

    watcher = asyncio.create_task(_model_watcher())

    yield

    M.shutdown_flag = True
    watcher.cancel()
    M.save_stats(WEB_STATS_PATH)


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
    with _queue_lock:
        q_len = len(_game_queue)
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
        "queue_length":      q_len,
        "device":            str(M.device),
        "selfplay_alive":    False,
        "model":             M.get_model_metadata(),
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
        _, value_t, _ = M.policy_net(state_t)

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


@app.post("/api/new-game")
async def new_game(sims: int = Query(default=None)):
    player_id = str(uuid.uuid4())[:8]
    with game_lock:
        if current_game.active:
            stale = time.time() - current_game.last_activity > GAME_INACTIVITY_TIMEOUT
            if stale:
                # Orphaned game — abandon it and clear any stale queue entries
                print(f"[queue] Stale game for {current_game.player_id} — force-replacing")
                current_game.active  = False
                current_game.outcome = "abandoned"
                M.human_game_active.clear()
                with _queue_lock:
                    _game_queue.clear()
            else:
                # Active game in progress — queue this player
                with _queue_lock:
                    _game_queue.append({"player_id": player_id, "n_sims": sims,
                                        "queued_at": time.time()})
                    position = len(_game_queue)
                return {"status": "queued", "player_id": player_id, "position": position}
        current_game.reset()
        current_game.active    = True
        current_game.player_id = player_id
        if sims is not None:
            current_game.n_sims = max(5, min(sims, 800))
    M.human_game_active.set()
    return {"status": "ok", "player_id": player_id, "fen": chess.Board().fen(),
            "n_sims": current_game.n_sims}


@app.get("/api/queue-status")
async def queue_status(player_id: str):
    """Poll this while queued. Returns 'your_turn' when it's time to play."""
    timed_out = False
    with game_lock:
        if current_game.active and current_game.player_id == player_id:
            return {"status": "your_turn", "fen": current_game.board.fen(),
                    "n_sims": current_game.n_sims}
        # Auto-abandon orphaned games that have had no activity for too long
        if (current_game.active
                and time.time() - current_game.last_activity > GAME_INACTIVITY_TIMEOUT):
            print(f"[queue] Game for player {current_game.player_id} timed out — abandoning")
            current_game.active  = False
            current_game.outcome = "abandoned"
            M.human_game_active.clear()
            timed_out = True
    if timed_out:
        _dequeue_next()
    with _queue_lock:
        for i, entry in enumerate(_game_queue):
            if entry["player_id"] == player_id:
                return {"status": "queued", "position": i + 1,
                        "queue_length": len(_game_queue)}
    return {"status": "not_found"}


@app.post("/api/resign")
async def resign(player_id: str = Query(...)):
    """Cleanly terminate the current game without affecting Elo or training."""
    with game_lock:
        if not current_game.active:
            raise HTTPException(400, "No active game.")
        if player_id != current_game.player_id:
            raise HTTPException(403, "Resign rejected: wrong player_id.")
        current_game.active  = False
        current_game.outcome = "resigned"
    M.human_game_active.clear()
    _dequeue_next()
    return {"status": "ok", "outcome": "resigned"}


@app.post("/api/cancel-queue")
async def cancel_queue(player_id: str = Query(...)):
    """Remove a player from the queue (called when they cancel while waiting)."""
    with _queue_lock:
        before = len(_game_queue)
        _game_queue[:] = [e for e in _game_queue if e["player_id"] != player_id]
        removed = before - len(_game_queue)
    return {"status": "ok", "removed": removed}


class MoveRequest(BaseModel):
    move:      str
    player_id: str


@app.post("/api/move")
async def human_move(req: MoveRequest):
    """Human (white) makes a move."""
    with game_lock:
        if not current_game.active:
            raise HTTPException(400, "No active game.")
        if req.player_id != current_game.player_id:
            raise HTTPException(403, "Move rejected: wrong player_id.")
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
        current_game.traj_w.append((state, policy,
                                    compute_concept_labels(current_game.board),
                                    current_game.board.copy(stack=False)))

        # Advance MCTS tree to match human's move
        if current_game.mcts_root is not None:
            current_game.mcts_root = current_game.mcts_root.children.get(move_to_idx(mv))

        current_game.board.push(mv)
        current_game.move_history.append(mv.uci())
        current_game.last_activity = time.time()

        if current_game.board.is_game_over():
            _finalize_human_game()
            return {"status": "game_over", "outcome": current_game.outcome,
                    "fen": current_game.board.fen()}

        return {"status": "ok", "fen": current_game.board.fen()}


@app.get("/api/ai-move")
async def ai_move(player_id: str = Query(...)):
    """AI (black) calculates and plays its move using MCTS with tree reuse."""
    with game_lock:
        if not current_game.active:
            raise HTTPException(400, "No active game.")
        if player_id != current_game.player_id:
            raise HTTPException(403, "AI move rejected: wrong player_id.")
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
            _, val_t, _ = M.policy_net(state_t)
        # val_t is from current player's (black/AI) perspective; claim if not winning
        if val_t.item() < -0.5:
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

    pv          = mcts.get_pv(new_root, board_snapshot)
    explanation = mcts.explain_move_v2(new_root, board_snapshot, action)

    # Resign if position is hopeless for the AI/Black root player.
    # Only resign if there have been enough moves to form a meaningful position
    if (mcts.root_value(new_root) < RESIGN_THRESHOLD
            and len(board_snapshot.move_stack) >= 10):
        with game_lock:
            if current_game.active and current_game.outcome is None:
                _finalize_human_game(ai_resigned=True)
        return {"move": None, "status": "game_over", "outcome": "white",
                "fen": board_snapshot.fen(), "pv": []}

    mv = idx_to_move(action, board_snapshot)
    if mv is None:
        mv     = random.choice(list(board_snapshot.legal_moves))
        action = mv.from_square * 64 + mv.to_square

    total  = counts.sum()
    policy = counts / total if total > 0 else counts

    # Re-acquire lock to push move and update game state
    with game_lock:
        if not current_game.active or current_game.outcome is not None:
            return {"move": None, "status": "resigned", "outcome": current_game.outcome,
                    "fen": current_game.board.fen(), "pv": []}
        if mv not in current_game.board.legal_moves:
            raise HTTPException(500, "AI selected an illegal move.")

        concepts = compute_concept_labels(board_snapshot)
        current_game.traj_b.append((state, policy, concepts,
                                    board_snapshot.copy(stack=False)))

        # Store the subtree under AI's chosen move for next search
        current_game.mcts_root = new_root.children.get(action)

        current_game.board.push(mv)
        current_game.move_history.append(mv.uci())
        current_game.last_activity = time.time()

        if current_game.board.is_game_over():
            _finalize_human_game()
            return {"move": mv.uci(), "status": "game_over",
                    "outcome": current_game.outcome,
                    "fen": current_game.board.fen(), "pv": pv,
                    "reasoning":  explanation["reasoning"],
                    "candidates": explanation["candidates"]}

        return {"move": mv.uci(), "status": "ok",
                "fen": current_game.board.fen(), "pv": pv,
                "reasoning":  explanation["reasoning"],
                "candidates": explanation["candidates"]}


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
