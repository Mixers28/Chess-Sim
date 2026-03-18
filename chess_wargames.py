"""
WARGAMES — Chess AlphaZero Edition

Self-play training loop using MCTS + AlphaZero architecture.
Learns purely from playing itself — no external chess knowledge.

Training data per game:
  For each board state s visited during a game:
    state         — 13×8×8 board encoding
    policy_target — MCTS visit-count distribution (normalised to sum 1)
    value_target  — game outcome from that player's perspective (+1/−1/0)
"""

import atexit
import math
import random
import time

import chess
import numpy as np
import torch
import torch.nn.functional as F

import chess_model as M
from chess_model import save_checkpoint, load_checkpoint
from chess_env import encode, idx_to_move, move_to_idx, legal_mask, mirror_sample, ACTION_SIZE, compute_concept_labels
from chess_mcts import MCTS

# ── Hyperparameters ────────────────────────────────────────────────────
TOTAL_GAMES  = 10_000
REPORT_EVERY = 10         # games between CLI status lines
BATCH_SIZE   = 512
TRAIN_STEPS  = 15         # gradient steps after each game
MCTS_SIMS    = 200        # simulations per move during self-play
MAX_MOVES    = 256        # half-moves per game cap
SAVE_EVERY   = 50         # games between checkpoint saves

# Temperature decays exponentially: τ(n) = max(0.05, exp(-n / TEMP_DECAY))
# At move 15: ~0.37  |  move 30: ~0.14  |  move 45: ~0.05 (floor)
TEMP_DECAY   = 20.0

# Resign: if MCTS root value stays below this for RESIGN_CONSECUTIVE moves, resign.
# Won't trigger until the value head learns to produce values near ±1.
RESIGN_THRESHOLD   = -0.9
RESIGN_CONSECUTIVE = 5


# ── Opening book ───────────────────────────────────────────────────────
# Key: first two FEN fields (piece placement + side to move).
# Value: list of UCI moves to pick uniformly at random (duplicates = weight).
_OPENING_BOOK: dict[str, list[str]] = {
    # Move 1 — white (weight e4/d4 higher)
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w": [
        "e2e4", "e2e4", "e2e4",
        "d2d4", "d2d4",
        "c2c4", "g1f3",
    ],
    # Move 1 — black responses to 1.e4
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b": [
        "e7e5", "e7e5", "c7c5", "c7c5",
        "e7e6", "c7c6", "d7d5", "g8f6",
    ],
    # Move 1 — black responses to 1.d4
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b": [
        "d7d5", "d7d5", "g8f6", "g8f6",
        "e7e6", "f7f5",
    ],
    # Move 1 — black responses to 1.c4
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b": [
        "e7e5", "c7c5", "g8f6", "e7e6",
    ],
    # Move 1 — black responses to 1.Nf3
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b": [
        "d7d5", "g8f6", "c7c5", "e7e6",
    ],
    # After 1.e4 e5 — white move 2
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w": [
        "g1f3", "g1f3", "b1c3", "f2f4",
    ],
    # After 1.e4 c5 (Sicilian) — white move 2
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w": [
        "g1f3", "g1f3", "b1c3", "f2f4",
    ],
    # After 1.e4 e6 (French) — white move 2
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w": [
        "d2d4", "d2d4", "g1f3",
    ],
    # After 1.e4 c6 (Caro-Kann) — white move 2
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w": [
        "d2d4", "d2d4", "b1c3", "g1f3",
    ],
    # After 1.e4 d5 (Scandinavian) — white move 2
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w": [
        "e4d5", "e4d5", "b1c3", "e4e5",
    ],
    # After 1.d4 d5 — white move 2
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w": [
        "c2c4", "c2c4", "g1f3", "e2e3",
    ],
    # After 1.d4 Nf6 — white move 2
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w": [
        "c2c4", "c2c4", "g1f3", "c1g5",
    ],
    # After 1.e4 e5 2.Nf3 — black move 2
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b": [
        "b8c6", "b8c6", "g8f6", "d7d6",
    ],
    # After 1.d4 d5 2.c4 (Queen's Gambit) — black move 2
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b": [
        "e7e6", "e7e6", "c7c6", "d5c4",
    ],
}


def _book_key(board: chess.Board) -> str:
    parts = board.fen().split()
    return parts[0] + " " + parts[1]


def _book_move(board: chess.Board) -> chess.Move | None:
    """Return a random legal book move for the current position, or None."""
    if board.fullmove_number > 8:
        return None
    candidates = [
        chess.Move.from_uci(uci)
        for uci in _OPENING_BOOK.get(_book_key(board), [])
    ]
    legal = [mv for mv in candidates if mv in board.legal_moves]
    return random.choice(legal) if legal else None


# ── AlphaZero training step ────────────────────────────────────────────
def az_update(net, buf, opt, sched=None) -> tuple | None:
    """
    One gradient step. Returns (total, policy, value, concept) loss floats,
    or None if the buffer is too small to sample a batch.
    """
    if len(buf) < BATCH_SIZE:
        return None

    states, policies, values, concept_labels = buf.sample(BATCH_SIZE)
    net.train()
    policy_logits, value_pred, concepts_pred = net(states)

    policy_loss  = -(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    value_loss   = F.mse_loss(value_pred, values)
    concept_loss = F.mse_loss(concepts_pred, concept_labels)
    # λ_value=0.5 prevents value head dominating; λ_concept=0.1 auxiliary signal
    loss = policy_loss + 0.5 * value_loss + 0.1 * concept_loss

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    if sched is not None:
        sched.step()
    net.eval()
    return loss.item(), policy_loss.item(), value_loss.item(), concept_loss.item()


# ── One self-play game ─────────────────────────────────────────────────
def selfplay_game(board: chess.Board, mcts: MCTS, position_cb=None):
    """
    Play one game with MCTS.
    Returns list of (state, policy_target, value_target) for training.

    position_cb: optional callable(fen, move_uci, move_n) called after each move,
                 used by the web server to broadcast live positions.
    """
    board.reset()
    records        = []    # (state_array, policy_array, player_color, concept_labels)
    move_n         = 0
    consecutive_low = 0   # consecutive moves where root value < RESIGN_THRESHOLD
    resigned       = False

    while not board.is_game_over() and move_n < MAX_MOVES:
        # Try book move first (early game diversity)
        book_mv = _book_move(board)
        if book_mv is not None:
            policy_target = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy_target[move_to_idx(book_mv)] = 1.0
            records.append((encode(board), policy_target, board.turn, compute_concept_labels(board)))
            board.push(book_mv)
            move_n += 1
            if position_cb is not None:
                position_cb(board.fen(), book_mv.uci(), move_n)
            continue

        # Smooth exponential temperature decay: high early, low late
        temp = max(0.05, math.exp(-move_n / TEMP_DECAY))
        action, counts, root = mcts.get_policy(board, temperature=temp, add_noise=True)

        # Compute root value from MCTS tree (weighted avg of children Q, negated)
        total_n = sum(c.N for c in root.children.values())
        if total_n > 0:
            root_value = -sum(c.Q * c.N for c in root.children.values()) / total_n
        else:
            root_value = 0.0

        # Resign check — activates naturally once value head produces values near ±1
        if root_value < RESIGN_THRESHOLD:
            consecutive_low += 1
            if consecutive_low >= RESIGN_CONSECUTIVE:
                resigned = True
                break
        else:
            consecutive_low = 0

        # Normalise visit counts → policy target
        total = counts.sum()
        policy_target = counts / total if total > 0 else counts

        records.append((encode(board), policy_target, board.turn, compute_concept_labels(board)))

        mv = idx_to_move(action, board)
        if mv is None:
            mv = random.choice(list(board.legal_moves))
        board.push(mv)
        move_n += 1

        if position_cb is not None:
            position_cb(board.fen(), mv.uci(), move_n)

    # Determine outcome
    if resigned:
        # Current player to move lost (they gave up)
        result = "B" if board.turn == chess.WHITE else "W"
        winner = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
    else:
        outcome = board.outcome()
        if outcome is None:
            result, winner = "D", None   # move cap
        elif outcome.winner == chess.WHITE:
            result, winner = "W", chess.WHITE
        elif outcome.winner == chess.BLACK:
            result, winner = "B", chess.BLACK
        else:
            result, winner = "D", None

    # Build training samples: fill in value_target from each player's perspective
    samples = []
    for state, policy, color, concepts in records:
        v = 0.0 if winner is None else (1.0 if color == winner else -1.0)
        samples.append((state, policy, v, concepts))

    return samples, result, move_n


# ── Training loop ──────────────────────────────────────────────────────
def train():
    if not load_checkpoint():
        print("[wargames] Starting fresh — no checkpoint found.")

    atexit.register(save_checkpoint)

    net  = M.policy_net
    buf  = M.replay_buf
    opt  = M.optimizer
    dev  = M.device

    net.eval()
    mcts = MCTS(net, dev, n_sims=MCTS_SIMS)

    start = M.total_games + 1
    board = chess.Board()

    print()
    print("=" * 72)
    print("  W A R G A M E S  —  AlphaZero Chess (MCTS + Residual Network)")
    print(f"  Device: {str(dev).upper()}  |  "
          f"{M.AZ_RES_BLOCKS} res blocks  |  "
          f"{M.AZ_CHANNELS} channels  |  "
          f"{M.n_params:,} params")
    if dev.type == "cpu":
        print("  ⚠  GPU strongly recommended — CPU self-play will be slow")
    print("=" * 72)
    print()
    print(f"  {'Games':>8}  {'Res':>6}  {'Mv':>5}  "
          f"{'Loss (total  policy  value  concept)':<36}  "
          f"{'Buffer':>8}  {'Elo':>6}")
    print("  " + "─" * 82)

    t_game_start = time.time()

    for game_n in range(start, start + TOTAL_GAMES):
        samples, result, n_moves = selfplay_game(board, mcts)

        # Store samples + mirrored augmentations (free 2× data)
        # Concepts are board-state labels — unchanged by horizontal mirror
        for state, policy, value, concepts in samples:
            buf.push(state, policy, value, concepts)
            buf.push(*mirror_sample(state, policy, value), concepts)

        # Gradient updates — accumulate split losses across steps for logging
        last_losses = None
        with M.model_lock:
            for _ in range(TRAIN_STEPS):
                l = az_update(net, buf, opt, M.scheduler)
                if l is not None:
                    last_losses = l

        M.total_games    = game_n
        M.selfplay_games += 1

        if game_n % REPORT_EVERY == 0:
            elapsed = time.time() - t_game_start
            gph = REPORT_EVERY / elapsed * 3600
            if last_losses:
                total, pol, val, con = last_losses
                loss_str = f"{total:.3f} (p:{pol:.3f} v:{val:.3f} c:{con:.3f})"
            else:
                loss_str = "—"
            print(f"  {game_n:>8,}  {result:>6}  {n_moves:>5}  "
                  f"{loss_str:<36}  {len(buf):>8,}  {M.ai_elo:>6.0f}"
                  f"  ({gph:.1f}/hr)")
            t_game_start = time.time()

        if game_n % SAVE_EVERY == 0:
            save_checkpoint()

    print("  " + "─" * 82)
    print(f"\n  Total games: {M.total_games:,}  |  Elo: {M.ai_elo:.0f}")
    print('\n  "The only winning move is not to play."\n')

    atexit.unregister(save_checkpoint)
    save_checkpoint()
    return net


# ── Entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
