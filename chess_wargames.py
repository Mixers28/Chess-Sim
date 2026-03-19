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
import multiprocessing as _mp
import random
import time
from concurrent.futures import ProcessPoolExecutor

import chess
import numpy as np
import torch
import torch.nn.functional as F

import chess_model as M
from chess_model import save_checkpoint, load_checkpoint
from chess_env import encode, idx_to_move, move_to_idx, legal_mask, mirror_sample, ACTION_SIZE, compute_concept_labels, _PIECE_VALUES
from chess_mcts import MCTS
from chess_net import AlphaZeroNet

# ── Hyperparameters ────────────────────────────────────────────────────
TOTAL_GAMES  = 10_000
REPORT_EVERY = 10         # games between CLI status lines
BATCH_SIZE   = 512
TRAIN_STEPS  = 5          # gradient steps after each game
MCTS_SIMS    = 50         # simulations per move during self-play
MAX_MOVES    = 60         # half-moves per game cap
MATERIAL_WIN = 6          # material advantage (in pawns) treated as decisive win
SAVE_EVERY   = 50         # games between checkpoint saves

# Temperature decays exponentially: τ(n) = max(0.05, exp(-n / TEMP_DECAY))
# At move 15: ~0.37  |  move 30: ~0.14  |  move 45: ~0.05 (floor)
TEMP_DECAY   = 20.0

# Resign: if MCTS root value stays below this for RESIGN_CONSECUTIVE moves, resign.
# Won't trigger until the value head learns to produce values near ±1.
RESIGN_THRESHOLD   = -0.70
RESIGN_CONSECUTIVE = 3
N_WORKERS          = 4 if torch.cuda.is_available() else 1  # parallel self-play workers


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
    # Mask out legacy zero-labeled samples (saved before concept tracking was added)
    concept_mask = concept_labels.sum(dim=1) > 0
    if concept_mask.any():
        concept_loss = F.mse_loss(concepts_pred[concept_mask], concept_labels[concept_mask])
    else:
        concept_loss = torch.zeros(1, device=concept_labels.device)
    # λ_value=2.0 amplifies value gradient to prevent collapse; λ_concept=1.0 drives concept learning
    loss = policy_loss + 2.0 * value_loss + 1.0 * concept_loss

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
    records        = []    # (state_array, policy_array, player_color, concept_labels, board_copy)
    move_n         = 0
    consecutive_low = 0   # consecutive moves where root value < RESIGN_THRESHOLD
    resigned       = False

    while not board.is_game_over() and move_n < MAX_MOVES:
        # Try book move first (early game diversity)
        book_mv = _book_move(board)
        if book_mv is not None:
            policy_target = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy_target[move_to_idx(book_mv)] = 1.0
            records.append((encode(board), policy_target, board.turn,
                            compute_concept_labels(board), board.copy(stack=False)))
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

        records.append((encode(board), policy_target, board.turn,
                        compute_concept_labels(board), board.copy(stack=False)))

        mv = idx_to_move(action, board)
        if mv is None:
            mv = random.choice(list(board.legal_moves))
        board.push(mv)
        move_n += 1

        if position_cb is not None:
            position_cb(board.fen(), mv.uci(), move_n)

        # Material termination: queen-level imbalance treated as decisive
        mat_w = sum(_PIECE_VALUES[p.piece_type]
                    for p in board.piece_map().values() if p.color == chess.WHITE)
        mat_b = sum(_PIECE_VALUES[p.piece_type]
                    for p in board.piece_map().values() if p.color == chess.BLACK)
        if abs(mat_w - mat_b) >= MATERIAL_WIN:
            winner = chess.WHITE if mat_w > mat_b else chess.BLACK
            result = "W" if winner == chess.WHITE else "B"
            samples = []
            for state, policy, color, concepts, board_copy in records:
                v = 1.0 if color == winner else -1.0
                samples.append((state, policy, v, concepts, board_copy))
            return samples, result, move_n

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
    for state, policy, color, concepts, board_copy in records:
        v = -0.15 if winner is None else (1.0 if color == winner else -1.0)
        samples.append((state, policy, v, concepts, board_copy))

    return samples, result, move_n


# ── Multiprocessing worker ─────────────────────────────────────────────
# Module-level globals so each worker process initialises once
_w_net  = None
_w_mcts = None
_w_dev  = None


def _worker_init(az_channels, az_res_blocks):
    """Run once per worker process: create model + MCTS on GPU."""
    global _w_net, _w_mcts, _w_dev
    import torch
    _w_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _w_net = AlphaZeroNet(az_channels, az_res_blocks).to(_w_dev)
    _w_net.eval()
    _w_mcts = MCTS(_w_net, _w_dev, n_sims=MCTS_SIMS)


def _play_game(state_dict_cpu):
    """Play one game with updated weights. Returns processed samples (no Board objects)."""
    global _w_net, _w_mcts, _w_dev
    import torch
    _w_net.load_state_dict({k: v.to(_w_dev) for k, v in state_dict_cpu.items()})
    _w_net.eval()
    raw_samples, result, n_moves = selfplay_game(chess.Board(), _w_mcts)
    # Pre-compute mirrored samples here — avoids pickling Board objects over IPC
    processed = []
    for state, policy, value, concepts, board_copy in raw_samples:
        processed.append((state, policy, value, concepts))
        ms, mp_arr, mv = mirror_sample(state, policy, value)
        mc = compute_concept_labels(board_copy.transform(chess.flip_horizontal))
        processed.append((ms, mp_arr, mv, mc))
    return processed, result, n_moves


# ── Training loop ──────────────────────────────────────────────────────
def train():
    if not load_checkpoint():
        print("[wargames] Starting fresh — no checkpoint found.")

    atexit.register(save_checkpoint)

    net  = M.policy_net
    buf  = M.replay_buf
    opt  = M.optimizer
    dev  = M.device

    start = M.total_games + 1

    print()
    print("=" * 72)
    print("  W A R G A M E S  —  AlphaZero Chess (MCTS + Residual Network)")
    print(f"  Device: {str(dev).upper()}  |  "
          f"{M.AZ_RES_BLOCKS} res blocks  |  "
          f"{M.AZ_CHANNELS} channels  |  "
          f"{M.n_params:,} params  |  "
          f"{N_WORKERS} workers")
    if dev.type == "cpu":
        print("  ⚠  GPU strongly recommended — CPU self-play will be slow")
    print("=" * 72)
    print()
    print(f"  {'Games':>8}  {'Res':>6}  {'Mv':>5}  "
          f"{'Loss (total  policy  value  concept)':<36}  "
          f"{'Buffer':>8}  {'Elo':>6}")
    print("  " + "─" * 82)

    t_round_start = time.time()
    game_n = start

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=_worker_init,
        initargs=(M.AZ_CHANNELS, M.AZ_RES_BLOCKS),
        mp_context=_mp.get_context("spawn"),
    ) as executor:
        while game_n < start + TOTAL_GAMES:
            # Serialize current weights once, send to all workers
            state_dict_cpu = {k: v.cpu() for k, v in net.state_dict().items()}
            futures = [executor.submit(_play_game, state_dict_cpu)
                       for _ in range(N_WORKERS)]
            round_results = [f.result() for f in futures]

            # Push pre-processed samples (mirroring done in worker, no Board objects)
            for processed_samples, _, _ in round_results:
                for sample in processed_samples:
                    buf.push(*sample)

            # Gradient updates
            last_losses = None
            with M.model_lock:
                net.train()
                for _ in range(TRAIN_STEPS):
                    l = az_update(net, buf, opt, M.scheduler)
                    if l is not None:
                        last_losses = l
                net.eval()

            M.total_games    = game_n + N_WORKERS - 1
            M.selfplay_games += N_WORKERS

            if M.total_games % REPORT_EVERY < N_WORKERS:
                elapsed = time.time() - t_round_start
                gph = (REPORT_EVERY / elapsed) * 3600
                # Show summary of round results (W/B/D counts, avg moves)
                outcomes = [r[1] for r in round_results]
                avg_mv   = sum(r[2] for r in round_results) // N_WORKERS
                res_str  = "/".join(outcomes)
                if last_losses:
                    total, pol, val, con = last_losses
                    loss_str = f"{total:.3f} (p:{pol:.3f} v:{val:.3f} c:{con:.3f})"
                else:
                    loss_str = "—"
                print(f"  {M.total_games:>8,}  {res_str:>6}  {avg_mv:>5}  "
                      f"{loss_str:<36}  {len(buf):>8,}  {M.ai_elo:>6.0f}"
                      f"  ({gph:.1f}/hr)")
                t_round_start = time.time()

            if M.total_games % SAVE_EVERY < N_WORKERS:
                save_checkpoint()

            game_n += N_WORKERS

    print("  " + "─" * 82)
    print(f"\n  Total games: {M.total_games:,}  |  Elo: {M.ai_elo:.0f}")
    print('\n  "The only winning move is not to play."\n')

    atexit.unregister(save_checkpoint)
    save_checkpoint()
    return net


# ── Entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    _mp.set_start_method("spawn", force=True)
    train()
