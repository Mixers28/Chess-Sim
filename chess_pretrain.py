"""
chess_pretrain.py — Supervised pre-training from Lichess PGN data.

Converts human game positions into training samples matching the replay
buffer schema and bootstraps the network before self-play begins.
Solves the cold-start problem: the network learns real chess patterns
(policy) and outcome prediction (value) from strong human games, so
self-play immediately produces meaningful signal.

Usage:
    python chess_pretrain.py lichess_elite_2025-11.pgn
    python chess_pretrain.py lichess_elite_2025-11.pgn --max-games 50000
    python chess_pretrain.py lichess_elite_2025-11.pgn --max-games 20000 --min-elo 2400

Each position in each game becomes one training sample:
  state   — 19×8×8 board encoding
  policy  — one-hot on the move actually played (supervised imitation)
  value   — game outcome from that player's perspective (+1/−0.15/−1)
  concepts — auto-labelled strategic scores
"""

import argparse
import time

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F

import chess_model as M
from chess_model import load_checkpoint, save_checkpoint
from chess_env import (encode, move_to_idx, mirror_sample,
                       compute_concept_labels, ACTION_SIZE, _PIECE_VALUES)
from chess_wargames import az_update

# ── Config ────────────────────────────────────────────────────────────
TRAIN_EVERY  = 200    # run gradient updates after every N games parsed
TRAIN_STEPS  = 10     # gradient steps per training interval
REPORT_EVERY = 1_000  # print progress every N games
DRAW_VALUE   = -0.15  # consistent with self-play


# ── PGN parsing ───────────────────────────────────────────────────────
def _result_to_winner(result_str: str):
    """Return chess.WHITE, chess.BLACK, or None (draw/unknown)."""
    if result_str == "1-0":
        return chess.WHITE
    if result_str == "0-1":
        return chess.BLACK
    return None   # draw or *


def games_from_pgn(pgn_path: str, max_games: int, min_elo: int):
    """
    Generator — yields parsed game dicts:
      winner  : chess.WHITE | chess.BLACK | None
      samples : list of (state, policy, value, concepts, board_copy)
    Skips games with unknown results, missing Elo, or sub-threshold Elo.
    """
    with open(pgn_path, encoding="utf-8", errors="ignore") as f:
        parsed = 0
        while parsed < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Filter
            try:
                w_elo = int(game.headers.get("WhiteElo", 0))
                b_elo = int(game.headers.get("BlackElo", 0))
            except ValueError:
                continue
            if w_elo < min_elo or b_elo < min_elo:
                continue

            result = game.headers.get("Result", "*")
            winner = _result_to_winner(result)
            if result == "*":
                continue   # unfinished game

            # Walk moves
            board   = game.board()
            samples = []
            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    break   # corrupt game

                idx = move_to_idx(move)
                if idx >= ACTION_SIZE:
                    board.push(move)
                    continue   # shouldn't happen, but guard anyway

                policy        = np.zeros(ACTION_SIZE, dtype=np.float32)
                policy[idx]   = 1.0
                v             = (DRAW_VALUE if winner is None
                                 else (1.0 if board.turn == winner else -1.0))
                concepts      = compute_concept_labels(board)
                board_copy    = board.copy(stack=False)
                samples.append((encode(board), policy, v, concepts, board_copy))
                board.push(move)

            if samples:
                yield samples
                parsed += 1


# ── Main ──────────────────────────────────────────────────────────────
def pretrain(pgn_path: str, max_games: int, min_elo: int):
    loaded = load_checkpoint()
    if not loaded:
        print("[pretrain] No checkpoint found — starting from scratch.")

    net = M.policy_net
    buf = M.replay_buf
    opt = M.optimizer
    net.eval()

    print()
    print("=" * 72)
    print("  P R E - T R A I N  —  Supervised learning from Lichess PGN")
    print(f"  Device : {str(M.device).upper()}")
    print(f"  Network: {M.AZ_RES_BLOCKS} res blocks | {M.AZ_CHANNELS} channels | "
          f"{M.n_params:,} params")
    print(f"  Source : {pgn_path}")
    print(f"  Filter : min Elo {min_elo} | max games {max_games:,}")
    print("=" * 72)
    print()
    print(f"  {'Games':>8}  {'Samples':>10}  {'Buffer':>8}  "
          f"{'Loss':>32}  {'Rate':>12}")
    print("  " + "─" * 80)

    total_samples = 0
    last_losses   = None
    t0            = time.time()
    t_report      = time.time()

    for game_n, samples in enumerate(
            games_from_pgn(pgn_path, max_games, min_elo), start=1):

        # Push original + horizontally mirrored samples
        for state, policy, value, concepts, board_copy in samples:
            buf.push(state, policy, value, concepts)
            ms, mp, mv = mirror_sample(state, policy, value)
            mirrored_concepts = compute_concept_labels(
                board_copy.transform(chess.flip_horizontal)
            )
            buf.push(ms, mp, mv, mirrored_concepts)
            total_samples += 2

        # Periodic training
        if game_n % TRAIN_EVERY == 0:
            with M.model_lock:
                for _ in range(TRAIN_STEPS):
                    l = az_update(net, buf, opt, M.scheduler)
                    if l is not None:
                        last_losses = l

        # Periodic reporting
        if game_n % REPORT_EVERY == 0:
            elapsed   = time.time() - t_report
            games_per_hr = REPORT_EVERY / elapsed * 3600
            if last_losses:
                total, pol, val, con = last_losses
                loss_str = f"{total:.3f} (p:{pol:.3f} v:{val:.3f} c:{con:.3f})"
            else:
                loss_str = "— (buffer filling)"
            print(f"  {game_n:>8,}  {total_samples:>10,}  {len(buf):>8,}  "
                  f"{loss_str:<32}  {games_per_hr:>8.0f} g/hr")
            t_report = time.time()

    # Final training pass
    print("\n  Final training pass...")
    with M.model_lock:
        for step in range(50):
            l = az_update(net, buf, opt, M.scheduler)
            if l is not None:
                last_losses = l
            if (step + 1) % 10 == 0 and last_losses:
                total, pol, val, con = last_losses
                print(f"    step {step+1:>3}  "
                      f"loss {total:.3f} (p:{pol:.3f} v:{val:.3f} c:{con:.3f})")

    save_checkpoint()
    elapsed_total = time.time() - t0
    print()
    print(f"  Done. {game_n:,} games | {total_samples:,} samples "
          f"| {elapsed_total/60:.1f} min")
    print(f"  Checkpoint saved → {M.CHECKPOINT_PATH}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-train chess network on Lichess PGN data."
    )
    parser.add_argument("pgn", help="Path to .pgn file")
    parser.add_argument("--max-games", type=int, default=50_000,
                        help="Max games to parse (default: 50000)")
    parser.add_argument("--min-elo",   type=int, default=2200,
                        help="Minimum Elo for both players (default: 2200)")
    args = parser.parse_args()

    pretrain(args.pgn, args.max_games, args.min_elo)
