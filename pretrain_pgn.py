"""
pretrain_pgn.py — Supervised pre-training on elite Lichess games.

Trains a FRESH 192ch AlphaZeroNet on human game data before self-play begins.
Run ONCE after updating AZ_CHANNELS=192 in chess_model.py.

Workflow (do in order):
    1.  Stop chess_wargames.py
    2.  Archive old weights:
            cp checkpoint/model.pt  checkpoint/model_128ch.pt
            cp checkpoint/stats.pt  checkpoint/stats_128ch.pt
    3.  Edit chess_model.py  →  AZ_CHANNELS = 192
    4.  python pretrain_pgn.py          (saves model.pt when done)
    5.  python chess_wargames.py        (self-play from pre-trained weights)

Why pre-train?
    - 192ch model starts from random weights.
    - 1 epoch over 200k elite games teaches basic tactics/strategy in minutes.
    - Self-play then refines and extends beyond human patterns via MCTS.
    - Dirichlet noise ensures exploration isn't killed by human priors.

Usage:
    python pretrain_pgn.py [--pgn FILE] [--games N] [--epochs N] [--batch N]
"""

import argparse
import os
import random

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F

import chess_model as M
from chess_env import encode, move_to_idx, compute_concept_labels, ACTION_SIZE, INPUT_PLANES

# ── Sampling config ────────────────────────────────────────────────────────────
POSITIONS_PER_GAME = 8   # random positions sampled per game (avoids memory pressure)
SKIP_MOVES         = 5   # skip first N half-moves (heavy opening theory)
MAX_MOVE           = 80  # ignore positions past move 80 (endgame noise)


def result_to_value(result: str, turn: bool) -> float:
    """PGN result string → signed value from current player's perspective."""
    if result == "1-0":
        return  1.0 if turn == chess.WHITE else -1.0
    if result == "0-1":
        return -1.0 if turn == chess.WHITE else  1.0
    return 0.0


def stream_batches(pgn_path: str, max_games: int, batch_size: int):
    """
    Yield (states, actions, values, concepts) batches streamed from PGN.
    Samples POSITIONS_PER_GAME positions per game — no large RAM allocation.
    """
    games_done = 0
    buf_s, buf_a, buf_v, buf_c = [], [], [], []

    with open(pgn_path, errors="replace") as f:
        while games_done < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            result = game.headers.get("Result", "*")
            if result == "*":
                continue

            moves    = list(game.mainline_moves())
            eligible = list(range(SKIP_MOVES, min(len(moves), MAX_MOVE)))
            if not eligible:
                continue

            selected = set(random.sample(eligible, min(POSITIONS_PER_GAME, len(eligible))))

            board = game.board()
            for i, move in enumerate(moves):
                if i >= MAX_MOVE:
                    break
                if i in selected:
                    action = move_to_idx(move, board)
                    if action is not None:
                        buf_s.append(encode(board))
                        buf_a.append(action)
                        buf_v.append(result_to_value(result, board.turn))
                        buf_c.append(compute_concept_labels(board))

                        if len(buf_s) >= batch_size:
                            yield buf_s, buf_a, buf_v, buf_c
                            buf_s, buf_a, buf_v, buf_c = [], [], [], []
                board.push(move)

            games_done += 1
            if games_done % 10_000 == 0:
                print(f"  Parsed {games_done:,} / {max_games:,} games…", flush=True)

    if buf_s:
        yield buf_s, buf_a, buf_v, buf_c


def pretrain(pgn_path: str, max_games: int, batch_size: int, epochs: int) -> None:
    net = M.policy_net
    opt = M.optimizer
    dev = M.device

    approx_positions = max_games * POSITIONS_PER_GAME
    approx_steps     = approx_positions // batch_size

    print(f"\n{'='*70}")
    print(f"  Pre-training {M.AZ_CHANNELS}ch / {M.AZ_RES_BLOCKS}-block AlphaZeroNet")
    print(f"  Device : {str(dev).upper()}")
    print(f"  PGN    : {pgn_path}")
    print(f"  Games  : {max_games:,}  |  ~{approx_positions:,} positions  |  "
          f"~{approx_steps:,} steps/epoch")
    print(f"  Epochs : {epochs}  |  Batch : {batch_size}")
    print(f"{'='*70}\n")

    for epoch in range(1, epochs + 1):
        print(f"── Epoch {epoch}/{epochs} {'─'*55}")
        net.train()

        step = 0
        totals = {"loss": 0.0, "p": 0.0, "v": 0.0, "c": 0.0}

        for states, actions, values, concepts in stream_batches(pgn_path, max_games, batch_size):
            s_t = torch.tensor(np.array(states),  dtype=torch.float32).to(dev)
            v_t = torch.tensor(values,             dtype=torch.float32).to(dev)
            c_t = torch.tensor(np.array(concepts), dtype=torch.float32).to(dev)

            # One-hot policy target from the human move played
            p_t = torch.zeros(len(states), ACTION_SIZE, device=dev)
            for j, a in enumerate(actions):
                p_t[j, a] = 1.0

            opt.zero_grad()
            p_pred, v_pred, c_pred = net(s_t)

            p_loss = -(p_t * F.log_softmax(p_pred, dim=1)).sum(dim=1).mean()
            v_loss = F.mse_loss(v_pred.squeeze(1), v_t)
            c_loss = F.mse_loss(c_pred, c_t)
            loss   = p_loss + 0.5 * v_loss + 0.1 * c_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            step += 1
            totals["loss"] += loss.item()
            totals["p"]    += p_loss.item()
            totals["v"]    += v_loss.item()
            totals["c"]    += c_loss.item()

            if step % 500 == 0:
                avg = {k: v / step for k, v in totals.items()}
                print(f"  Step {step:>6,} | loss {avg['loss']:.3f}  "
                      f"(p:{avg['p']:.3f}  v:{avg['v']:.3f}  c:{avg['c']:.3f})",
                      flush=True)

        avg = {k: v / max(step, 1) for k, v in totals.items()}
        print(f"\n  Epoch {epoch} complete — {step:,} steps | "
              f"loss {avg['loss']:.3f}  "
              f"(p:{avg['p']:.3f}  v:{avg['v']:.3f}  c:{avg['c']:.3f})\n")

    # Save pre-trained model
    os.makedirs(M.CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "policy_state_dict": net.state_dict(),
        "optimizer_state":   opt.state_dict(),
        "scheduler_state":   M.scheduler.state_dict(),
        "az_channels":       M.AZ_CHANNELS,
        "az_res_blocks":     M.AZ_RES_BLOCKS,
        "az_input_planes":   INPUT_PLANES,
    }, M.MODEL_PATH)
    print(f"[pretrain] Saved → {M.MODEL_PATH}")
    print("[pretrain] Run chess_wargames.py to begin self-play.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pgn",    default="lichess_elite_2025-11.pgn")
    p.add_argument("--games",  type=int, default=200_000,
                   help="Number of PGN games to train on (default 200k of 280k available)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch",  type=int, default=512)
    args = p.parse_args()

    pretrain(args.pgn, args.games, args.batch, args.epochs)
