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
import time

import chess
import numpy as np
import torch
import torch.nn.functional as F

import chess_model as M
from chess_model import save_checkpoint, load_checkpoint
from chess_env import encode, idx_to_move, legal_mask
from chess_mcts import MCTS

# ── Hyperparameters ────────────────────────────────────────────────────
TOTAL_GAMES   = 10_000
REPORT_EVERY  = 10        # games between CLI status lines
BATCH_SIZE    = 512
TRAIN_STEPS   = 5         # gradient steps after each game
MCTS_SIMS     = 100       # simulations per move during self-play
TEMP_THRESHOLD = 30       # moves before temperature → 0
MAX_MOVES     = 150       # half-moves per game cap
SAVE_EVERY    = 50        # games between checkpoint saves


# ── AlphaZero training step ────────────────────────────────────────────
def az_update(net, buf, opt) -> float | None:
    if len(buf) < BATCH_SIZE:
        return None

    states, policies, values = buf.sample(BATCH_SIZE)
    net.train()
    policy_logits, value_pred = net(states)

    policy_loss = -(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    value_loss  = F.mse_loss(value_pred, values)
    loss        = policy_loss + value_loss

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    net.eval()
    return loss.item()


# ── One self-play game ─────────────────────────────────────────────────
def selfplay_game(board: chess.Board, mcts: MCTS):
    """
    Play one game with MCTS.
    Returns list of (state, policy_target, value_target) for training.
    """
    board.reset()
    records = []   # (state_array, policy_array, player_color)
    move_n  = 0

    while not board.is_game_over() and move_n < MAX_MOVES:
        temp = 1.0 if move_n < TEMP_THRESHOLD else 0.0
        action, counts = mcts.get_policy(board, temperature=temp)

        # Normalise visit counts → policy target
        total = counts.sum()
        policy_target = counts / total if total > 0 else counts

        records.append((encode(board), policy_target, board.turn))

        mv = idx_to_move(action, board)
        if mv is None:
            import random
            mv = random.choice(list(board.legal_moves))
        board.push(mv)
        move_n += 1

    # Determine outcome
    outcome = board.outcome()
    if outcome is None:
        result = "D"
        winner = None
    elif outcome.winner == chess.WHITE:
        result, winner = "W", chess.WHITE
    elif outcome.winner == chess.BLACK:
        result, winner = "B", chess.BLACK
    else:
        result, winner = "D", None

    # Build training samples: fill in value_target from each player's perspective
    samples = []
    for state, policy, color in records:
        if winner is None:
            v = 0.0
        else:
            v = 1.0 if color == winner else -1.0
        samples.append((state, policy, v))

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
    print(f"  {'Games':>8}  {'Result':>8}  {'Moves':>7}  {'Loss':>9}  "
          f"{'Buffer':>8}  {'Elo':>6}")
    print("  " + "─" * 58)

    t_game_start = time.time()

    for game_n in range(start, start + TOTAL_GAMES):
        samples, result, n_moves = selfplay_game(board, mcts)

        # Store samples
        for state, policy, value in samples:
            buf.push(state, policy, value)

        # Gradient updates
        last_loss = None
        with M.model_lock:
            for _ in range(TRAIN_STEPS):
                l = az_update(net, buf, opt)
                if l is not None:
                    last_loss = l

        M.total_games    = game_n
        M.selfplay_games += 1

        if game_n % REPORT_EVERY == 0:
            elapsed = time.time() - t_game_start
            gph = REPORT_EVERY / elapsed * 3600
            loss_str = f"{last_loss:.4f}" if last_loss else "  —"
            print(f"  {game_n:>8,}  {result:>8}  {n_moves:>7}  "
                  f"{loss_str:>9}  {len(buf):>8,}  {M.ai_elo:>6.0f}"
                  f"  ({gph:.1f} games/hr)")
            t_game_start = time.time()

        if game_n % SAVE_EVERY == 0:
            save_checkpoint()

    print("  " + "─" * 58)
    print(f"\n  Total games: {M.total_games:,}  |  Elo: {M.ai_elo:.0f}")
    print('\n  "The only winning move is not to play."\n')

    atexit.unregister(save_checkpoint)
    save_checkpoint()
    return net


# ── Entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
