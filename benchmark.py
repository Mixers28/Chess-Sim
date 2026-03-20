"""
benchmark.py — Evaluate the AlphaZero engine against fixed opponents.

Measures win/draw/loss rate over N games against each opponent.
AI plays equal numbers of games as White and Black.

Opponents:
  random     — picks a random legal move
  heuristic  — prefers captures by net material gain, otherwise random
  stockfish  — Stockfish at low skill level (requires stockfish in PATH or --stockfish)

Usage:
    python benchmark.py
    python benchmark.py --games 50 --sims 50 --vs random heuristic
    python benchmark.py --checkpoint checkpoint/model.pt --games 100
    python benchmark.py --stockfish /usr/games/stockfish --skill 2
"""

import argparse
import json
import os
import random
import time

import chess
import chess.engine
import numpy as np
import torch

from chess_env import _PIECE_VALUES, INPUT_PLANES, idx_to_move, move_to_idx
from chess_mcts import MCTS
from chess_net import AlphaZeroNet

MAX_MOVES = 120   # half-move cap; longer than self-play to let games finish


# ── Opponents ─────────────────────────────────────────────────────────────────

class RandomPlayer:
    name = "random"

    def move(self, board: chess.Board) -> chess.Move:
        return random.choice(list(board.legal_moves))


class HeuristicPlayer:
    """
    Picks captures by net material gain (MVV-LVA minus hanging penalty).
    Falls back to random for non-captures.
    Meaningfully stronger than random, much weaker than MCTS.
    """
    name = "heuristic"

    def move(self, board: chess.Board) -> chess.Move:
        best_score = -999
        best_moves = []

        for mv in board.legal_moves:
            score = self._score(board, mv)
            if score > best_score:
                best_score = score
                best_moves = [mv]
            elif score == best_score:
                best_moves.append(mv)

        return random.choice(best_moves)

    @staticmethod
    def _score(board: chess.Board, move: chess.Move) -> int:
        score = 0
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                score += _PIECE_VALUES.get(captured.piece_type, 0)
            mover = board.piece_at(move.from_square)
            if mover and board.is_attacked_by(not board.turn, move.to_square):
                score -= _PIECE_VALUES.get(mover.piece_type, 0)
        return score


class StockfishPlayer:
    name = "stockfish"

    def __init__(self, path: str, skill: int = 1, movetime: float = 0.05):
        self.engine   = chess.engine.SimpleEngine.popen_uci(path)
        self.movetime = movetime
        self.engine.configure({"Skill Level": skill})

    def move(self, board: chess.Board) -> chess.Move:
        result = self.engine.play(board, chess.engine.Limit(time=self.movetime))
        return result.move

    def close(self):
        self.engine.quit()


# ── AI player ─────────────────────────────────────────────────────────────────

def load_ai(checkpoint_path: str, n_sims: int):
    """Load model weights and return (net, mcts, meta_dict)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    az_channels   = ckpt.get("az_channels",    192)
    az_res_blocks = ckpt.get("az_res_blocks",   10)
    az_input_planes = ckpt.get("az_input_planes", INPUT_PLANES)

    if az_input_planes != INPUT_PLANES:
        raise ValueError(
            f"Checkpoint input_planes={az_input_planes}, "
            f"current INPUT_PLANES={INPUT_PLANES}"
        )

    net = AlphaZeroNet(az_channels, az_res_blocks).to(device)
    net.load_state_dict(ckpt["policy_state_dict"], strict=False)
    net.eval()

    mcts = MCTS(net, device, n_sims=n_sims, batch_size=min(n_sims, 32))

    meta = {
        "az_channels":   az_channels,
        "az_res_blocks": az_res_blocks,
        "checkpoint":    checkpoint_path,
    }
    return net, mcts, device, meta


# ── Single game ───────────────────────────────────────────────────────────────

def play_game(ai_color: chess.Color, mcts: MCTS, opponent) -> str:
    """
    Play one game. Returns "win", "loss", or "draw" from the AI's perspective.
    """
    board = chess.Board()
    root  = None

    for _ in range(MAX_MOVES):
        if board.is_game_over():
            break

        if board.turn == ai_color:
            action, _, root = mcts.get_policy(board, temperature=0, root=root)
            mv = idx_to_move(action, board)
            if mv is None:
                mv = random.choice(list(board.legal_moves))
            # Advance MCTS tree to the chosen child
            root = root.children.get(action)
        else:
            mv = opponent.move(board)
            # Advance tree to opponent's move if we have it
            if root is not None:
                opp_idx = move_to_idx(mv)
                root = root.children.get(opp_idx) if opp_idx is not None else None

        board.push(mv)

    outcome = board.outcome()
    if outcome is None:
        return "draw"          # move cap
    if outcome.winner == ai_color:
        return "win"
    if outcome.winner is None:
        return "draw"
    return "loss"


# ── Run one match ─────────────────────────────────────────────────────────────

def run_match(mcts, opponent, n_games: int, verbose: bool = True) -> dict:
    """
    Play n_games against opponent (half as White, half as Black).
    Returns {"wins": W, "draws": D, "losses": L, "games": N}.
    """
    results = {"wins": 0, "draws": 0, "losses": 0, "games": n_games}
    half    = n_games // 2

    for i in range(n_games):
        ai_color = chess.WHITE if i < half else chess.BLACK
        outcome  = play_game(ai_color, mcts, opponent)
        results[outcome + "s"] += 1

        if verbose:
            symbol = {"wins": "W", "draws": "D", "losses": "L"}[outcome + "s"]
            print(f"  {opponent.name:12s} game {i+1:>3}/{n_games}  "
                  f"({'White' if ai_color == chess.WHITE else 'Black'})  {symbol}",
                  flush=True)

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoint/model.pt",
                   help="Path to model.pt checkpoint")
    p.add_argument("--games",      type=int, default=40,
                   help="Games per opponent (split evenly White/Black)")
    p.add_argument("--sims",       type=int, default=50,
                   help="MCTS simulations per AI move")
    p.add_argument("--vs",         nargs="+",
                   choices=["random", "heuristic", "stockfish"],
                   default=["random", "heuristic"],
                   help="Opponents to benchmark against")
    p.add_argument("--stockfish",  default="stockfish",
                   help="Path to Stockfish binary (default: 'stockfish' in PATH)")
    p.add_argument("--skill",      type=int, default=1,
                   help="Stockfish skill level 0–20 (default 1)")
    p.add_argument("--quiet",      action="store_true",
                   help="Suppress per-game output")
    args = p.parse_args()

    # ── Load AI ────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    net, mcts, device, meta = load_ai(args.checkpoint, args.sims)
    print(f"  {meta['az_channels']}ch / {meta['az_res_blocks']}-block  |  "
          f"device: {device}  |  MCTS sims: {args.sims}")

    # Also load stats if available to show Elo
    stats_path = os.path.join(os.path.dirname(args.checkpoint), "stats.pt")
    elo_str = ""
    if os.path.exists(stats_path):
        s = torch.load(stats_path, map_location="cpu", weights_only=True)
        elo_str = f"  Elo: {s.get('ai_elo', '?'):.0f}  |  games: {s.get('total_games', '?'):,}"
        print(elo_str)

    # ── Build opponents ────────────────────────────────────────────────
    opponents = []
    for name in args.vs:
        if name == "random":
            opponents.append(RandomPlayer())
        elif name == "heuristic":
            opponents.append(HeuristicPlayer())
        elif name == "stockfish":
            try:
                opponents.append(StockfishPlayer(args.stockfish, skill=args.skill))
                print(f"  Stockfish: {args.stockfish}  skill={args.skill}")
            except FileNotFoundError:
                print(f"  [warning] Stockfish not found at '{args.stockfish}' — skipping")

    if not opponents:
        print("No opponents available. Exiting.")
        return

    # ── Run matches ────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  {args.games} games per opponent  |  AI plays both colours")
    print(f"{'='*58}\n")

    all_results = {}
    t_start = time.time()

    for opp in opponents:
        print(f"--- vs {opp.name} ---")
        r = run_match(mcts, opp, args.games, verbose=not args.quiet)
        all_results[opp.name] = r
        wp  = r["wins"]   / r["games"] * 100
        dp  = r["draws"]  / r["games"] * 100
        lp  = r["losses"] / r["games"] * 100
        score_pct = (r["wins"] + 0.5 * r["draws"]) / r["games"] * 100
        print(f"  Result: {r['wins']}W / {r['draws']}D / {r['losses']}L  "
              f"({wp:.0f}% / {dp:.0f}% / {lp:.0f}%)  "
              f"score: {score_pct:.1f}%\n")
        if hasattr(opp, "close"):
            opp.close()

    elapsed = time.time() - t_start

    # ── Summary table ──────────────────────────────────────────────────
    print(f"{'='*58}")
    print(f"  {'Opponent':<14} {'W':>4} {'D':>4} {'L':>4}  {'Score%':>7}")
    print(f"  {'-'*14} {'----':>4} {'----':>4} {'----':>4}  {'-------':>7}")
    for name, r in all_results.items():
        score_pct = (r["wins"] + 0.5 * r["draws"]) / r["games"] * 100
        print(f"  {name:<14} {r['wins']:>4} {r['draws']:>4} {r['losses']:>4}  {score_pct:>6.1f}%")
    print(f"{'='*58}")
    print(f"  Total time: {elapsed:.1f}s")

    # ── Save results ───────────────────────────────────────────────────
    os.makedirs("benchmark", exist_ok=True)
    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"benchmark/results_{timestamp}.json"
    payload = {
        "timestamp":  timestamp,
        "checkpoint": args.checkpoint,
        "n_sims":     args.sims,
        "n_games":    args.games,
        "results":    all_results,
    }
    if os.path.exists(stats_path):
        s = torch.load(stats_path, map_location="cpu", weights_only=True)
        payload["ai_elo"]      = float(s.get("ai_elo", 0))
        payload["total_games"] = int(s.get("total_games", 0))

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved → {output_path}\n")


if __name__ == "__main__":
    main()
