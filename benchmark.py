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
    python benchmark.py --games 100 --sims 100 --vs heuristic stockfish \
        --stockfish /usr/games/stockfish --skill 1 --stockfish-movetime 0.05
"""

import argparse
import json
import math
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

DEFAULT_MAX_MOVES = 120   # half-move cap; longer than self-play
RESULT_SCORES = {"win": 1.0, "draw": 0.5, "loss": 0.0}


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

    def __init__(
        self,
        path: str,
        skill: int = 1,
        movetime: float = 0.05,
        threads: int = 1,
        hash_mb: int = 64,
    ):
        self.engine   = chess.engine.SimpleEngine.popen_uci(path)
        self.movetime = movetime
        options = {}
        if "Skill Level" in self.engine.options:
            options["Skill Level"] = skill
        if "Threads" in self.engine.options:
            options["Threads"] = threads
        if "Hash" in self.engine.options:
            options["Hash"] = hash_mb
        if options:
            self.engine.configure(options)

    def move(self, board: chess.Board) -> chess.Move:
        result = self.engine.play(board, chess.engine.Limit(time=self.movetime))
        return result.move

    def close(self):
        self.engine.quit()


# ── AI player ─────────────────────────────────────────────────────────────────

def load_ai(checkpoint_path: str, n_sims: int):
    """Load model weights and return (net, mcts, meta_dict)."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

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
        "az_channels":    az_channels,
        "az_res_blocks":  az_res_blocks,
        "checkpoint":     checkpoint_path,
        "training_games": int(ckpt.get("training_games", 0)),
        "selfplay_games": int(ckpt.get("selfplay_games", 0)),
        "model_saved_at": str(ckpt.get("model_saved_at", "")),
        "model_version":  str(ckpt.get("model_version", "unversioned")),
    }
    return net, mcts, device, meta


# ── Single game ───────────────────────────────────────────────────────────────

def play_game(
    ai_color: chess.Color,
    mcts: MCTS,
    opponent,
    max_moves: int = DEFAULT_MAX_MOVES,
    material_adjudication: float = 0.0,
    return_details: bool = False,
):
    """
    Play one game. Returns "win", "loss", or "draw" from the AI's perspective.
    """
    board = chess.Board()
    root  = None

    moves_played = 0
    for _ in range(max_moves):
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
        moves_played += 1

    outcome = board.outcome()
    material_white = sum(
        _PIECE_VALUES.get(piece.piece_type, 0)
        for piece in board.piece_map().values()
        if piece.color == chess.WHITE
    )
    material_black = sum(
        _PIECE_VALUES.get(piece.piece_type, 0)
        for piece in board.piece_map().values()
        if piece.color == chess.BLACK
    )
    material_for_ai = (
        material_white - material_black
        if ai_color == chess.WHITE
        else material_black - material_white
    )

    if outcome is None:
        if material_adjudication > 0 and material_for_ai >= material_adjudication:
            result = "win"
            termination = "move_cap_material"
        elif material_adjudication > 0 and material_for_ai <= -material_adjudication:
            result = "loss"
            termination = "move_cap_material"
        else:
            result = "draw"
            termination = "move_cap_draw"
    elif outcome.winner == ai_color:
        result = "win"
        termination = outcome.termination.name.lower()
    elif outcome.winner is None:
        result = "draw"
        termination = outcome.termination.name.lower()
    else:
        result = "loss"
        termination = outcome.termination.name.lower()

    details = {
        "outcome": result,
        "termination": termination,
        "moves": moves_played,
        "material_balance": round(float(material_for_ai), 2),
        "final_fen": board.fen(),
    }
    return details if return_details else result


# ── Run one match ─────────────────────────────────────────────────────────────

def run_match(
    mcts,
    opponent,
    n_games: int,
    verbose: bool = True,
    max_moves: int = DEFAULT_MAX_MOVES,
    material_adjudication: float = 0.0,
    existing_results: dict | None = None,
    progress_callback=None,
    seed: int | None = None,
) -> dict:
    """
    Play n_games against opponent (half as White, half as Black).
    Returns {"wins": W, "draws": D, "losses": L, "games": N}.
    """
    results = existing_results or {
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "games": n_games,
        "outcomes": [],
        "game_records": [],
        "terminations": {},
        "by_color": {
            "white": {"wins": 0, "draws": 0, "losses": 0, "games": 0},
            "black": {"wins": 0, "draws": 0, "losses": 0, "games": 0},
        },
    }
    results.setdefault("game_records", [])
    results.setdefault("terminations", {})
    completed = len(results.get("outcomes", []))

    for i in range(completed, n_games):
        game_seed = seed + i if seed is not None else None
        if game_seed is not None:
            random.seed(game_seed)
            np.random.seed(game_seed)
            torch.manual_seed(game_seed)
        ai_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        game_result = play_game(
            ai_color,
            mcts,
            opponent,
            max_moves=max_moves,
            material_adjudication=material_adjudication,
            return_details=True,
        )
        outcome = game_result["outcome"]
        key      = {"win": "wins", "draw": "draws", "loss": "losses"}[outcome]
        results[key] += 1
        results["outcomes"].append(outcome)
        results["game_records"].append({
            "game": i + 1,
            "seed": game_seed,
            "color": "white" if ai_color == chess.WHITE else "black",
            **game_result,
        })
        termination = game_result["termination"]
        results["terminations"][termination] = (
            results["terminations"].get(termination, 0) + 1
        )

        color_key = "white" if ai_color == chess.WHITE else "black"
        results["by_color"][color_key][key] += 1
        results["by_color"][color_key]["games"] += 1

        if verbose:
            symbol = {"wins": "W", "draws": "D", "losses": "L"}[key]
            print(f"  {opponent.name:12s} game {i+1:>3}/{n_games}  "
                  f"({'White' if ai_color == chess.WHITE else 'Black'})  {symbol}",
                  flush=True)
        if progress_callback is not None:
            progress_callback(results)

    return results


def _elo_difference(score: float) -> float:
    """Convert a match score in [0, 1] to an estimated Elo difference."""
    bounded = min(max(score, 1e-6), 1.0 - 1e-6)
    elo = -400.0 * math.log10(1.0 / bounded - 1.0)
    return 0.0 if abs(elo) < 0.05 else elo


def score_statistics(
    results: dict,
    *,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Return score, bootstrap confidence interval, and estimated Elo delta."""
    outcomes = results.get("outcomes", [])
    scores = np.asarray([RESULT_SCORES[outcome] for outcome in outcomes], dtype=np.float64)
    if scores.size == 0:
        return {
            "score_pct": 0.0,
            "score_ci_pct": [0.0, 0.0],
            "estimated_elo_delta": 0.0,
            "estimated_elo_ci": [0.0, 0.0],
            "confidence": confidence,
            "bootstrap_samples": bootstrap_samples,
        }

    score = float(scores.mean())
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        scores.size,
        size=(bootstrap_samples, scores.size),
    )
    bootstrap_means = scores[sample_indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [tail, 1.0 - tail])

    return {
        "score_pct": round(score * 100.0, 2),
        "score_ci_pct": [round(float(low) * 100.0, 2), round(float(high) * 100.0, 2)],
        "estimated_elo_delta": round(_elo_difference(score), 1),
        "estimated_elo_ci": [
            round(_elo_difference(float(low)), 1),
            round(_elo_difference(float(high)), 1),
        ],
        "confidence": confidence,
        "bootstrap_samples": bootstrap_samples,
    }


def _atomic_json_dump(payload: dict, path: str) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(temp_path, path)


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
    p.add_argument("--stockfish-movetime", type=float, default=0.05,
                   help="Stockfish seconds per move (default 0.05)")
    p.add_argument("--stockfish-threads", type=int, default=1,
                   help="Stockfish threads (default 1)")
    p.add_argument("--stockfish-hash", type=int, default=64,
                   help="Stockfish hash size in MB (default 64)")
    p.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES,
                   help=f"Half-move cap per game (default {DEFAULT_MAX_MOVES})")
    p.add_argument("--material-adjudication", type=float, default=3.0,
                   help="At the move cap, adjudicate a result at this pawn advantage; "
                        "0 disables (default 3.0)")
    p.add_argument("--seed", type=int, default=20260614,
                   help="Random seed for opponents and confidence intervals")
    p.add_argument("--bootstrap-samples", type=int, default=10_000,
                   help="Bootstrap samples for score confidence intervals")
    p.add_argument("--output",
                   help="Explicit JSON output path")
    p.add_argument("--resume", action="store_true",
                   help="Resume an interrupted run from --output")
    p.add_argument("--quiet",      action="store_true",
                   help="Suppress per-game output")
    args = p.parse_args()
    if args.games < 2:
        p.error("--games must be at least 2")
    if args.sims < 1:
        p.error("--sims must be at least 1")
    if args.bootstrap_samples < 100:
        p.error("--bootstrap-samples must be at least 100")
    if args.material_adjudication < 0:
        p.error("--material-adjudication cannot be negative")
    if args.resume and not args.output:
        p.error("--resume requires --output")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    resume_payload = None
    if args.resume:
        if not os.path.exists(args.output):
            p.error(f"resume report not found: {args.output}")
        with open(args.output) as f:
            resume_payload = json.load(f)
        expected = {
            "checkpoint": args.checkpoint,
            "n_sims": args.sims,
            "n_games": args.games,
            "max_moves": args.max_moves,
            "seed": args.seed,
            "material_adjudication": args.material_adjudication,
        }
        mismatches = [
            key for key, value in expected.items()
            if resume_payload.get(key) != value
        ]
        resume_stockfish = resume_payload.get("stockfish", {})
        expected_stockfish = {
            "path": args.stockfish,
            "skill": args.skill,
            "movetime": args.stockfish_movetime,
            "threads": args.stockfish_threads,
            "hash_mb": args.stockfish_hash,
        }
        mismatches.extend(
            f"stockfish.{key}"
            for key, value in expected_stockfish.items()
            if resume_stockfish.get(key) != value
        )
        if mismatches:
            p.error("resume settings differ for: " + ", ".join(mismatches))

    # ── Load AI ────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    net, mcts, device, meta = load_ai(args.checkpoint, args.sims)
    print(f"  {meta['az_channels']}ch / {meta['az_res_blocks']}-block  |  "
          f"device: {device}  |  MCTS sims: {args.sims}")
    print(f"  model: {meta['model_version']}  |  "
          f"training games: {meta['training_games']:,}")

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
                opponents.append(StockfishPlayer(
                    args.stockfish,
                    skill=args.skill,
                    movetime=args.stockfish_movetime,
                    threads=args.stockfish_threads,
                    hash_mb=args.stockfish_hash,
                ))
                print(f"  Stockfish: {args.stockfish}  skill={args.skill}  "
                      f"time={args.stockfish_movetime:.3f}s  "
                      f"threads={args.stockfish_threads}")
            except FileNotFoundError:
                print(f"  [warning] Stockfish not found at '{args.stockfish}' — skipping")

    if not opponents:
        print("No opponents available. Exiting.")
        return

    # ── Run matches ────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  {args.games} games per opponent  |  AI plays both colours")
    print(f"{'='*58}\n")

    all_results = dict(resume_payload.get("results", {})) if resume_payload else {}
    t_start = time.time()
    timestamp = (
        resume_payload.get("timestamp")
        if resume_payload
        else time.strftime("%Y%m%d_%H%M%S")
    )
    output_path = args.output or f"benchmark/results_{timestamp}.json"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    payload = {
        "status": "running",
        "timestamp": timestamp,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": args.checkpoint,
        "model": meta,
        "n_sims": args.sims,
        "n_games": args.games,
        "max_moves": args.max_moves,
        "material_adjudication": args.material_adjudication,
        "seed": args.seed,
        "stockfish": {
            "path": args.stockfish,
            "skill": args.skill,
            "movetime": args.stockfish_movetime,
            "threads": args.stockfish_threads,
            "hash_mb": args.stockfish_hash,
        },
        "results": all_results,
    }
    if os.path.exists(stats_path):
        s = torch.load(stats_path, map_location="cpu", weights_only=True)
        payload["ai_elo"] = float(s.get("ai_elo", 0))
        payload["total_games"] = int(s.get("total_games", 0))
    _atomic_json_dump(payload, output_path)

    try:
        for opponent_index, opp in enumerate(opponents):
            try:
                existing = all_results.get(opp.name)
                completed = len(existing.get("outcomes", [])) if existing else 0
                print(f"--- vs {opp.name} ---")
                if completed:
                    print(f"  Resuming after {completed}/{args.games} games")

                def save_progress(current_results):
                    all_results[opp.name] = current_results
                    payload["results"] = all_results
                    payload["updated_at"] = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    )
                    _atomic_json_dump(payload, output_path)

                r = run_match(
                    mcts,
                    opp,
                    args.games,
                    verbose=not args.quiet,
                    max_moves=args.max_moves,
                    material_adjudication=args.material_adjudication,
                    existing_results=existing,
                    progress_callback=save_progress,
                    seed=args.seed + opponent_index * 1_000_000,
                )
                stats = score_statistics(
                    r,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + opponent_index,
                )
                r["statistics"] = stats
                all_results[opp.name] = r
                payload["results"] = all_results
                _atomic_json_dump(payload, output_path)
                wp  = r["wins"]   / r["games"] * 100
                dp  = r["draws"]  / r["games"] * 100
                lp  = r["losses"] / r["games"] * 100
                ci_low, ci_high = stats["score_ci_pct"]
                print(f"  Result: {r['wins']}W / {r['draws']}D / {r['losses']}L  "
                      f"({wp:.0f}% / {dp:.0f}% / {lp:.0f}%)")
                print(f"  Score: {stats['score_pct']:.1f}%  "
                      f"95% CI [{ci_low:.1f}%, {ci_high:.1f}%]  "
                      f"estimated Elo delta: {stats['estimated_elo_delta']:+.0f}")
                for color in ("white", "black"):
                    split = r["by_color"][color]
                    print(f"    as {color.title():5s}: "
                          f"{split['wins']}W/{split['draws']}D/{split['losses']}L")
                termination_text = ", ".join(
                    f"{name}={count}"
                    for name, count in sorted(r["terminations"].items())
                )
                print(f"    terminations: {termination_text}")
                print()
            finally:
                if hasattr(opp, "close"):
                    opp.close()
    except KeyboardInterrupt:
        payload["status"] = "interrupted"
        payload["updated_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        _atomic_json_dump(payload, output_path)
        print(f"\nInterrupted. Progress saved → {output_path}")
        raise SystemExit(130)

    elapsed = time.time() - t_start

    # ── Summary table ──────────────────────────────────────────────────
    print(f"{'='*58}")
    print(f"  {'Opponent':<14} {'W':>4} {'D':>4} {'L':>4}  {'Score%':>7}  {'95% CI':>17}")
    print(f"  {'-'*14} {'----':>4} {'----':>4} {'----':>4}  {'-------':>7}  {'-'*17}")
    for name, r in all_results.items():
        stats = r["statistics"]
        ci_low, ci_high = stats["score_ci_pct"]
        print(f"  {name:<14} {r['wins']:>4} {r['draws']:>4} {r['losses']:>4}  "
              f"{stats['score_pct']:>6.1f}%  [{ci_low:>5.1f}, {ci_high:>5.1f}]")
    print(f"{'='*58}")
    print(f"  Total time: {elapsed:.1f}s")

    payload["status"] = "completed"
    payload["elapsed_seconds"] = round(elapsed, 2)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_json_dump(payload, output_path)
    print(f"\n  Saved → {output_path}\n")


if __name__ == "__main__":
    main()
