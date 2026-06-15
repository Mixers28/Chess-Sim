#!/usr/bin/env python3
"""Measure mate recognition and won-endgame conversion without adjudication."""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import chess

from benchmark import StockfishPlayer, load_ai
from chess_env import idx_to_move, move_to_idx


def mating_moves(board: chess.Board) -> list[chess.Move]:
    moves = []
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            moves.append(move)
        board.pop()
    return moves


def generate_mate_in_one_positions(count: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    positions = []
    seen = set()
    attempts = 0
    max_attempts = max(200_000, count * 400)
    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        board = chess.Board(None)
        side = chess.WHITE if len(positions) % 2 == 0 else chess.BLACK
        king_square = rng.choice(chess.SQUARES)
        enemy_king_square = rng.choice(chess.SQUARES)
        queen_square = rng.choice(chess.SQUARES)
        if len({king_square, enemy_king_square, queen_square}) != 3:
            continue
        board.set_piece_at(king_square, chess.Piece(chess.KING, side))
        board.set_piece_at(enemy_king_square, chess.Piece(chess.KING, not side))
        board.set_piece_at(queen_square, chess.Piece(chess.QUEEN, side))
        board.turn = side
        if not board.is_valid() or board.is_check() or board.is_game_over():
            continue
        mates = mating_moves(board)
        if not mates:
            continue
        key = board.board_fen() + (" w" if board.turn else " b")
        if key in seen:
            continue
        seen.add(key)
        positions.append(board)
    if len(positions) != count:
        raise RuntimeError(f"could only generate {len(positions)}/{count} mate probes")
    return positions


def generate_conversion_positions(
    piece_type: chess.PieceType,
    count: int,
    seed: int,
) -> list[tuple[chess.Board, chess.Color]]:
    rng = random.Random(seed + piece_type * 10_000)
    positions = []
    seen = set()
    attempts = 0
    while len(positions) < count and attempts < 200_000:
        attempts += 1
        ai_color = chess.WHITE if len(positions) % 2 == 0 else chess.BLACK
        board = chess.Board(None)
        squares = rng.sample(list(chess.SQUARES), 3)
        board.set_piece_at(squares[0], chess.Piece(chess.KING, ai_color))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, not ai_color))
        board.set_piece_at(squares[2], chess.Piece(piece_type, ai_color))
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        if (
            not board.is_valid()
            or board.is_check()
            or board.is_game_over()
            or mating_moves(board)
            or board.is_attacked_by(
                not ai_color,
                squares[2],
            )
        ):
            continue
        key = board.fen()
        if key in seen:
            continue
        seen.add(key)
        positions.append((board, ai_color))
    if len(positions) != count:
        raise RuntimeError(
            f"could only generate {len(positions)}/{count} conversion positions"
        )
    return positions


def run_mate_probes(mcts, count: int, seed: int) -> dict:
    records = []
    for index, board in enumerate(generate_mate_in_one_positions(count, seed), 1):
        expected = mating_moves(board)
        action, _, _ = mcts.get_policy(board, temperature=0)
        chosen = idx_to_move(action, board)
        passed = chosen in expected
        records.append({
            "probe": index,
            "fen": board.fen(),
            "chosen": chosen.uci() if chosen else None,
            "mating_moves": [move.uci() for move in expected],
            "passed": passed,
        })
        print(
            f"  mate {index:>2}/{count}: "
            f"{'PASS' if passed else 'FAIL'} "
            f"chosen={chosen.uci() if chosen else 'none'}",
            flush=True,
        )
    passed = sum(record["passed"] for record in records)
    return {
        "passed": passed,
        "total": count,
        "accuracy_pct": round(passed / count * 100.0, 2),
        "records": records,
    }


def play_conversion(
    board: chess.Board,
    ai_color: chess.Color,
    mcts,
    defender,
    max_plies: int,
) -> dict:
    root = None
    start_fen = board.fen()
    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        if board.turn == ai_color:
            action, _, root = mcts.get_policy(board, temperature=0, root=root)
            move = idx_to_move(action, board)
            if move is None:
                move = random.choice(list(board.legal_moves))
            root = root.children.get(action)
        else:
            move = defender.move(board)
            if root is not None:
                root = root.children.get(move_to_idx(move))
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result = "move_cap"
    elif outcome.winner == ai_color:
        result = "win"
    elif outcome.winner is None:
        result = "draw"
    else:
        result = "loss"
    return {
        "start_fen": start_fen,
        "ai_color": "white" if ai_color else "black",
        "result": result,
        "termination": (
            outcome.termination.name.lower() if outcome is not None else "move_cap"
        ),
        "plies": len(board.move_stack),
        "final_fen": board.fen(),
    }


def run_conversion_suite(
    mcts,
    defender,
    positions_per_type: int,
    max_plies: int,
    seed: int,
) -> dict:
    records = []
    for piece_type, label in ((chess.QUEEN, "KQK"), (chess.ROOK, "KRK")):
        positions = generate_conversion_positions(
            piece_type,
            positions_per_type,
            seed,
        )
        for index, (board, ai_color) in enumerate(positions, 1):
            record = play_conversion(
                board.copy(stack=False),
                ai_color,
                mcts,
                defender,
                max_plies,
            )
            record["type"] = label
            record["case"] = index
            records.append(record)
            print(
                f"  {label} {index:>2}/{positions_per_type}: "
                f"{record['result'].upper()} "
                f"{record['termination']} {record['plies']} plies",
                flush=True,
            )

    counts = Counter(record["result"] for record in records)
    wins = counts["win"]
    return {
        "wins": wins,
        "draws": counts["draw"],
        "losses": counts["loss"],
        "move_caps": counts["move_cap"],
        "total": len(records),
        "conversion_pct": round(wins / len(records) * 100.0, 2),
        "by_type": {
            label: dict(Counter(
                record["result"] for record in records if record["type"] == label
            ))
            for label in ("KQK", "KRK")
        },
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(ROOT_DIR, "checkpoint", "model.pt"),
    )
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--mate-probes", type=int, default=20)
    parser.add_argument("--positions-per-type", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--stockfish", default="/usr/games/stockfish")
    parser.add_argument("--stockfish-skill", type=int, default=1)
    parser.add_argument("--stockfish-time", type=float, default=0.02)
    parser.add_argument("--output")
    args = parser.parse_args()

    _, mcts, _, meta = load_ai(args.checkpoint, args.sims)
    defender = StockfishPlayer(
        args.stockfish,
        skill=args.stockfish_skill,
        movetime=args.stockfish_time,
        threads=1,
        hash_mb=32,
    )
    try:
        print("Mate-in-one probes", flush=True)
        mates = run_mate_probes(mcts, args.mate_probes, args.seed)
        print("\nWon-endgame conversion, no material adjudication", flush=True)
        conversion = run_conversion_suite(
            mcts,
            defender,
            args.positions_per_type,
            args.max_plies,
            args.seed,
        )
    finally:
        defender.close()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": args.checkpoint,
        "model": meta,
        "sims": args.sims,
        "seed": args.seed,
        "mate_in_one": mates,
        "conversion": conversion,
    }
    output = args.output or os.path.join(
        ROOT_DIR,
        "benchmark",
        f"endgame_diagnostic_{meta['model_version']}.json",
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\nSummary")
    print(
        f"  mate-in-one: {mates['passed']}/{mates['total']} "
        f"({mates['accuracy_pct']:.1f}%)"
    )
    print(
        f"  conversion: {conversion['wins']}/{conversion['total']} "
        f"({conversion['conversion_pct']:.1f}%)"
    )
    print(f"  saved: {output}")


if __name__ == "__main__":
    main()
