#!/usr/bin/env python3
"""Generate compact mate and endgame-conversion supervision with Stockfish."""

import argparse
import os
import random
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import chess
import chess.engine
import numpy as np

from chess_env import compute_concept_labels, encode, move_to_idx
from chess_model import EXPERT_SCHEMA_VERSION
from tools.diagnose_endgames import (
    generate_conversion_positions,
    generate_mate_in_one_positions,
    mating_moves,
)


def append_sample(states, actions, values, concepts, board, move, value):
    states.append(encode(board).astype(np.float16))
    actions.append(move_to_idx(move))
    values.append(value)
    concepts.append(compute_concept_labels(board))

    mirrored_board = board.mirror()
    mirrored_move = chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )
    if mirrored_move in mirrored_board.legal_moves:
        states.append(encode(mirrored_board).astype(np.float16))
        actions.append(move_to_idx(mirrored_move))
        values.append(value)
        concepts.append(compute_concept_labels(mirrored_board))


def generate_mate_samples(count, seed, states, actions, values, concepts):
    boards = generate_mate_in_one_positions(count, seed)
    for board in boards:
        move = sorted(mating_moves(board), key=lambda item: item.uci())[0]
        append_sample(states, actions, values, concepts, board, move, 1.0)


def generate_conversion_samples(
    engine,
    target_samples,
    seed,
    depth,
    max_plies,
    states,
    actions,
    values,
    concepts,
):
    rng = random.Random(seed)
    accepted_games = 0
    attempts = 0
    while len(values) < target_samples and attempts < target_samples * 10:
        attempts += 1
        piece_type = chess.QUEEN if attempts % 2 else chess.ROOK
        board, winning_color = generate_conversion_positions(
            piece_type,
            1,
            seed=rng.randrange(1 << 30),
        )[0]
        trajectory = []
        for _ in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break
            result = engine.play(board, chess.engine.Limit(depth=depth))
            move = result.move
            value = 1.0 if board.turn == winning_color else -1.0
            trajectory.append((board.copy(stack=False), move, value))
            board.push(move)

        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner != winning_color:
            continue
        accepted_games += 1
        for position, move, value in trajectory:
            append_sample(
                states,
                actions,
                values,
                concepts,
                position,
                move,
                value,
            )
            if len(values) >= target_samples:
                break
        if accepted_games % 25 == 0:
            print(
                f"  accepted games={accepted_games:,} "
                f"samples={len(values):,}/{target_samples:,}",
                flush=True,
            )
    if len(values) < target_samples:
        raise RuntimeError(
            f"generated only {len(values):,}/{target_samples:,} samples"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--mate-positions", type=int, default=2_000)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--stockfish", default="/usr/games/stockfish")
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT_DIR, "checkpoint", "endgame_expert.npz"),
    )
    args = parser.parse_args()
    if args.samples < args.mate_positions * 2:
        parser.error("--samples must allow two mirrored samples per mate position")

    states = []
    actions = []
    values = []
    concepts = []

    print("Generating exact mate-in-one supervision", flush=True)
    generate_mate_samples(
        args.mate_positions,
        args.seed,
        states,
        actions,
        values,
        concepts,
    )

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({"Threads": 1, "Hash": 64})
    try:
        print("Generating Stockfish conversion trajectories", flush=True)
        generate_conversion_samples(
            engine,
            args.samples,
            args.seed + 1,
            args.depth,
            args.max_plies,
            states,
            actions,
            values,
            concepts,
        )
    finally:
        engine.quit()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        states=np.asarray(states[:args.samples], dtype=np.float16),
        actions=np.asarray(actions[:args.samples], dtype=np.int32),
        values=np.asarray(values[:args.samples], dtype=np.float32),
        concepts=np.asarray(concepts[:args.samples], dtype=np.float32),
        schema_version=np.array(EXPERT_SCHEMA_VERSION, dtype=np.int64),
    )
    print(f"Saved {args.samples:,} samples -> {args.output}")


if __name__ == "__main__":
    main()
