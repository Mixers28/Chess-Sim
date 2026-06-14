#!/usr/bin/env python3
"""Audit chess training invariants and run a tiny overfit experiment."""

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import chess
import numpy as np
import torch
import torch.nn.functional as F

from chess_env import ACTION_SIZE, encode, idx_to_move, legal_mask, move_to_idx
from chess_net import AlphaZeroNet


PROBE_POSITIONS = [
    ("start", chess.STARTING_FEN),
    ("white_queen_up", "7k/8/8/8/8/8/8/Q6K w - - 0 1"),
    ("black_queen_down", "7k/8/8/8/8/8/8/Q6K b - - 0 1"),
    ("white_queen_down", "7k/8/8/8/8/8/q7/7K w - - 0 1"),
    ("black_queen_up", "7k/8/8/8/8/8/q7/7K b - - 0 1"),
]

OVERFIT_SAMPLES = [
    (chess.STARTING_FEN, "e2e4", 0.0),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
     "c7c5", 0.0),
    ("7k/8/8/8/8/8/8/Q6K w - - 0 1", "a1a8", 1.0),
    ("7k/8/8/8/8/8/8/Q6K b - - 0 1", "h8g8", -1.0),
    ("7k/8/8/8/8/8/q7/7K b - - 0 1", "a2a1", 1.0),
    ("7k/8/8/8/8/8/q7/7K w - - 0 1", "h1g1", -1.0),
]


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    net = AlphaZeroNet(
        int(checkpoint["az_channels"]),
        int(checkpoint["az_res_blocks"]),
    )
    net.load_state_dict(checkpoint["policy_state_dict"])
    net.eval()
    return checkpoint, net


@torch.no_grad()
def audit_checkpoint(path):
    checkpoint, net = load_checkpoint(path)
    states = torch.tensor(
        np.stack([encode(chess.Board(fen)) for _, fen in PROBE_POSITIONS]),
        dtype=torch.float32,
    )
    policy_logits, values, _ = net(states)

    rows = []
    for i, (name, fen) in enumerate(PROBE_POSITIONS):
        board = chess.Board(fen)
        mask = torch.tensor(legal_mask(board), dtype=torch.bool)
        masked_logits = policy_logits[i].masked_fill(~mask, -torch.inf)
        action = int(masked_logits.argmax())
        move = idx_to_move(action, board)
        rows.append({
            "name": name,
            "value": round(float(values[i]), 4),
            "top_move": move.uci() if move else None,
        })

    value_array = values.numpy()
    return {
        "model_version": checkpoint.get("model_version", "unversioned"),
        "training_games": int(checkpoint.get("training_games", 0)),
        "positions": rows,
        "value_std": round(float(value_array.std()), 6),
        "value_range": round(float(value_array.max() - value_array.min()), 6),
        "value_head_collapsed": bool(value_array.std() < 0.02),
    }


def build_overfit_batch(device):
    states = []
    actions = []
    values = []
    for fen, uci, value in OVERFIT_SAMPLES:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"diagnostic move {uci} is illegal in {fen}")
        states.append(encode(board))
        actions.append(move_to_idx(move))
        values.append(value)
    return (
        torch.tensor(np.stack(states), dtype=torch.float32, device=device),
        torch.tensor(actions, dtype=torch.long, device=device),
        torch.tensor(values, dtype=torch.float32, device=device),
    )


def run_overfit(steps, device):
    torch.manual_seed(7)
    np.random.seed(7)
    net = AlphaZeroNet(channels=16, n_res=1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    states, actions, values = build_overfit_batch(device)

    net.train()
    for _ in range(steps):
        policy_logits, value_predictions, _ = net(states)
        policy_loss = F.cross_entropy(policy_logits, actions)
        value_loss = F.mse_loss(value_predictions, values)
        loss = policy_loss + 2.0 * value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        policy_logits, value_predictions, _ = net(states)
        accuracy = float((policy_logits.argmax(dim=1) == actions).float().mean())
        value_mae = float((value_predictions - values).abs().mean())
        final_loss = float(
            F.cross_entropy(policy_logits, actions)
            + 2.0 * F.mse_loss(value_predictions, values)
        )

    return {
        "steps": steps,
        "device": str(device),
        "policy_accuracy": round(accuracy, 4),
        "value_mae": round(value_mae, 4),
        "loss": round(final_loss, 4),
        "passed": bool(accuracy == 1.0 and value_mae < 0.1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(ROOT_DIR, "checkpoint", "model.pt"),
    )
    parser.add_argument("--overfit-steps", type=int, default=250)
    parser.add_argument("--skip-overfit", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"checkpoint": audit_checkpoint(args.checkpoint)}
    if not args.skip_overfit:
        report["overfit"] = run_overfit(args.overfit_steps, device)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        checkpoint = report["checkpoint"]
        print("Checkpoint audit")
        print(f"  model: {checkpoint['model_version']}")
        print(f"  training games: {checkpoint['training_games']:,}")
        for row in checkpoint["positions"]:
            print(
                f"  {row['name']:<18} value={row['value']:+.4f} "
                f"top={row['top_move']}"
            )
        print(
            f"  value std={checkpoint['value_std']:.6f} "
            f"range={checkpoint['value_range']:.6f}"
        )
        print(
            "  checkpoint value head: "
            + ("COLLAPSED" if checkpoint["value_head_collapsed"] else "varied")
        )
        if "overfit" in report:
            overfit = report["overfit"]
            print("\nTiny-dataset overfit")
            print(
                f"  steps={overfit['steps']} device={overfit['device']} "
                f"policy_accuracy={overfit['policy_accuracy']:.1%} "
                f"value_mae={overfit['value_mae']:.4f} "
                f"loss={overfit['loss']:.4f}"
            )
            print("  training path: " + ("PASS" if overfit["passed"] else "FAIL"))

    failed = report["checkpoint"]["value_head_collapsed"]
    failed = failed or not report.get("overfit", {"passed": True})["passed"]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
