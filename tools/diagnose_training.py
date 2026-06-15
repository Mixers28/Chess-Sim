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
        "probe_value_head_flat": bool(value_array.std() < 0.02),
    }


@torch.no_grad()
def audit_replay(net, replay_path, sample_size=2048):
    data = np.load(replay_path)
    total = len(data["values"])
    rng = np.random.default_rng(7)
    indices = rng.choice(total, size=min(sample_size, total), replace=False)
    states = data["states"][indices].astype(np.float32)
    targets = data["values"][indices].astype(np.float32)

    predictions = []
    for start in range(0, len(states), 128):
        batch = torch.tensor(states[start:start + 128], dtype=torch.float32)
        predictions.append(net(batch)[1].numpy())
    predictions = np.concatenate(predictions)

    target_std = float(targets.std())
    prediction_std = float(predictions.std())
    correlation = (
        float(np.corrcoef(predictions, targets)[0, 1])
        if target_std > 0 and prediction_std > 0
        else 0.0
    )
    by_target = {}
    for target in (-1.0, 0.0, 1.0):
        selected = predictions[targets == target]
        by_target[str(int(target))] = {
            "count": int(len(selected)),
            "prediction_mean": (
                round(float(selected.mean()), 4) if len(selected) else None
            ),
        }

    return {
        "path": replay_path,
        "samples": int(len(targets)),
        "target_std": round(target_std, 6),
        "prediction_std": round(prediction_std, 6),
        "prediction_range": [
            round(float(predictions.min()), 4),
            round(float(predictions.max()), 4),
        ],
        "correlation": round(correlation, 4),
        "value_mae": round(float(np.abs(predictions - targets).mean()), 4),
        "by_target": by_target,
        "value_head_collapsed": bool(
            prediction_std < 0.02 or abs(correlation) < 0.05
        ),
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
    parser.add_argument(
        "--replay",
        default=os.path.join(ROOT_DIR, "checkpoint", "replay_buffer.npz"),
    )
    parser.add_argument("--overfit-steps", type=int, default=250)
    parser.add_argument("--skip-overfit", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, net = load_checkpoint(args.checkpoint)
    checkpoint_report = audit_checkpoint(args.checkpoint)
    report = {"checkpoint": checkpoint_report}
    if args.replay and os.path.exists(args.replay):
        report["replay"] = audit_replay(net, args.replay)
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
            "  synthetic probe: "
            + ("flat" if checkpoint["probe_value_head_flat"] else "varied")
        )
        if "replay" in report:
            replay = report["replay"]
            print("\nReplay audit")
            print(
                f"  samples={replay['samples']} "
                f"prediction_std={replay['prediction_std']:.4f} "
                f"range={replay['prediction_range']} "
                f"correlation={replay['correlation']:.4f} "
                f"value_mae={replay['value_mae']:.4f}"
            )
            print(
                "  replay value head: "
                + ("COLLAPSED" if replay["value_head_collapsed"] else "varied")
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

    if "replay" in report:
        failed = report["replay"]["value_head_collapsed"]
    else:
        failed = report["checkpoint"]["probe_value_head_flat"]
    failed = failed or not report.get("overfit", {"passed": True})["passed"]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
