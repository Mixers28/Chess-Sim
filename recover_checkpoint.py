"""
Restore weights from checkpoint.pt (legacy format, 15,504 games)
into model.pt (current format). Optimizer resets to fresh state —
that's fine, the trunk weights are what matter.

Run while wargames is STOPPED.
"""
import torch
from chess_model import (policy_net, optimizer, scheduler,
                         MODEL_PATH, STATS_PATH, CHECKPOINT_PATH,
                         AZ_CHANNELS, AZ_RES_BLOCKS, INPUT_PLANES)
import chess_model as M

ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

# Verify architecture matches
assert ckpt["az_channels"]    == AZ_CHANNELS,    f"Channel mismatch: {ckpt['az_channels']} vs {AZ_CHANNELS}"
assert ckpt["az_res_blocks"]  == AZ_RES_BLOCKS,  f"Block mismatch: {ckpt['az_res_blocks']} vs {AZ_RES_BLOCKS}"
assert ckpt["az_input_planes"]== INPUT_PLANES,    f"Plane mismatch: {ckpt['az_input_planes']} vs {INPUT_PLANES}"

# Load trunk weights (concept head will keep its fresh init — that's correct)
policy_net.load_state_dict(ckpt["policy_state_dict"], strict=False)
print(f"Weights loaded — {ckpt['total_games']:,} games, Elo {ckpt['ai_elo']:.0f}")

# Save model.pt with correct format (fresh optimizer — trunk weights restored)
torch.save({
    "policy_state_dict": policy_net.state_dict(),
    "optimizer_state":   optimizer.state_dict(),
    "scheduler_state":   scheduler.state_dict(),
    "az_channels":       AZ_CHANNELS,
    "az_res_blocks":     AZ_RES_BLOCKS,
    "az_input_planes":   INPUT_PLANES,
}, MODEL_PATH)

# Restore stats
torch.save({
    "total_games":    ckpt["total_games"],
    "selfplay_games": ckpt.get("selfplay_games", ckpt["total_games"]),
    "human_games":    ckpt.get("human_games", 0),
    "human_wins":     ckpt.get("human_wins", 0),
    "human_losses":   ckpt.get("human_losses", 0),
    "human_draws":    ckpt.get("human_draws", 0),
    "ai_elo":         ckpt["ai_elo"],
    "elo_history":    ckpt.get("elo_history", []),
}, STATS_PATH)

print(f"Saved model.pt + stats.pt — ready to resume from game {ckpt['total_games']:,}")
