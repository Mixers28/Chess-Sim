"""
Reinitialise concept_bottleneck weights in the saved checkpoint.
Run once while wargames is STOPPED, before restarting it.
"""
import torch
import torch.nn as nn
from chess_model import (policy_net, optimizer, scheduler,
                         MODEL_PATH, AZ_CHANNELS, AZ_RES_BLOCKS, INPUT_PLANES)


def _kaiming_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


policy_net.concept_bottleneck.apply(_kaiming_init)
print("Concept head reinitialized.")

torch.save({
    "policy_state_dict": policy_net.state_dict(),
    "optimizer_state":   optimizer.state_dict(),
    "scheduler_state":   scheduler.state_dict(),
    "az_channels":       AZ_CHANNELS,
    "az_res_blocks":     AZ_RES_BLOCKS,
    "az_input_planes":   INPUT_PLANES,
}, MODEL_PATH)
print(f"Saved → {MODEL_PATH}")
