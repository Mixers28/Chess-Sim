"""
logistics_net.py — Transfer learning wrapper for logistics offer ranking.

Reuses the chess AlphaZeroNet res_tower trunk (frozen by default) by mapping
structured logistics offer features into the same 256×8×8 latent space.

Concept mapping (chess → logistics):
  material_balance  → margin_headroom
  king_safety       → critical_node_risk
  piece_mobility    → route_optionality
  pawn_structure    → supply_chain_dependency
  space_control     → network_coverage
  tactical_threat   → disruption_probability

Usage:
    from chess_model import policy_net, load_checkpoint
    from logistics_net import LogisticsNet

    load_checkpoint()
    model = LogisticsNet(policy_net, freeze_trunk=True)

    # offer_features: (B, 8) float tensor — see OFFER_FEATURE_NAMES
    route_logits, offer_value, concepts = model(offer_features)
"""

import torch
import torch.nn as nn

from chess_net import AlphaZeroNet
from chess_env import N_CONCEPTS

# Ordered feature names expected in the input vector.
OFFER_FEATURE_NAMES = [
    "origin",          # encoded location id
    "destination",     # encoded location id
    "cargo_class",     # categorical (encoded)
    "weight_kg",       # continuous
    "deadline_hours",  # continuous
    "declared_value",  # continuous
    "dgr_class",       # dangerous goods classification (0 = none)
    "service_tier",    # categorical (encoded)
]
N_OFFER_FEATURES = len(OFFER_FEATURE_NAMES)


class LogisticsInputAdapter(nn.Module):
    """
    Maps a flat logistics offer feature vector (B, N_OFFER_FEATURES) into
    the (B, channels, 8, 8) spatial latent space expected by the chess res_tower.

    Categorical features (origin, destination, cargo_class, dgr_class,
    service_tier) should be pre-encoded as integers or normalised floats
    before passing to this adapter.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(N_OFFER_FEATURES, 256),
            nn.ReLU(),
            nn.Linear(256, channels * 8 * 8),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N_OFFER_FEATURES) → (B, channels, 8, 8)"""
        return self.net(x).view(-1, self.channels, 8, 8)


class LogisticsNet(nn.Module):
    """
    Transfer learning network for logistics offer ranking.

    Shares the chess res_tower trunk with AlphaZeroNet.
    The trunk is frozen by default; call unfreeze_trunk() to fine-tune end-to-end.

    Outputs:
      route_logits  (B, 5)  — scores for top-5 candidate routes (use softmax/argsort)
      offer_value   (B,)    — predicted on-time probability × margin, in [0, 1]
      concepts      (B, 6)  — strategic concept activations (same as chess model)
    """

    def __init__(self, chess_net: AlphaZeroNet, freeze_trunk: bool = True):
        super().__init__()
        channels = chess_net.channels

        self.input_adapter     = LogisticsInputAdapter(channels)
        self.res_tower         = chess_net.res_tower
        self.concept_bottleneck = chess_net.concept_bottleneck

        if freeze_trunk:
            for p in self.res_tower.parameters():
                p.requires_grad = False

        # Route ranking: score top-5 candidate routes
        self.route_ranking_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

        # Offer value: on-time probability × margin estimate
        self.offer_value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, offer_features: torch.Tensor):
        """
        offer_features: (B, N_OFFER_FEATURES)
        Returns: (route_logits (B,5), offer_value (B,), concepts (B, N_CONCEPTS))
        """
        x            = self.input_adapter(offer_features)
        x            = self.res_tower(x)
        route_logits = self.route_ranking_head(x)
        offer_value  = self.offer_value_head(x).squeeze(1)
        concepts     = self.concept_bottleneck(x)
        return route_logits, offer_value, concepts

    def unfreeze_trunk(self):
        """Allow end-to-end fine-tuning after initial head training."""
        for p in self.res_tower.parameters():
            p.requires_grad = True
