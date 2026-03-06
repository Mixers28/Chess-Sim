"""
chess_mcts.py — Monte Carlo Tree Search for AlphaZero chess.

Each node stores:
  N  — visit count
  W  — accumulated value (from the perspective of the player who MOVED to reach this node)
  P  — prior probability (from the network's policy head)

Q(node) = W/N  →  used in UCB from the parent's perspective.

Backup: value from leaf is from the LEAF player's perspective.
        As we walk up the path, we negate at each step (alternating players).
"""

import math
import random

import chess
import numpy as np
import torch
import torch.nn.functional as F

from chess_env import encode, idx_to_move, legal_mask


class MCTSNode:
    __slots__ = ("N", "W", "P", "children")

    def __init__(self, prior: float):
        self.N: int   = 0
        self.W: float = 0.0
        self.P: float = prior
        self.children: dict[int, "MCTSNode"] = {}

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def ucb(self, parent_N: int, c_puct: float) -> float:
        return self.Q + c_puct * self.P * math.sqrt(parent_N) / (1 + self.N)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class MCTS:
    def __init__(self,
                 net,
                 device: torch.device,
                 n_sims: int = 100,
                 c_puct: float = 1.5):
        self.net    = net
        self.device = device
        self.n_sims = n_sims
        self.c_puct = c_puct

    # ── Neural network evaluation ──────────────────────────────────────
    @torch.no_grad()
    def _evaluate(self, board: chess.Board):
        """
        Returns (priors: np.ndarray shape (4096,), value: float).
        value is from the perspective of the player to move at `board`.
        """
        state_t = torch.tensor(encode(board), dtype=torch.float32) \
                       .unsqueeze(0).to(self.device)
        policy_logits, value_t = self.net(state_t)

        mask    = legal_mask(board)
        priors  = F.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()
        priors  = priors * mask
        s = priors.sum()
        priors  = priors / s if s > 0 else mask / max(mask.sum(), 1e-8)

        return priors, value_t.item()

    # ── One MCTS simulation ────────────────────────────────────────────
    def _simulate(self, root: MCTSNode, board: chess.Board) -> float:
        node     = root
        sim_board = board.copy(stack=False)
        path: list[tuple[MCTSNode, int, MCTSNode]] = []

        # ── Selection ─────────────────────────────────────────────────
        while not node.is_leaf and not sim_board.is_game_over():
            best_a = max(
                node.children.keys(),
                key=lambda a: node.children[a].ucb(node.N, self.c_puct)
            )
            child = node.children[best_a]
            path.append((node, best_a, child))
            mv = idx_to_move(best_a, sim_board)
            if mv is None:
                mv = random.choice(list(sim_board.legal_moves))
            sim_board.push(mv)
            node = child

        # ── Evaluation ────────────────────────────────────────────────
        if sim_board.is_game_over():
            outcome = sim_board.outcome()
            # sim_board.turn = the player to move = the one who was just checkmated
            # (or the stalemated player). In all terminal cases they either lost or drew.
            value = 0.0 if outcome.winner is None else -1.0
        else:
            # Expand leaf
            priors, value = self._evaluate(sim_board)
            for a in np.where(priors > 0)[0]:
                node.children[int(a)] = MCTSNode(prior=float(priors[a]))

        # ── Backup ────────────────────────────────────────────────────
        # value is from sim_board.turn's perspective at the leaf.
        # Walk the path in reverse; negate at each step (perspective switches).
        for parent, action, child in reversed(path):
            value = -value                # switch perspective to parent's player
            child.N += 1
            child.W += value

        root.N += 1
        return value

    # ── Public interface ───────────────────────────────────────────────
    def get_policy(self, board: chess.Board, temperature: float = 1.0):
        """
        Run `n_sims` simulations from `board`.

        Returns:
          action        — chosen move index (int)
          visit_counts  — np.ndarray shape (4096,), raw visit counts
        """
        root = MCTSNode(prior=1.0)

        # Evaluate and expand root before simulations
        priors, _ = self._evaluate(board)
        for a in np.where(priors > 0)[0]:
            root.children[int(a)] = MCTSNode(prior=float(priors[a]))
        root.N = 1

        for _ in range(self.n_sims):
            self._simulate(root, board)

        # Build visit count vector
        counts = np.zeros(4096, dtype=np.float32)
        for a, child in root.children.items():
            counts[a] = child.N

        # Choose action
        if counts.sum() == 0:
            # Fallback: uniform over legal moves
            mask = legal_mask(board)
            counts = mask

        if temperature == 0 or temperature < 1e-6:
            action = int(counts.argmax())
        else:
            # Sample proportional to counts^(1/temperature)
            probs = counts ** (1.0 / temperature)
            probs /= probs.sum()
            action = int(np.random.choice(len(probs), p=probs))

        return action, counts
