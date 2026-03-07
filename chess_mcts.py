"""
chess_mcts.py — Monte Carlo Tree Search for AlphaZero chess.

Each node stores:
  N  — visit count
  W  — accumulated value (from the perspective of the player who MOVED to reach this node)
  P  — prior probability (from the network's policy head)

Q(node) = W/N  →  used in UCB from the parent's perspective.

Backup: value from leaf is from the LEAF player's perspective.
        As we walk up the path, we negate at each step (alternating players).

Tree reuse: get_policy accepts an optional `root` from a previous search.
            Passing the subtree saves re-exploring already-visited positions.
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
        node      = root
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
            value = 0.0 if outcome.winner is None else -1.0
        else:
            priors, value = self._evaluate(sim_board)
            for a in np.where(priors > 0)[0]:
                node.children[int(a)] = MCTSNode(prior=float(priors[a]))

        # ── Backup ────────────────────────────────────────────────────
        for parent, action, child in reversed(path):
            value = -value
            child.N += 1
            child.W += value

        root.N += 1
        return value

    # ── Public interface ───────────────────────────────────────────────
    def get_policy(self, board: chess.Board, temperature: float = 1.0,
                   add_noise: bool = False, root: MCTSNode = None):
        """
        Run `n_sims` simulations from `board`.

        Args:
          add_noise — mix Dirichlet noise into root priors (self-play exploration).
          root      — reuse a subtree from a previous search (tree reuse).
                      If None or empty, builds a fresh root.

        Returns:
          action        — chosen move index (int)
          visit_counts  — np.ndarray shape (4096,), raw visit counts
          root          — the root MCTSNode (pass back next call for tree reuse)
        """
        # Build or reuse root
        if root is None or not root.children:
            root = MCTSNode(prior=1.0)
            priors, _ = self._evaluate(board)

            if add_noise:
                legal_idx = np.where(priors > 0)[0]
                if len(legal_idx) > 0:
                    noise = np.random.dirichlet([0.3] * len(legal_idx))
                    noisy = priors.copy()
                    noisy[legal_idx] = 0.75 * priors[legal_idx] + 0.25 * noise
                    priors = noisy

            for a in np.where(priors > 0)[0]:
                root.children[int(a)] = MCTSNode(prior=float(priors[a]))
            root.N = 1

        for _ in range(self.n_sims):
            self._simulate(root, board)

        # Build visit count vector
        counts = np.zeros(4096, dtype=np.float32)
        for a, child in root.children.items():
            counts[a] = child.N

        if counts.sum() == 0:
            mask = legal_mask(board)
            counts = mask

        if temperature == 0 or temperature < 1e-6:
            action = int(counts.argmax())
        else:
            probs = counts ** (1.0 / temperature)
            probs /= probs.sum()
            action = int(np.random.choice(len(probs), p=probs))

        return action, counts, root

    def get_pv(self, root: MCTSNode, board: chess.Board, depth: int = 6) -> list[str]:
        """
        Extract the principal variation by greedily following max visit counts.
        Returns a list of UCI move strings (may be shorter than depth if game ends).
        """
        pv   = []
        node = root
        b    = board.copy(stack=False)

        for _ in range(depth):
            if not node.children or b.is_game_over():
                break
            best_a = max(node.children, key=lambda a: node.children[a].N)
            mv = idx_to_move(best_a, b)
            if mv is None:
                break
            pv.append(mv.uci())
            b.push(mv)
            node = node.children[best_a]

        return pv
