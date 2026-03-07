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

from chess_env import encode, idx_to_move, legal_mask, ACTION_SIZE


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
                 c_puct: float = 1.5,
                 batch_size: int = 16):
        self.net        = net
        self.device     = device
        self.n_sims     = n_sims
        self.c_puct     = c_puct
        self.batch_size = batch_size

    # ── Single-position evaluation (root initialisation only) ─────────
    @torch.no_grad()
    def _evaluate(self, board: chess.Board):
        """Returns (priors (ACTION_SIZE,), value float) for one board."""
        state_t = torch.tensor(encode(board), dtype=torch.float32) \
                       .unsqueeze(0).to(self.device)
        policy_logits, value_t = self.net(state_t)

        mask   = legal_mask(board)
        priors = F.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()
        priors = priors * mask
        s      = priors.sum()
        priors = priors / s if s > 0 else mask / max(mask.sum(), 1e-8)

        return priors, value_t.item()

    # ── Batched evaluation (main simulation path) ──────────────────────
    @torch.no_grad()
    def _batch_evaluate(self, boards: list):
        """
        Evaluate N board positions in a single GPU forward pass.
        Returns (priors (N, ACTION_SIZE), values (N,)).
        """
        states  = np.stack([encode(b) for b in boards]).astype(np.float32)
        state_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        policy_logits, value_preds = self.net(state_t)

        masks   = np.stack([legal_mask(b) for b in boards])
        priors  = F.softmax(policy_logits, dim=1).cpu().numpy() * masks
        row_sum = priors.sum(axis=1, keepdims=True)
        priors /= np.where(row_sum > 0, row_sum, 1.0)

        return priors, value_preds.cpu().numpy().ravel()

    # ── Batched simulation: selection → batch eval → backup ───────────
    def _batch_simulate(self, root: MCTSNode, board: chess.Board, batch_size: int):
        """
        Run `batch_size` simulations in parallel using virtual loss.

        Virtual loss temporarily penalises each visited path so subsequent
        simulations in the same batch explore different lines.  All leaf
        positions are evaluated in one GPU call, then values are backed up
        and the virtual loss is cancelled.
        """
        paths       = []
        leaf_nodes  = []
        leaf_boards = []

        # ── Phase 1: Selection (sequential, virtual loss applied) ─────
        for _ in range(batch_size):
            node      = root
            sim_board = board.copy(stack=False)
            path: list[tuple[MCTSNode, int, MCTSNode]] = []

            while not node.is_leaf and not sim_board.is_game_over() \
                    and not sim_board.can_claim_draw():
                best_a = max(
                    node.children.keys(),
                    key=lambda a: node.children[a].ucb(node.N, self.c_puct)
                )
                child = node.children[best_a]
                path.append((node, best_a, child))
                # Virtual loss: makes this path less attractive for later sims
                child.N += 1
                child.W -= 1
                mv = idx_to_move(best_a, sim_board)
                if mv is None:
                    mv = random.choice(list(sim_board.legal_moves))
                sim_board.push(mv)
                node = child

            paths.append(path)
            leaf_nodes.append(node)
            leaf_boards.append(sim_board)

        # ── Phase 2: Batch evaluation ─────────────────────────────────
        values         = [0.0] * batch_size
        to_eval_idx    = []
        to_eval_boards = []

        for i, (node, sim_board) in enumerate(zip(leaf_nodes, leaf_boards)):
            if sim_board.is_game_over():
                outcome   = sim_board.outcome()
                values[i] = 0.0 if outcome.winner is None else -1.0
            elif sim_board.can_claim_draw():
                values[i] = 0.0
            else:
                to_eval_idx.append(i)
                to_eval_boards.append(sim_board)

        if to_eval_boards:
            priors_batch, value_batch = self._batch_evaluate(to_eval_boards)
            for j, i in enumerate(to_eval_idx):
                node   = leaf_nodes[i]
                priors = priors_batch[j]
                # Expand only once even if multiple sims reached the same leaf
                if node.is_leaf:
                    for a in np.where(priors > 0)[0]:
                        node.children[int(a)] = MCTSNode(prior=float(priors[a]))
                values[i] = float(value_batch[j])

        # ── Phase 3: Backup (undo virtual loss, propagate real values) ─
        for i, path in enumerate(paths):
            value = values[i]
            for parent, action, child in reversed(path):
                value    = -value
                # Virtual loss set W -= 1; undo it and add the real value
                child.W += value + 1
                # N was already incremented during virtual loss; don't re-add
            root.N += 1

    # ── Public interface ───────────────────────────────────────────────
    def get_policy(self, board: chess.Board, temperature: float = 1.0,
                   add_noise: bool = False, root: MCTSNode = None):
        """
        Run `n_sims` simulations from `board` using batched virtual-loss MCTS.

        Args:
          add_noise — mix Dirichlet noise into root priors (self-play exploration).
          root      — reuse a subtree from a previous search (tree reuse).

        Returns:
          action        — chosen move index (int)
          visit_counts  — np.ndarray shape (ACTION_SIZE,), raw visit counts
          root          — the root MCTSNode (pass back next call for tree reuse)
        """
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

        bs = self.batch_size
        for _ in range(self.n_sims // bs):
            self._batch_simulate(root, board, bs)
        remainder = self.n_sims % bs
        if remainder:
            self._batch_simulate(root, board, remainder)

        counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        for a, child in root.children.items():
            counts[a] = child.N

        if counts.sum() == 0:
            counts = legal_mask(board)

        if temperature == 0 or temperature < 1e-6:
            action = int(counts.argmax())
        else:
            # Normalise before raising to power to prevent float32 overflow
            # at low temperatures (e.g. counts**20 overflows for large visit counts)
            probs  = counts / counts.max()
            probs  = probs ** (1.0 / temperature)
            probs /= probs.sum()
            action = int(np.random.choice(len(probs), p=probs))

        return action, counts, root

    def get_pv(self, root: MCTSNode, board: chess.Board, depth: int = 6) -> list[str]:
        """Extract the principal variation by greedily following max visit counts."""
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
