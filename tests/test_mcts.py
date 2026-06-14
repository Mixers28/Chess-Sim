import chess
import pytest
import torch

from chess_env import ACTION_SIZE
from chess_mcts import MCTS, MCTSNode


class ConstantValueNet(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, states):
        batch = states.shape[0]
        return (
            torch.zeros(batch, ACTION_SIZE),
            torch.full((batch,), self.value),
            torch.zeros(batch, 6),
        )


def test_root_value_uses_root_player_perspective():
    root = MCTSNode(prior=1.0)
    first = MCTSNode(prior=0.5)
    first.N, first.W = 3, 1.5
    second = MCTSNode(prior=0.5)
    second.N, second.W = 1, -1.0
    root.children = {1: first, 2: second}

    assert MCTS.root_value(root) == pytest.approx(0.125)


def test_leaf_value_is_negated_once_for_root_player():
    mcts = MCTS(
        ConstantValueNet(0.75),
        torch.device("cpu"),
        n_sims=1,
        batch_size=1,
    )

    _, _, root = mcts.get_policy(chess.Board(), temperature=0)

    assert MCTS.root_value(root) == pytest.approx(-0.75)
