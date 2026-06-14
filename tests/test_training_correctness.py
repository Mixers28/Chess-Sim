import chess
import numpy as np

import chess_wargames
from chess_env import ACTION_SIZE, move_to_idx
from chess_mcts import MCTSNode


class OneMoveMCTS:
    def get_policy(self, board, temperature, add_noise):
        move = next(iter(board.legal_moves))
        action = move_to_idx(move)
        counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        counts[action] = 1.0
        root = MCTSNode(prior=1.0)
        child = MCTSNode(prior=1.0)
        child.N = 1
        root.children[action] = child
        return action, counts, root

    @staticmethod
    def root_value(root):
        return 0.0


def test_move_cap_is_a_neutral_draw_target(monkeypatch):
    monkeypatch.setattr(chess_wargames, "MAX_MOVES", 1)
    monkeypatch.setattr(chess_wargames, "_book_move", lambda board: None)
    monkeypatch.setattr(chess_wargames, "RESIGN_DISABLE_PROB", 1.0)

    samples, result, moves = chess_wargames.selfplay_game(
        chess.Board(),
        OneMoveMCTS(),
    )

    assert result == "D"
    assert moves == 1
    assert [sample[2] for sample in samples] == [0.0]
