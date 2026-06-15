import chess

from tools.diagnose_endgames import (
    generate_conversion_positions,
    generate_mate_in_one_positions,
    mating_moves,
)


def test_generated_mate_positions_are_valid_and_have_mate():
    positions = generate_mate_in_one_positions(6, seed=17)

    assert len(positions) == 6
    for board in positions:
        assert board.is_valid()
        assert not board.is_check()
        assert mating_moves(board)


def test_generated_conversion_positions_are_valid_and_not_mate_in_one():
    positions = generate_conversion_positions(chess.QUEEN, 4, seed=19)

    assert len(positions) == 4
    for board, ai_color in positions:
        assert board.is_valid()
        assert not board.is_game_over()
        assert not mating_moves(board)
        assert len(board.pieces(chess.QUEEN, ai_color)) == 1
        assert len(board.pieces(chess.KING, ai_color)) == 1
        assert len(board.pieces(chess.KING, not ai_color)) == 1
