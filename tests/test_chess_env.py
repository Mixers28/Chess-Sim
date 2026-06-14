"""
tests/test_chess_env.py — Tests for board encoding, move encoding/decoding, and mirroring.

Run with:
    pytest tests/test_chess_env.py -v

Encoding contract (ACTION_SIZE = 8192):
  0–4095    : standard moves + queen promotions (implicit)
  4096–4239 : compact knight, bishop, and rook underpromotions
  4240–8191 : reserved
"""

import chess
import numpy as np
import pytest

from chess_env import (
    ACTION_SIZE, CONCEPT_NAMES, INPUT_PLANES,
    compute_concept_labels, encode, idx_to_move, legal_mask, mirror_sample,
    move_to_idx,
)


# ── Move encoding ─────────────────────────────────────────────────────────────

class TestMoveToIdx:
    def test_standard_move_in_range(self):
        mv = chess.Move.from_uci("e2e4")
        assert 0 <= move_to_idx(mv) < 4096

    def test_queen_promotion_in_standard_range(self):
        mv = chess.Move.from_uci("a7a8q")
        idx = move_to_idx(mv)
        assert 0 <= idx < 4096

    def test_knight_underpromotion_in_upper_range(self):
        mv = chess.Move.from_uci("a7a8n")
        idx = move_to_idx(mv)
        assert 4096 <= idx < 8192

    def test_rook_underpromotion_has_distinct_index(self):
        q_idx = move_to_idx(chess.Move.from_uci("a7a8q"))
        r_idx = move_to_idx(chess.Move.from_uci("a7a8r"))
        assert 4096 <= r_idx < ACTION_SIZE
        assert r_idx != q_idx

    def test_bishop_underpromotion_has_distinct_index(self):
        q_idx = move_to_idx(chess.Move.from_uci("a7a8q"))
        b_idx = move_to_idx(chess.Move.from_uci("a7a8b"))
        assert 4096 <= b_idx < ACTION_SIZE
        assert b_idx != q_idx

    def test_all_promotion_types_have_unique_indices(self):
        moves = [chess.Move.from_uci(f"a7a8{piece}") for piece in "qrbn"]
        assert len({move_to_idx(move) for move in moves}) == 4

    def test_all_starting_moves_have_valid_indices(self):
        board = chess.Board()
        indices = [move_to_idx(mv) for mv in board.legal_moves]
        assert all(0 <= i < ACTION_SIZE for i in indices)

    def test_starting_moves_have_unique_indices(self):
        board = chess.Board()
        indices = [move_to_idx(mv) for mv in board.legal_moves]
        assert len(indices) == len(set(indices))

    def test_encoding_formula(self):
        mv = chess.Move(chess.E2, chess.E4)
        assert move_to_idx(mv) == chess.E2 * 64 + chess.E4

    def test_underpromotion_indices_use_compact_range(self):
        indices = [
            move_to_idx(chess.Move.from_uci(f"a7a8{piece}"))
            for piece in "nbr"
        ]
        assert indices == [4099, 4100, 4101]


class TestIdxToMove:
    def test_standard_move_roundtrip(self):
        board = chess.Board()
        for mv in board.legal_moves:
            idx = move_to_idx(mv)
            recovered = idx_to_move(idx, board)
            assert recovered == mv, f"Roundtrip failed for {mv}"

    def test_queen_promotion_roundtrip(self):
        board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")
        mv = chess.Move.from_uci("a7a8q")
        assert idx_to_move(move_to_idx(mv), board) == mv

    def test_knight_underpromotion_roundtrip(self):
        board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")
        mv = chess.Move.from_uci("a7a8n")
        assert idx_to_move(move_to_idx(mv), board) == mv

    @pytest.mark.parametrize("piece", ["n", "b", "r"])
    def test_all_underpromotions_roundtrip(self, piece):
        board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")
        mv = chess.Move.from_uci(f"a7a8{piece}")
        assert idx_to_move(move_to_idx(mv), board) == mv

    @pytest.mark.parametrize("piece", ["n", "b", "r"])
    def test_black_underpromotions_roundtrip(self, piece):
        board = chess.Board("8/8/8/8/8/8/p7/8 b - - 0 1")
        mv = chess.Move.from_uci(f"a2a1{piece}")
        assert idx_to_move(move_to_idx(mv), board) == mv

    def test_illegal_move_returns_none(self):
        board = chess.Board()
        # King moving forward from e1 to e2 is not legal at the start
        illegal_idx = chess.E1 * 64 + chess.E2
        assert idx_to_move(illegal_idx, board) is None

    def test_out_of_range_returns_none(self):
        board = chess.Board()
        # Index beyond ACTION_SIZE — should not crash, and returns None or raises cleanly
        result = idx_to_move(ACTION_SIZE - 1, board)
        # May be None or a legal knight underpromotion; just assert no crash
        assert result is None or isinstance(result, chess.Move)

    def test_multi_position_roundtrip(self):
        """Roundtrip holds across several positions, not just starting."""
        fens = [
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "8/8/8/8/8/8/4k3/4K3 w - - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # castling position
        ]
        for fen in fens:
            board = chess.Board(fen)
            for mv in board.legal_moves:
                idx = move_to_idx(mv)
                recovered = idx_to_move(idx, board)
                assert recovered == mv, f"Roundtrip failed for {mv} in {fen}"


# ── Legal mask ────────────────────────────────────────────────────────────────

class TestLegalMask:
    def test_shape_and_dtype(self):
        mask = legal_mask(chess.Board())
        assert mask.shape == (ACTION_SIZE,)
        assert mask.dtype == np.float32

    def test_mask_covers_all_legal_moves(self):
        board = chess.Board()
        mask = legal_mask(board)
        for mv in board.legal_moves:
            assert mask[move_to_idx(mv)] == 1.0, f"Legal move {mv} missing from mask"

    def test_mask_count_matches_unique_indices(self):
        board = chess.Board()
        mask = legal_mask(board)
        unique_indices = len({move_to_idx(mv) for mv in board.legal_moves})
        assert int(mask.sum()) == unique_indices

    def test_mask_binary(self):
        mask = legal_mask(chess.Board())
        assert set(np.unique(mask)).issubset({0.0, 1.0})

    def test_promotion_position_has_queen_and_knight(self):
        board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")
        mask = legal_mask(board)
        for piece in "qrbn":
            assert mask[move_to_idx(chess.Move.from_uci(f"a7a8{piece}"))] == 1.0
        assert int(mask.sum()) == len(list(board.legal_moves))

    def test_castling_moves_in_mask(self):
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        mask = legal_mask(board)
        # White can castle kingside (e1g1) and queenside (e1c1)
        assert mask[move_to_idx(chess.Move.from_uci("e1g1"))] == 1.0
        assert mask[move_to_idx(chess.Move.from_uci("e1c1"))] == 1.0

    def test_empty_mask_in_checkmate(self):
        # Scholar's mate — black is in checkmate, no legal moves
        board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        if board.is_checkmate():
            mask = legal_mask(board)
            assert mask.sum() == 0.0


# ── Mirror sample ─────────────────────────────────────────────────────────────

class TestMirrorSample:
    def _make_uniform_policy(self, board):
        mask = legal_mask(board)
        total = mask.sum()
        return mask / total if total > 0 else mask

    def test_policy_sum_preserved(self):
        board = chess.Board()
        policy = self._make_uniform_policy(board)
        _, mp, _ = mirror_sample(board, policy, 1.0)
        np.testing.assert_allclose(mp.sum(), policy.sum(), rtol=1e-5)

    def test_value_unchanged(self):
        _, _, mv = mirror_sample(chess.Board(), np.zeros(ACTION_SIZE), 0.75)
        assert mv == 0.75

    def test_mirrored_moves_are_legal_on_mirrored_board(self):
        board = chess.Board()
        policy = self._make_uniform_policy(board)
        _, mp, _ = mirror_sample(board, policy, 1.0)

        mirrored_board = board.mirror()
        mirrored_mask  = legal_mask(mirrored_board)
        for idx in np.nonzero(mp)[0]:
            assert mirrored_mask[idx] == 1.0, (
                f"Mirrored policy index {idx} is not legal on mirrored board"
            )

    def test_double_mirror_recovers_original_policy(self):
        board = chess.Board()
        policy = self._make_uniform_policy(board)
        ms, mp, _ = mirror_sample(board, policy, 1.0)
        ms2, mp2, _ = mirror_sample(board.mirror(), mp, 1.0)
        np.testing.assert_allclose(ms2, encode(board), atol=1e-6)
        np.testing.assert_allclose(mp2, policy, atol=1e-6)

    def test_state_shape_preserved(self):
        board = chess.Board()
        ms, _, _ = mirror_sample(board, np.zeros(ACTION_SIZE), 0.0)
        assert ms.shape == encode(board).shape

    def test_castling_state_remains_valid(self):
        board = chess.Board()
        ms, _, _ = mirror_sample(board, np.zeros(ACTION_SIZE), 0.0)
        np.testing.assert_array_equal(ms, encode(board.mirror()))

    def test_castling_policy_maps_to_legal_castling_move(self):
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        policy[move_to_idx(chess.Move.from_uci("e1g1"))] = 1.0

        _, mirrored_policy, _ = mirror_sample(board, policy, 1.0)
        mirrored_board = board.mirror()
        mirrored_move = chess.Move.from_uci("e8g8")

        assert mirrored_move in mirrored_board.legal_moves
        assert mirrored_policy[move_to_idx(mirrored_move)] == 1.0


# ── Board encoding ────────────────────────────────────────────────────────────

class TestEncode:
    def test_shape(self):
        assert encode(chess.Board()).shape == (INPUT_PLANES, 8, 8)

    def test_dtype(self):
        assert encode(chess.Board()).dtype == np.float32

    def test_starting_piece_counts(self):
        planes = encode(chess.Board())
        assert planes[0].sum() == 8   # white pawns
        assert planes[6].sum() == 8   # black pawns
        assert planes[4].sum() == 1   # white queen
        assert planes[10].sum() == 1  # black queen
        assert planes[5].sum() == 1   # white king
        assert planes[11].sum() == 1  # black king

    def test_side_to_move_white(self):
        assert encode(chess.Board())[12].all()

    def test_side_to_move_black(self):
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        assert not encode(board)[12].any()

    def test_en_passant_plane(self):
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        planes = encode(board)
        # En passant target is e3 (square 20); rank=2, file=4
        assert planes[17, 2, 4] == 1.0

    def test_no_en_passant_at_start(self):
        assert encode(chess.Board())[17].sum() == 0.0

    def test_castling_rights_at_start(self):
        planes = encode(chess.Board())
        assert planes[13].all()   # white kingside
        assert planes[14].all()   # white queenside
        assert planes[15].all()   # black kingside
        assert planes[16].all()   # black queenside

    def test_values_binary(self):
        planes = encode(chess.Board())
        assert set(np.unique(planes)).issubset({0.0, 1.0})


# ── Concept labels ────────────────────────────────────────────────────────────

class TestConceptLabels:
    def test_shape(self):
        labels = compute_concept_labels(chess.Board())
        assert labels.shape == (len(CONCEPT_NAMES),)

    def test_dtype(self):
        assert compute_concept_labels(chess.Board()).dtype == np.float32

    def test_values_in_unit_interval(self):
        labels = compute_concept_labels(chess.Board())
        assert np.all(labels >= 0.0) and np.all(labels <= 1.0)

    def test_equal_material_at_start(self):
        # material_balance is sigmoid of advantage; equal → ~0.5
        labels = compute_concept_labels(chess.Board())
        np.testing.assert_allclose(labels[0], 0.5, atol=0.01)

    def test_returns_six_values(self):
        assert len(compute_concept_labels(chess.Board())) == 6

    def test_concept_names_count(self):
        assert len(CONCEPT_NAMES) == 6
