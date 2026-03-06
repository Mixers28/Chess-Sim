"""
Chess environment: board encoding, move encoding, legal move masks.

Board is encoded as a 13×8×8 tensor:
  Planes 0–5 : white pieces  (P N B R Q K)
  Planes 6–11: black pieces  (P N B R Q K)
  Plane 12   : side to move  (1.0 = white, 0.0 = black)

Moves are encoded as integers 0–4095  (from_square * 64 + to_square).
Promotions are always to queen.
"""

import chess
import numpy as np

# piece_type (1-indexed) → plane offset
_PT_OFFSET = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4,  chess.KING: 5,
}


def encode(board: chess.Board) -> np.ndarray:
    """Return 13×8×8 float32 array representing the board."""
    planes = np.zeros((13, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            offset = 0 if piece.color == chess.WHITE else 6
            planes[offset + _PT_OFFSET[piece.piece_type], sq >> 3, sq & 7] = 1.0
    planes[12] = 1.0 if board.turn == chess.WHITE else 0.0
    return planes


def legal_mask(board: chess.Board) -> np.ndarray:
    """Return (4096,) float32 mask: 1 for each legal move index."""
    mask = np.zeros(4096, dtype=np.float32)
    for mv in board.legal_moves:
        # Collapse all promotions to queen — same (from, to) index
        if mv.promotion is None or mv.promotion == chess.QUEEN:
            mask[mv.from_square * 64 + mv.to_square] = 1.0
    return mask


def idx_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Convert a move index back to a legal chess.Move (queen promotion if needed)."""
    from_sq, to_sq = idx >> 6, idx & 63
    for promo in (None, chess.QUEEN):
        mv = chess.Move(from_sq, to_sq, promotion=promo)
        if mv in board.legal_moves:
            return mv
    # Fallback: scan legal moves for matching squares
    for mv in board.legal_moves:
        if mv.from_square == from_sq and mv.to_square == to_sq:
            return mv
    return None
