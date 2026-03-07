"""
Chess environment: board encoding, move encoding, legal move masks.

Board is encoded as a 19×8×8 tensor:
  Planes  0–5  : white pieces      (P N B R Q K)
  Planes  6–11 : black pieces      (P N B R Q K)
  Plane  12    : side to move      (1.0 = white, 0.0 = black)
  Plane  13    : white kingside castling right
  Plane  14    : white queenside castling right
  Plane  15    : black kingside castling right
  Plane  16    : black queenside castling right
  Plane  17    : en passant target square
  Plane  18    : position has been seen before in this game (draw pressure)

Moves are encoded as integers 0–4095  (from_square * 64 + to_square).
Promotions are always to queen.
"""

import chess
import numpy as np

INPUT_PLANES = 19

# piece_type (1-indexed) → plane offset
_PT_OFFSET = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4,  chess.KING: 5,
}


def encode(board: chess.Board) -> np.ndarray:
    """Return 19×8×8 float32 array representing the board."""
    planes = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)

    # Planes 0-11: piece positions
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            offset = 0 if piece.color == chess.WHITE else 6
            planes[offset + _PT_OFFSET[piece.piece_type], sq >> 3, sq & 7] = 1.0

    # Plane 12: side to move
    planes[12] = 1.0 if board.turn == chess.WHITE else 0.0

    # Planes 13-16: castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16] = 1.0

    # Plane 17: en passant target square
    if board.ep_square is not None:
        planes[17, board.ep_square >> 3, board.ep_square & 7] = 1.0

    # Plane 18: position repetition (has this position appeared before in the game)
    if board.is_repetition(2):
        planes[18] = 1.0

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


def mirror_sample(state: np.ndarray, policy: np.ndarray, value: float):
    """
    Horizontally mirror a training sample (left-right board flip).

    Chess is symmetric along the vertical axis: a mirrored position is equally
    valid and provides free 2× training data per game.

    Mirror rules:
      - Board planes: flip file axis (axis=2)
      - Castling: kingside ↔ queenside planes swap (files reversed after flip)
      - En passant plane: correctly repositioned by the file flip
      - Policy: remap from_sq → (from_sq ^ 7), to_sq → (to_sq ^ 7)
        (XOR with 7 flips the file bits while preserving the rank)
      - Value: unchanged (symmetric position has the same game value)
    """
    # Flip all planes along the file axis
    ms = state[:, :, ::-1].copy()

    # After the file flip, what was kingside (h-file) is now on the a-file side
    # (conceptually queenside), so swap the castling rights planes.
    ms[13], ms[14] = ms[14].copy(), ms[13].copy()   # white K/Q-side
    ms[15], ms[16] = ms[16].copy(), ms[15].copy()   # black K/Q-side

    # Remap policy: mirror each move index by flipping file bits on both squares
    mp = np.zeros_like(policy)
    for idx in np.nonzero(policy)[0]:
        idx = int(idx)
        new_idx = (idx >> 6 ^ 7) * 64 + (idx & 63 ^ 7)
        mp[new_idx] = policy[idx]

    return ms, mp, value
