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

Move encoding (ACTION_SIZE = 8192):
  0–4095    : standard moves  (from_square * 64 + to_square)
              queen promotion is implicit for pawns reaching the last rank.
  4096–8191 : knight underpromotions  (4096 + from_square * 64 + to_square)
"""

import chess
import numpy as np

INPUT_PLANES = 19
ACTION_SIZE  = 8192   # 4096 standard + 4096 knight underpromotions

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


def move_to_idx(mv: chess.Move) -> int:
    """Encode a chess.Move to an action index (0–8191)."""
    base = mv.from_square * 64 + mv.to_square
    return 4096 + base if mv.promotion == chess.KNIGHT else base


def legal_mask(board: chess.Board) -> np.ndarray:
    """Return (ACTION_SIZE,) float32 mask: 1 for each legal move index."""
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for mv in board.legal_moves:
        mask[move_to_idx(mv)] = 1.0
    return mask


def idx_to_move(idx: int, board: chess.Board) -> chess.Move | None:
    """Convert an action index back to a legal chess.Move."""
    if idx >= 4096:
        base    = idx - 4096
        from_sq = base >> 6
        to_sq   = base & 63
        mv = chess.Move(from_sq, to_sq, promotion=chess.KNIGHT)
        return mv if mv in board.legal_moves else None

    from_sq = idx >> 6
    to_sq   = idx & 63
    piece   = board.piece_at(from_sq)
    if piece and piece.piece_type == chess.PAWN:
        if (piece.color == chess.WHITE and chess.square_rank(to_sq) == 7) or \
           (piece.color == chess.BLACK and chess.square_rank(to_sq) == 0):
            mv = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
            return mv if mv in board.legal_moves else None
    mv = chess.Move(from_sq, to_sq)
    return mv if mv in board.legal_moves else None


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
        if idx >= 4096:
            base     = idx - 4096
            new_base = (base >> 6 ^ 7) * 64 + (base & 63 ^ 7)
            mp[4096 + new_base] = policy[idx]
        else:
            new_idx = (idx >> 6 ^ 7) * 64 + (idx & 63 ^ 7)
            mp[new_idx] = policy[idx]

    return ms, mp, value
