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

import math

import chess
import numpy as np

INPUT_PLANES = 19
ACTION_SIZE  = 8192   # 4096 standard + 4096 knight underpromotions

N_CONCEPTS = 6
CONCEPT_NAMES = [
    "material_balance", "king_safety", "piece_mobility",
    "pawn_structure",   "space_control", "tactical_threat",
]

_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9,  chess.KING: 0,
}

# Extended centre: files C-F × ranks 3-6 (16 squares)
_EXTENDED_CENTRE = (
    (chess.BB_RANK_3 | chess.BB_RANK_4 | chess.BB_RANK_5 | chess.BB_RANK_6)
    & (chess.BB_FILE_C | chess.BB_FILE_D | chess.BB_FILE_E | chess.BB_FILE_F)
)


def compute_concept_labels(board: chess.Board) -> np.ndarray:
    """
    Compute 6 transferable strategic concept scores from the board.
    Returns float32 (6,) with all values in [0, 1], from the side-to-move's perspective.

    Concept → logistics/security analogue:
      material_balance  → margin_headroom
      king_safety       → critical_node_risk (inverted)
      piece_mobility    → route_optionality
      pawn_structure    → supply_chain_dependency
      space_control     → network_coverage
      tactical_threat   → disruption_probability
    """
    side = board.turn
    opp  = not side

    # 1. material_balance — sigmoid of side-to-move's material advantage
    mat = {chess.WHITE: 0, chess.BLACK: 0}
    for p in board.piece_map().values():
        mat[p.color] += _PIECE_VALUES[p.piece_type]
    adv = mat[side] - mat[opp]
    material_balance = 1.0 / (1.0 + math.exp(-adv / 3.0))

    # 2. king_safety — fewer attacker vs more pawn shield → safer
    king_sq = board.king(side)
    attacker_count = len(board.attackers(opp, king_sq)) if king_sq is not None else 0
    shield = 0
    if king_sq is not None:
        kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
        shield_rank = kr + (1 if side == chess.WHITE else -1)
        if 0 <= shield_rank <= 7:
            for f in range(max(0, kf - 1), min(8, kf + 2)):
                p = board.piece_at(chess.square(f, shield_rank))
                if p and p.piece_type == chess.PAWN and p.color == side:
                    shield += 1
    king_safety = 1.0 / (1.0 + math.exp(attacker_count - shield))

    # 3. piece_mobility — legal move count normalised (40 ≈ typical mean)
    n_legal = board.legal_moves.count()
    piece_mobility = min(n_legal / 40.0, 1.0)

    # 4. pawn_structure — penalise doubled and isolated pawns
    pawns = list(board.pieces(chess.PAWN, side))
    if pawns:
        pfiles   = [chess.square_file(sq) for sq in pawns]
        fset     = set(pfiles)
        doubled  = sum(1 for f in fset if pfiles.count(f) > 1)
        isolated = sum(1 for f in fset if (f - 1) not in fset and (f + 1) not in fset)
        pawn_structure = max(0.0, 1.0 - (doubled + isolated) / (2.0 * len(pawns)))
    else:
        pawn_structure = 0.5

    # 5. space_control — fraction of extended centre squares we attack
    our_attacks = chess.BB_EMPTY
    for sq in chess.scan_forward(board.occupied_co[side]):
        our_attacks |= board.attacks_mask(sq)
    space_control = bin(our_attacks & _EXTENDED_CENTRE).count("1") / 16.0

    # 6. tactical_threat — fraction of legal moves that are captures
    captures = sum(1 for mv in board.legal_moves if board.is_capture(mv))
    tactical_threat = captures / max(n_legal, 1)

    return np.array(
        [material_balance, king_safety, piece_mobility,
         pawn_structure, space_control, tactical_threat],
        dtype=np.float32,
    )

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
