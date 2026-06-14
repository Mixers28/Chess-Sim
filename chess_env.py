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
  4096–4239 : compact underpromotions (knight, bishop, rook)
  4240–8191 : reserved
"""

import math

import chess
import numpy as np

INPUT_PLANES = 19
ACTION_SIZE  = 8192   # 4096 standard + compact underpromotions + reserved space
_UNDERPROMOTION_OFFSET = 4096
_UNDERPROMOTION_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
_UNDERPROMOTION_ACTIONS = 2 * 8 * 3 * len(_UNDERPROMOTION_PIECES)

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
    Compute 6 strategic concept scores from the board.
    Returns float32 (6,) with all values in [0, 1], from the side-to-move's perspective.

    Concepts: material_balance, king_safety, piece_mobility,
    pawn_structure, space_control, tactical_threat.
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


def narrate_concepts(concepts) -> str:
    """
    Convert 6 concept scores to a plain-English position description.
    All scores are from the side-to-move's perspective, in [0, 1].
    0.5 ≈ neutral; higher = better for the moving side.
    """
    if len(concepts) < 6:
        return ""
    mb, ks, pm, ps, sc, tt = (float(c) for c in concepts[:6])

    parts = []

    # Material balance (sigmoid: 0.5 = equal, 0.62 ≈ +3 pawns ahead)
    if mb > 0.62:
        parts.append("I'm up on material")
    elif mb < 0.38:
        parts.append("I'm down on material")
    else:
        parts.append("material is equal")

    # King safety (sigmoid of shield − attackers)
    if ks < 0.35:
        parts.append("my king is under pressure")
    elif ks > 0.70:
        parts.append("my king is safe")

    # Piece mobility (legal_moves / 40; 0.5 ≈ 20 moves, 1.0 ≈ 40+ moves)
    if pm > 0.65:
        parts.append("my pieces are active")
    elif pm < 0.35:
        parts.append("my pieces are cramped")

    # Pawn structure (1 − islands/4)
    if ps < 0.35:
        parts.append("my pawns are weak")
    elif ps > 0.70:
        parts.append("my pawn structure is solid")

    # Space control (fraction of center squares attacked)
    if sc > 0.60:
        parts.append("I control more space")
    elif sc < 0.25:
        parts.append("I'm short on space")

    # Tactical threat (captures / legal_moves)
    if tt > 0.35:
        parts.append("there are sharp tactics")
    elif tt > 0.20:
        parts.append("I see a tactical opportunity")

    if not parts:
        return "The position is balanced."

    first = parts[0].capitalize()
    if len(parts) == 1:
        return first + "."
    elif len(parts) == 2:
        return first + ", and " + parts[1] + "."
    else:
        return first + ", " + ", ".join(parts[1:-1]) + ", and " + parts[-1] + "."

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
    if mv.promotion not in _UNDERPROMOTION_PIECES:
        return base

    from_rank = chess.square_rank(mv.from_square)
    to_rank = chess.square_rank(mv.to_square)
    if from_rank == 6 and to_rank == 7:
        color_index = 0
    elif from_rank == 1 and to_rank == 0:
        color_index = 1
    else:
        raise ValueError(f"invalid underpromotion geometry: {mv.uci()}")

    from_file = chess.square_file(mv.from_square)
    file_delta = chess.square_file(mv.to_square) - from_file
    if file_delta not in (-1, 0, 1):
        raise ValueError(f"invalid underpromotion geometry: {mv.uci()}")

    piece_index = _UNDERPROMOTION_PIECES.index(mv.promotion)
    code = (((color_index * 8 + from_file) * 3 + (file_delta + 1)) * 3
            + piece_index)
    return _UNDERPROMOTION_OFFSET + code


def legal_mask(board: chess.Board) -> np.ndarray:
    """Return (ACTION_SIZE,) float32 mask: 1 for each legal move index."""
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for mv in board.legal_moves:
        mask[move_to_idx(mv)] = 1.0
    return mask


def idx_to_move(idx: int, board: chess.Board) -> chess.Move | None:
    """Convert an action index back to a legal chess.Move."""
    if idx < 0 or idx >= ACTION_SIZE:
        return None

    if idx >= _UNDERPROMOTION_OFFSET:
        code = idx - _UNDERPROMOTION_OFFSET
        if code >= _UNDERPROMOTION_ACTIONS:
            return None
        piece_index = code % 3
        code //= 3
        file_delta = code % 3 - 1
        code //= 3
        from_file = code % 8
        color_index = code // 8
        to_file = from_file + file_delta
        if not 0 <= to_file < 8:
            return None
        from_rank, to_rank = ((6, 7) if color_index == 0 else (1, 0))
        mv = chess.Move(
            chess.square(from_file, from_rank),
            chess.square(to_file, to_rank),
            promotion=_UNDERPROMOTION_PIECES[piece_index],
        )
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


def mirror_sample(
    board: chess.Board,
    policy: np.ndarray,
    value: float,
):
    """
    Mirror a sample vertically while swapping piece colours.

    ``Board.mirror()`` is an exact chess symmetry: ranks are flipped, colours
    and side-to-move are swapped, and castling/en-passant state stays valid.
    A left-right file flip is not exact because standard castling is tied to
    the king's e-file starting square.

    Mirror rules:
      - Board: python-chess vertical mirror + colour swap
      - Policy: square_mirror on from/to squares; promotion type unchanged
      - Value: unchanged because it remains from side-to-move's perspective
    """
    mirrored_board = board.mirror()
    ms = encode(mirrored_board)
    # Board transforms do not retain move history, but repetition status is
    # invariant under this symmetry.
    ms[18] = encode(board)[18]

    mp = np.zeros_like(policy)
    for idx in np.nonzero(policy)[0]:
        mv = idx_to_move(int(idx), board)
        if mv is None:
            continue
        mirrored_move = chess.Move(
            chess.square_mirror(mv.from_square),
            chess.square_mirror(mv.to_square),
            promotion=mv.promotion,
        )
        if mirrored_move in mirrored_board.legal_moves:
            mp[move_to_idx(mirrored_move)] += policy[idx]

    return ms, mp, value
