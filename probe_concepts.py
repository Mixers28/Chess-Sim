import chess
import torch
from chess_env import encode, compute_concept_labels, CONCEPT_NAMES
from chess_model import policy_net, device

def probe(label, board):
    x = torch.tensor(encode(board), dtype=torch.float32).unsqueeze(0).to(device)
    policy_net.eval()
    with torch.no_grad():
        _, _, concepts = policy_net(x)
    policy_net.train()
    pred = concepts.squeeze(0).cpu().numpy()
    lbl = compute_concept_labels(board)
    print(f"\n--- {label} ---")
    print(f"{'Concept':<22} {'Predicted':>10} {'Label':>10}")
    print("-" * 44)
    for name, p, l in zip(CONCEPT_NAMES, pred, lbl):
        print(f"{name:<22} {p:>10.3f} {l:>10.3f}")

# Starting position (equal)
probe("Starting position", chess.Board())

# White up a queen
probe("White up material", chess.Board(
    "rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPPQPPP/RNB1KBNR b KQkq - 0 1"
))

# Exposed black king (e8 open after castling rights gone)
probe("Black king exposed", chess.Board(
    "rnb2bnr/pppp1ppp/8/4p3/4P1kQ/8/PPPP1PPP/RNB1KBNR b KQ - 0 1"
))
