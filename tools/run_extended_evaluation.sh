#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/venv/bin/python}"
GAMES_PER_OPPONENT="${GAMES_PER_OPPONENT:-100}"
MCTS_SIMS="${MCTS_SIMS:-100}"
STOCKFISH_SKILL="${STOCKFISH_SKILL:-1}"
STOCKFISH_MOVETIME="${STOCKFISH_MOVETIME:-0.05}"
STOCKFISH_THREADS="${STOCKFISH_THREADS:-1}"
STOCKFISH_HASH_MB="${STOCKFISH_HASH_MB:-64}"
MAX_MOVES="${MAX_MOVES:-120}"
BENCHMARK_SEED="${BENCHMARK_SEED:-20260614}"
RESUME=false
REPORT=""

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi

if [[ "${1:-}" == "--resume" ]]; then
  RESUME=true
  REPORT="${2:-}"
  if [[ -z "$REPORT" || ! -f "$REPORT" ]]; then
    echo "Usage: $0 --resume benchmark/extended/report.json" >&2
    exit 1
  fi
  read -r CANDIDATE GAMES_PER_OPPONENT MCTS_SIMS MAX_MOVES \
    MATERIAL_ADJUDICATION BENCHMARK_SEED STOCKFISH_BIN STOCKFISH_SKILL \
    STOCKFISH_MOVETIME STOCKFISH_THREADS STOCKFISH_HASH_MB < <(
    "$PYTHON" - "$REPORT" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    report = json.load(f)
stockfish = report["stockfish"]
print(
    report["checkpoint"],
    report["n_games"],
    report["n_sims"],
    report["max_moves"],
    report.get("material_adjudication", 3.0),
    report["seed"],
    stockfish["path"],
    stockfish["skill"],
    stockfish["movetime"],
    stockfish["threads"],
    stockfish["hash_mb"],
)
PY
  )
else
  CANDIDATE="${1:-$ROOT_DIR/checkpoint/model.pt}"
  MATERIAL_ADJUDICATION="${MATERIAL_ADJUDICATION:-3.0}"
fi

if [[ ! -f "$CANDIDATE" ]]; then
  echo "Candidate checkpoint not found: $CANDIDATE" >&2
  exit 1
fi

STOCKFISH_BIN="${STOCKFISH_BIN:-$(command -v stockfish || true)}"
if [[ -z "$STOCKFISH_BIN" && -x /usr/games/stockfish ]]; then
  STOCKFISH_BIN=/usr/games/stockfish
fi
if [[ -z "$STOCKFISH_BIN" ]]; then
  echo "Stockfish not found. Set STOCKFISH_BIN=/path/to/stockfish." >&2
  exit 1
fi

read -r MODEL_VERSION TRAINING_GAMES < <(
  "$PYTHON" - "$CANDIDATE" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
version = str(checkpoint.get("model_version", "unversioned"))
games = int(checkpoint.get("training_games", 0))
print(version.replace("/", "_"), games)
PY
)

mkdir -p benchmark/candidates benchmark/extended

if [[ "$RESUME" == true ]]; then
  SNAPSHOT="$CANDIDATE"
  OUTPUT="$REPORT"
  LOG="${REPORT%.json}.log"
else
  TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  SNAPSHOT="benchmark/candidates/${MODEL_VERSION}_${TIMESTAMP}.pt"
  OUTPUT="benchmark/extended/${MODEL_VERSION}_${TIMESTAMP}.json"
  LOG="benchmark/extended/${MODEL_VERSION}_${TIMESTAMP}.log"
  cp "$CANDIDATE" "$SNAPSHOT"
fi

echo "Extended evaluation"
echo "  Candidate: $CANDIDATE"
echo "  Snapshot:  $SNAPSHOT"
echo "  Version:   $MODEL_VERSION ($TRAINING_GAMES training games)"
echo "  Match:     $GAMES_PER_OPPONENT games each vs heuristic and Stockfish"
echo "  Search:    $MCTS_SIMS MCTS simulations/move"
echo "  Stockfish: skill=$STOCKFISH_SKILL time=${STOCKFISH_MOVETIME}s"
echo "  Cap rule:   ${MATERIAL_ADJUDICATION}-pawn material adjudication"
echo "  Output:    $OUTPUT"

COMMAND=(
  "$PYTHON" benchmark.py
  --checkpoint "$SNAPSHOT" \
  --games "$GAMES_PER_OPPONENT" \
  --sims "$MCTS_SIMS" \
  --vs heuristic stockfish \
  --stockfish "$STOCKFISH_BIN" \
  --skill "$STOCKFISH_SKILL" \
  --stockfish-movetime "$STOCKFISH_MOVETIME" \
  --stockfish-threads "$STOCKFISH_THREADS" \
  --stockfish-hash "$STOCKFISH_HASH_MB" \
  --max-moves "$MAX_MOVES" \
  --material-adjudication "$MATERIAL_ADJUDICATION" \
  --seed "$BENCHMARK_SEED" \
  --output "$OUTPUT"
)
if [[ "$RESUME" == true ]]; then
  COMMAND+=(--resume)
fi

set +e
"${COMMAND[@]}" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

if [[ $STATUS -eq 130 ]]; then
  echo "Evaluation interrupted. Resume with:"
  echo "  $0 --resume $OUTPUT"
  exit 130
fi
if [[ $STATUS -ne 0 ]]; then
  exit "$STATUS"
fi

echo "Evaluation complete."
echo "  Report: $OUTPUT"
echo "  Log:    $LOG"
