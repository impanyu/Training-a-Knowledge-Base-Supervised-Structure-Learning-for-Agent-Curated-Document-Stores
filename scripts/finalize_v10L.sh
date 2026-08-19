#!/usr/bin/env bash
# Final exam for the v10L dedup run: trained store at both budgets, then report.
# B1 (untrained) was already measured in runs/v10L_b1_m{15,8}.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH="$PWD/src"

RUN=runs/v10L_dedup
SNAP=$RUN/kb_epoch_2.json

[ -f "$SNAP" ] || { echo "missing $SNAP"; exit 1; }

echo "=== final exam: trained store, M=15 ==="
python3 -m kb.test --kb "$SNAP" --universe data/v10L/universe.json \
  --split both --epoch 2 --model gpt-5-mini --out runs/v10L_trained_m15 \
  > runs/v10L_trained_m15.log 2>&1

echo "=== final exam: trained store, M=8 ==="
python3 -m kb.test --kb "$SNAP" --universe data/v10L/universe.json \
  --split both --epoch 2 --m 8 --model gpt-5-mini --out runs/v10L_trained_m8 \
  > runs/v10L_trained_m8.log 2>&1

echo "=== per-iteration store trajectory (replay, no API) ==="
python3 -m kb.replay --universe data/v10L/universe.json \
  --trace $RUN/trace.jsonl --series stats --out $RUN/series.jsonl

echo "=== report ==="
python3 -m kb.report $RUN > $RUN/report.txt 2>&1 || true
echo "DONE"
