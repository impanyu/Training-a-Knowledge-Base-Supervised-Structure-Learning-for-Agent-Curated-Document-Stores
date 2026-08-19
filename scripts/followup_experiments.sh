#!/usr/bin/env bash
# Runs after the M=8 exam finishes: the M=5 budget point (completes E1's
# accuracy-budget curve) and the dense learning curve (Fig 2a from 3 to ~13
# points). Waits by polling the M=8 log's line count -- no pgrep self-match.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH="$PWD/src"

TARGET=150
M8=runs/v10L_trained_m8/test_log.jsonl

echo "=== waiting for M=8 exam ($TARGET questions) ==="
while :; do
  n=$([ -f "$M8" ] && wc -l < "$M8" || echo 0)
  [ "$n" -ge "$TARGET" ] && break
  sleep 60
done
echo "M=8 done ($n rows)"

# --- E1 third budget point: M=5, trained and B1 ---
echo "=== M=5: trained store ==="
python3 -m kb.test --kb runs/v10L_dedup/kb_epoch_2.json \
  --universe data/v10L/universe.json --split both --epoch 2 --m 5 \
  --model gpt-5-mini --out runs/v10L_trained_m5 \
  > runs/v10L_trained_m5.log 2>&1

echo "=== M=5: B1 flat store ==="
python3 -m kb.test --kb data/v10L/universe.json \
  --universe data/v10L/universe.json --split both --epoch 0 --m 5 \
  --model gpt-5-mini --out runs/v10L_b1_m5 \
  > runs/v10L_b1_m5.log 2>&1

# --- dense learning curve: eval split on replayed snapshots ---
echo "=== dense learning curve ==="
for IT in 25 50 75 100 125 175 200 225 250 275; do
  SNAP=/tmp/kbdense_$IT.json
  python3 -m kb.replay --universe data/v10L/universe.json \
    --trace runs/v10L_dedup/trace.jsonl --at "$IT" --embed --out "$SNAP" \
    >> runs/dense_curve.log 2>&1
  python3 -m kb.test --kb "$SNAP" --universe data/v10L/universe.json \
    --split eval --epoch 1 --model gpt-5-mini --out "runs/v10L_dense_$IT" \
    >> runs/dense_curve.log 2>&1
  echo "  iteration $IT done"
done

echo "DONE"
