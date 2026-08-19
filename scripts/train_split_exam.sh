#!/usr/bin/env bash
# Clean train/test generalization gap: the SAME 150 training questions under
# frozen stores -- initial (untrained) vs final (epoch 2). The epoch1-vs-epoch2
# contrast from the training log is confounded, because the store evolves
# within epoch 1; this is the controlled version.
# Waits for the followup chain to finish so the two do not contend for the API.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH="$PWD/src"

echo "=== waiting for the followup chain (M=5 + dense curve) ==="
while pgrep -f "followup_experiments.sh" > /dev/null; do sleep 60; done
echo "followup chain finished"

echo "=== train split: initial store (untrained) ==="
python3 -m kb.test --kb data/v10L/universe.json --universe data/v10L/universe.json \
  --split train --epoch 0 --model gpt-5-mini --out runs/v10L_b1_train \
  > runs/v10L_b1_train.log 2>&1

echo "=== train split: trained store (epoch 2) ==="
python3 -m kb.test --kb runs/v10L_dedup/kb_epoch_2.json --universe data/v10L/universe.json \
  --split train --epoch 2 --model gpt-5-mini --out runs/v10L_trained_train \
  > runs/v10L_trained_train.log 2>&1

echo "TRAIN SPLIT DONE"
