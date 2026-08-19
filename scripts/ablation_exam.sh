#!/usr/bin/env bash
# Causal test of what the agent built: intervene on the STORE, hold reader,
# questions and budget fixed. Runs after the other queued chains.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH="$PWD/src"

echo "=== waiting for queued chains ==="
while pgrep -f "followup_experiments.sh|train_split_exam.sh" > /dev/null; do sleep 60; done

python3 scripts/make_ablations.py --drop nav   --out /tmp/kb_no_nav.json
python3 scripts/make_ablations.py --drop links --out /tmp/kb_no_links.json

echo "=== trained store minus navigation documents ==="
python3 -m kb.test --kb /tmp/kb_no_nav.json --universe data/v10L/universe.json \
  --split both --epoch 2 --model gpt-5-mini --out runs/v10L_abl_nonav \
  > runs/v10L_abl_nonav.log 2>&1

echo "=== trained store minus links ==="
python3 -m kb.test --kb /tmp/kb_no_links.json --universe data/v10L/universe.json \
  --split both --epoch 2 --model gpt-5-mini --out runs/v10L_abl_nolinks \
  > runs/v10L_abl_nolinks.log 2>&1

echo "ABLATIONS DONE"
