#!/usr/bin/env bash
# Pilot for the class-generalization prompt (objectives 3-4 rewritten).
# Success criterion, measured on the authored notes the pilot writes:
# the entity-coverage distribution must shift right of the v10L baseline
# (55% of notes cover <=1 entity, 1% cover 10+; mean 2.07).
# 40 iterations is enough to characterize what the agent writes without
# paying for a full 300-iteration run.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH="$PWD/src"

echo "=== waiting for queued chains ==="
while pgrep -f "followup_experiments.sh|train_split_exam.sh|ablation_exam.sh" > /dev/null; do
  sleep 60
done

echo "=== generalization pilot: 40 iterations ==="
python3 -m kb.train --universe data/v10L/universe.json --epochs 1 \
  --train-size 40 --seed 0 --model gpt-5-mini --out runs/v11_genpilot \
  > runs/v11_genpilot.log 2>&1

echo "=== authored-note character vs baseline ==="
python3 scripts/note_character.py --kb runs/v11_genpilot/kb_epoch_1.json \
  --baseline runs/v10L_dedup/kb_epoch_2.json
echo "PILOT DONE"
