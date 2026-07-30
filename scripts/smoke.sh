#!/usr/bin/env bash
# Tiny live run: 3 questions, L6 single agent, ~15 LLM calls. Costs cents.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
MODEL="${1:-deepseek-v4-flash}"
python3 scripts/prepare_data.py --hotpot-n 3 --musique-n 0 --out data/smoke
python3 -m ca.runner --level L6 --questions data/smoke/pool.jsonl \
  --index data/smoke/index --seed 0 --capital 30000 --max-rounds 15 \
  --model "$MODEL" --out "runs/smoke_L6_$MODEL"
echo "=== metrics ==="
cat "runs/smoke_L6_$MODEL/metrics.json"
