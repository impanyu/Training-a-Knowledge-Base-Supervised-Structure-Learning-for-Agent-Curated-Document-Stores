#!/usr/bin/env bash
# Tiny live run: 6 questions, 2 agents, ~20 LLM calls per arm. Costs cents.
# Arms: P0 (proactive) B0 (passive baseline). Default: P0.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
MODEL="${1:-deepseek-v4-flash}"
LEVEL="${2:-P0}"
python3 scripts/prepare_data.py --hotpot-n 6 --musique-n 0 --out data/smoke
python3 scripts/build_bank.py --pool data/smoke/pool.jsonl --out data/smoke/bank.json \
  --topics 1 --seed 0
python3 -m ca.runner --level "$LEVEL" --bank data/smoke/bank.json \
  --agents 2 --arrival-rate 1.0 --seed 0 --max-rounds 10 \
  --model "$MODEL" --out "runs/smoke_${LEVEL}_$MODEL"
echo "=== metrics ==="
cat "runs/smoke_${LEVEL}_$MODEL/metrics.json"
