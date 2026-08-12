#!/usr/bin/env bash
# Tiny live run: 6 questions, C7 single agent, ~15 LLM calls. Costs cents.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
MODEL="${1:-deepseek-v4-flash}"
python3 scripts/prepare_data.py --hotpot-n 6 --musique-n 0 --out data/smoke
python3 scripts/build_bank.py --pool data/smoke/pool.jsonl --out data/smoke/bank.json \
  --topics 1 --seed 0
python3 -m ca.runner --level C7 --bank data/smoke/bank.json \
  --seed 0 --capital 30000 --max-rounds 15 \
  --model "$MODEL" --out "runs/smoke_C7_$MODEL"
echo "=== metrics ==="
cat "runs/smoke_C7_$MODEL/metrics.json"
