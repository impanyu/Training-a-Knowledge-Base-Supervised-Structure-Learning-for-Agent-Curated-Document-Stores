#!/usr/bin/env bash
# Tiny live run: 6 questions, ~15 LLM calls per config. Costs cents.
# Configs: C0 C1 C2 C5 C7 (default: C7, the solo baseline).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
MODEL="${1:-deepseek-v4-flash}"
LEVEL="${2:-C7}"
python3 scripts/prepare_data.py --hotpot-n 6 --musique-n 0 --out data/smoke
python3 scripts/build_bank.py --pool data/smoke/pool.jsonl --out data/smoke/bank.json \
  --topics 1 --seed 0
python3 -m ca.runner --level "$LEVEL" --bank data/smoke/bank.json \
  --seed 0 --max-rounds 15 \
  --model "$MODEL" --out "runs/smoke_${LEVEL}_$MODEL"
echo "=== metrics ==="
cat "runs/smoke_${LEVEL}_$MODEL/metrics.json"
