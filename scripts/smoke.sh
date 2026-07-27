#!/usr/bin/env bash
# Tiny live run: 3 questions, L5 single agent, ~10 LLM calls. Costs cents.
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -e ".[data,dev]" -q
python scripts/prepare_data.py --hotpot-n 3 --musique-n 0 --out data/smoke
python -m ca.runner --level L5 --questions data/smoke/pool.jsonl \
  --index data/smoke/index --seed 0 --capital 30000 --max-rounds 15 \
  --out runs/smoke_L5
echo "=== metrics ==="
cat runs/smoke_L5/metrics.json
