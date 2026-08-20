#!/usr/bin/env bash
# Measure the ceiling: the fixed reader on an oracle-built access layer.
# Runs after the v11 chain so the two do not contend for the API.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH="$PWD/src"
while pgrep -f "retrain_v11.sh" > /dev/null; do sleep 120; done
python3 scripts/build_oracle_index.py --out data/v10L_oracle
for M in 15 8; do
  python3 -m kb.test --kb data/v10L_oracle/universe.json \
    --universe data/v10L/universe.json --split both --epoch 0 --m $M \
    --model gpt-5-mini --out runs/v11base_oracle_m$M > runs/v11base_oracle_m$M.log 2>&1
  echo "oracle M=$M done"
done
echo "ORACLE DONE"
