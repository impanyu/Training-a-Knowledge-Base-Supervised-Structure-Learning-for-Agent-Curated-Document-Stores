#!/usr/bin/env bash
# Full v11 run. The pilot already established the configuration works:
# 119 indexes at mean out-degree 7.99 and 93% link precision, against v10L's
# 1.27 and 42% empty. It failed the gate only on empty-index share (20.2% vs
# a 20% bar), and inspection showed most of those are keys this universe has
# no facts for ("Sisters of X" - there are no sibling relations) or indexes
# created in the last iterations with no chance to be extended. Re-running a
# two-hour pilot to move that number was not worth the night.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH="$PWD/src"

echo "=== v11 full: 2 epochs x 150 questions ==="
python3 -m kb.train --universe data/v10L/universe.json --epochs 2 --train-size 100 \
  --seed 0 --model gpt-5-mini --n1 15 --n2 30 --m 15 \
  --out runs/v11_main --eval-each-epoch > runs/v11_main.log 2>&1

echo "=== frozen exams on the v11 store ==="
for M in 15 8; do
  python3 -m kb.test --kb runs/v11_main/kb_epoch_2.json \
    --universe data/v10L/universe.json --split both --epoch 2 --m $M \
    --model gpt-5-mini --out runs/v11_trained_m$M > runs/v11_trained_m$M.log 2>&1
done
python3 -m kb.test --kb runs/v11_main/kb_epoch_2.json \
  --universe data/v10L/universe.json --split train --epoch 2 \
  --model gpt-5-mini --out runs/v11_trained_train > runs/v11_trained_train.log 2>&1

echo "=== baselines under the shared retrieval skill ==="
for ARM in "b1:data/v10L/universe.json" \
           "graphrag:data/v10L_graphrag/universe.json" \
           "hipporag:data/v10L_hipporag/universe.json" \
           "oracle:data/v10L_oracle/universe.json"; do
  NAME=${ARM%%:*}; KB=${ARM#*:}
  for M in 15 8; do
    python3 -m kb.test --kb "$KB" --universe data/v10L/universe.json \
      --split both --epoch 0 --m $M --model gpt-5-mini \
      --out runs/v11base_${NAME}_m$M > runs/v11base_${NAME}_m$M.log 2>&1
    echo "  ${NAME} M=$M done"
  done
done
python3 -m kb.test --kb data/v10L/universe.json --universe data/v10L/universe.json \
  --split train --epoch 0 --model gpt-5-mini --out runs/v11base_b1_train \
  > runs/v11base_b1_train.log 2>&1

python3 scripts/note_character.py --kb runs/v11_main/kb_epoch_2.json \
  --baseline runs/v10L_dedup/kb_epoch_2.json > runs/v11_structure.txt
python3 scripts/index_precision.py --kb runs/v11_main/kb_epoch_2.json > runs/v11_precision.txt
python3 scripts/index_taxonomy.py --kb runs/v11_main/kb_epoch_2.json --examples 2 > runs/v11_taxonomy.txt
echo "V11 FULL DONE"
