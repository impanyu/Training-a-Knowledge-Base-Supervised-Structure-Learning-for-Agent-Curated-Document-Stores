#!/usr/bin/env bash
# External-validity arm: the same protocol on an official PhantomWiki
# generation, so the result does not rest on a universe we wrote ourselves.
#
# Only B1 and the trained store are measured here. The GraphRAG and HippoRAG2
# adapters extract triples with regexes written for kb.build's templates, and
# PhantomWiki renders different sentence forms, so running them unchanged
# would measure a broken extractor rather than the baseline. PhantomWiki also
# ships no support sets, so the structure-semantics alignment analysis is
# unavailable on this arm.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH="$PWD/src"

echo "=== pw1 training: 2 epochs x 150 questions ==="
python3 -m kb.train --universe data/pw1/universe.json --epochs 2 \
  --seed 0 --model gpt-5-mini --n1 15 --n2 30 --m 15 \
  --out runs/pw1_main --eval-each-epoch > runs/pw1_main.log 2>&1

echo "=== pw1 frozen exams ==="
for M in 15 8; do
  python3 -m kb.test --kb runs/pw1_main/kb_epoch_2.json \
    --universe data/pw1/universe.json --split both --epoch 2 --m $M \
    --model gpt-5-mini --out runs/pw1_trained_m$M > runs/pw1_trained_m$M.log 2>&1
  python3 -m kb.test --kb data/pw1/universe.json \
    --universe data/pw1/universe.json --split both --epoch 0 --m $M \
    --model gpt-5-mini --out runs/pw1_b1_m$M > runs/pw1_b1_m$M.log 2>&1
  echo "  M=$M done"
done
python3 -m kb.test --kb runs/pw1_main/kb_epoch_2.json \
  --universe data/pw1/universe.json --split train --epoch 2 \
  --model gpt-5-mini --out runs/pw1_trained_train > runs/pw1_trained_train.log 2>&1
python3 -m kb.test --kb data/pw1/universe.json --universe data/pw1/universe.json \
  --split train --epoch 0 --model gpt-5-mini --out runs/pw1_b1_train \
  > runs/pw1_b1_train.log 2>&1

python3 scripts/note_character.py --kb runs/pw1_main/kb_epoch_2.json > runs/pw1_structure.txt
python3 scripts/index_taxonomy.py --kb runs/pw1_main/kb_epoch_2.json \
  --universe data/pw1/universe.json --examples 2 > runs/pw1_taxonomy.txt || true
echo "PW1 DONE"
