#!/usr/bin/env bash
# v11: retrain with the index-construction protocol.
#
# The v10L store never became navigable -- 42% of its authored index notes
# carried no links and only 4.2% of facts were reachable from an index in one
# read -- so its null transfer tested the procedure, not the hypothesis. The
# revised objectives name indexes as the unit of construction and require
# EXTENDING an existing index rather than creating a new one.
#
# Stage 1 is a 40-iteration pilot with a structural gate; the full run starts
# automatically only if the gate passes, so a bad prompt costs ~$1.3 and an
# hour instead of ~$10 and ten hours.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH="$PWD/src"

echo "=== waiting for the M=5 / dense-curve chain ==="
while pgrep -f "followup_experiments.sh" > /dev/null; do sleep 60; done

echo "=== stage 1: 40-iteration pilot ==="
python3 -m kb.train --universe data/v10L/universe.json --epochs 1 \
  --train-size 40 --seed 0 --model gpt-5-mini --n1 15 --n2 20 --m 15 \
  --out runs/v11_pilot \
  > runs/v11_pilot.log 2>&1

python3 scripts/note_character.py --kb runs/v11_pilot/kb_epoch_1.json \
  --baseline runs/v10L_dedup/kb_epoch_2.json | tee runs/v11_pilot_structure.txt

# Gate: the two failures we measured must be fixed. Baseline was mean
# out-degree 1.27 with 42% of indexes empty.
python3 scripts/index_precision.py --kb runs/v11_pilot/kb_epoch_1.json \
  | tee runs/v11_pilot_precision.txt | head -3
python3 - <<'PY'
import json, re, sys, statistics as S
# Out-degree alone passed a store whose attribute indexes were 20% correct,
# so the gate scores precision too: an index full of the wrong notes misleads
# the reader more than an empty one does.
d = json.load(open("runs/v11_pilot/kb_epoch_1.json"))
nodes = d["store"]["nodes"]
by_id = {n["id"]: n for n in nodes}
idx = [n for n in nodes if n.get("flag") == "authored"]
if not idx:
    sys.exit("GATE FAIL: no index notes authored")
deg = [len(n.get("links", [])) for n in idx]
mean, empty = S.mean(deg), sum(1 for x in deg if x == 0) / len(deg)
NAME = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
uni = json.load(open("data/v10L/universe.json"))["vocab"]
vocab = list(uni.get("jobs", [])) + list(uni.get("hobbies", [])) + list(uni.get("cities", []))
tot = hit = 0
for n in idx:
    names = NAME.findall(n["text"])
    key = names[0] if len(names) == 1 else next(
        (v for v in vocab if re.search(rf"\b{re.escape(v)}\b", n["text"], re.I)), None)
    links = [t for t in n.get("links", []) if t in by_id]
    if not key or not links:
        continue
    tot += len(links)
    hit += sum(1 for t in links if re.search(re.escape(key), by_id[t]["text"], re.I))
prec = hit / tot if tot else 0.0
print(f"gate: {len(idx)} indexes, mean out-degree {mean:.2f}, empty {empty:.0%}, "
      f"precision {prec:.0%}")
if mean < 4.0 or empty > 0.20 or prec < 0.80:
    sys.exit(f"GATE FAIL: need mean>=4.0 ({mean:.2f}), empty<=20% ({empty:.0%}), "
             f"precision>=80% ({prec:.0%})")
print("GATE PASS")
PY

echo "=== stage 2: full run, 2 epochs x 150 questions ==="
python3 -m kb.train --universe data/v10L/universe.json --epochs 2 \
  --seed 0 --model gpt-5-mini --n1 15 --n2 20 --m 15 \
  --out runs/v11_main --eval-each-epoch \
  > runs/v11_main.log 2>&1

echo "=== frozen exams on the v11 store ==="
for M in 15 8; do
  python3 -m kb.test --kb runs/v11_main/kb_epoch_2.json \
    --universe data/v10L/universe.json --split both --epoch 2 --m $M \
    --model gpt-5-mini --out runs/v11_trained_m$M > runs/v11_trained_m$M.log 2>&1
done
python3 -m kb.test --kb runs/v11_main/kb_epoch_2.json \
  --universe data/v10L/universe.json --split train --epoch 2 \
  --model gpt-5-mini --out runs/v11_trained_train > runs/v11_trained_train.log 2>&1

python3 scripts/note_character.py --kb runs/v11_main/kb_epoch_2.json \
  --baseline runs/v10L_dedup/kb_epoch_2.json | tee runs/v11_structure.txt
# The retrieval skill changed (Phase 1 and the exam now share it), so every
# baseline must be re-measured under it before the E2 table means anything.
echo "=== re-measuring baselines under the shared retrieval skill ==="
for ARM in "b1:data/v10L/universe.json" \
           "graphrag:data/v10L_graphrag/universe.json" \
           "hipporag:data/v10L_hipporag/universe.json"; do
  NAME=${ARM%%:*}; KB=${ARM#*:}
  for M in 15 8; do
    python3 -m kb.test --kb "$KB" --universe data/v10L/universe.json \
      --split both --epoch 0 --m $M --model gpt-5-mini \
      --out runs/v11base_${NAME}_m$M > runs/v11base_${NAME}_m$M.log 2>&1
    echo "  ${NAME} M=$M done"
  done
done

echo "V11 DONE"
