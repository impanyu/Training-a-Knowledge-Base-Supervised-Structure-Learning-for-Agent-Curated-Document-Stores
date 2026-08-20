#!/usr/bin/env bash
# Runs when both chains finish: assemble every number the paper needs into
# runs/RESULTS.txt, so writing up is reading one file rather than re-deriving
# figures from a dozen run directories.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

while pgrep -f "v11_full.sh|pw1_full.sh" > /dev/null; do sleep 300; done

OUT=runs/RESULTS.txt
{
  echo "==================== KBGym (main arm) ===================="
  python3 scripts/results_table.py 2>&1
  echo
  echo "==================== store structure ===================="
  python3 scripts/note_character.py --kb runs/v11_main/kb_epoch_2.json \
      --baseline runs/v10L_dedup/kb_epoch_2.json 2>&1
  echo
  python3 scripts/index_precision.py --kb runs/v11_main/kb_epoch_2.json 2>&1
  echo
  echo "==================== index taxonomy ===================="
  python3 scripts/index_taxonomy.py --kb runs/v11_main/kb_epoch_2.json --examples 2 2>&1
  echo
  echo "==================== PhantomWiki (external arm) ===================="
  python3 - <<'PY' 2>&1
import json, os
def agg(run, split=None):
    p=f"runs/{run}/test_log.jsonl"
    if not os.path.exists(p): return None
    rs=[json.loads(l) for l in open(p)]
    if split: rs=[r for r in rs if r["split"]==split]
    if not rs: return None
    n=len(rs)
    return n, sum(r["f1"] for r in rs)/n, sum(r["steps"] for r in rs)/n
print(f"{'arm':<22}{'budget':<8}{'split':<10}{'n':>4}{'F1':>8}{'steps':>7}")
for label, base in (("B1 flat","pw1_b1"), ("Ours (trained)","pw1_trained")):
    for M in (15, 8):
        for sp in ("test_in","test_out"):
            a=agg(f"{base}_m{M}", sp)
            if a: print(f"{label:<22}{'M='+str(M):<8}{sp:<10}{a[0]:>4}{a[1]:>8.3f}{a[2]:>7.1f}")
for label, run in (("B1 flat","pw1_b1_train"), ("Ours","pw1_trained_train")):
    a=agg(run)
    if a: print(f"{label:<22}{'M=15':<8}{'train':<10}{a[0]:>4}{a[1]:>8.3f}{a[2]:>7.1f}")
PY
  echo
  echo "==================== pw1 structure ===================="
  python3 scripts/note_character.py --kb runs/pw1_main/kb_epoch_2.json 2>&1
  echo
  python3 scripts/index_taxonomy.py --kb runs/pw1_main/kb_epoch_2.json \
      --universe data/pw1/universe.json --examples 2 2>&1
  echo
  echo "==================== training cost ===================="
  python3 - <<'PY' 2>&1
import json
for nm, p in (("KBGym","runs/v11_main/train_log.jsonl"),
              ("PhantomWiki","runs/pw1_main/train_log.jsonl")):
    t=[json.loads(l) for l in open(p)]
    tin=sum(r["tokens_in"] for r in t); tout=sum(r["tokens_out"] for r in t)
    print(f"{nm:<14} {len(t)} iterations, {(tin+tout)/1e6:.1f}M tokens, "
          f"{sum(r['seconds'] for r in t)/3600:.1f} h")
    for ep in sorted({r["epoch"] for r in t}):
        rs=[r for r in t if r["epoch"]==ep]
        print(f"   epoch {ep}: forward F1 {sum(r['f1'] for r in rs)/len(rs):.3f} "
              f"({len(rs)} questions), {sum(r['p1_steps'] for r in rs)/len(rs):.1f} steps")
PY
} > "$OUT" 2>&1
echo "RESULTS COLLECTED -> $OUT"
