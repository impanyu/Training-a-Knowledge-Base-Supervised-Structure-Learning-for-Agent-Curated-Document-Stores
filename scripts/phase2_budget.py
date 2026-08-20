"""Where does the backward pass spend its 15 actions?

The v11 protocol asks the agent to diagnose the trajectory first (searching
for the gold answer, locating where it lives, comparing against what its own
searches returned) and only then to build or extend an index. Both compete
for the same budget, and v10L failed precisely by under-building links (mean
index out-degree 1.27). If diagnosis eats the budget, the index protocol
never fires and the run reproduces the old failure for a new reason.

Splits phase-2 actions into diagnosis (search/read) and construction
(add/edit/delete/link/unlink), overall and by outcome, since the diagnosis
branch only does real work when the forward pass failed.

    python3 scripts/phase2_budget.py --run runs/v11_pilot [--baseline runs/v10L_dedup]
"""
import argparse
import json
from collections import Counter, defaultdict

DIAG = {"search", "read"}
BUILD = {"add", "edit", "delete", "link", "unlink"}


def load(run):
    f1 = {}
    for line in open(f"{run}/train_log.jsonl"):
        r = json.loads(line)
        f1[(r["epoch"], r["qid"])] = r["f1"]

    steps = defaultdict(Counter)      # (epoch,qid) -> action counter
    for line in open(f"{run}/trace.jsonl"):
        r = json.loads(line)
        if r.get("kind") == "train" and int(r.get("phase", 0)) == 2:
            steps[(int(r["epoch"]), r["qid"])][r["action"]] += 1
    return f1, steps


def band(v):
    return "failed (F1<0.5)" if v < 0.5 else "solved (F1>=0.5)"


def report(label, run):
    f1, steps = load(run)
    groups = defaultdict(lambda: {"n": 0, "acts": Counter()})
    for key, c in steps.items():
        g = groups[band(f1.get(key, 0.0))]
        g["n"] += 1
        g["acts"].update(c)
    total = {"n": sum(g["n"] for g in groups.values()), "acts": Counter()}
    for g in groups.values():
        total["acts"].update(g["acts"])

    print(f"{label}  ({total['n']} backward passes)")
    hdr = f"  {'outcome':<18}{'iters':>6}{'steps/iter':>11}{'diagnose':>10}{'build':>8}{'links/iter':>12}"
    print(hdr)
    for name in ("failed (F1<0.5)", "solved (F1>=0.5)", "ALL"):
        g = total if name == "ALL" else groups.get(name)
        if not g or not g["n"]:
            continue
        a = g["acts"]
        tot = sum(a.values())
        d = sum(a[x] for x in DIAG)
        b = sum(a[x] for x in BUILD)
        print(f"  {name:<18}{g['n']:>6}{tot/g['n']:>11.1f}"
              f"{d/max(tot,1):>9.0%}{b/max(tot,1):>8.0%}"
              f"{a['link']/g['n']:>12.2f}")
    print(f"  action mix: "
          + ", ".join(f"{k} {v}" for k, v in total["acts"].most_common()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", default=None)
    a = ap.parse_args()
    if a.baseline:
        report("baseline " + a.baseline, a.baseline)
        print()
    report("run      " + a.run, a.run)
