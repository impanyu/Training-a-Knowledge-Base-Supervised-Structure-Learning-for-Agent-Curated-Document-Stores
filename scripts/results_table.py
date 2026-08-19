"""Assemble the E1/E2/E3 result tables from finished run directories.

Reads test_log.jsonl from each arm's run dir and prints:
  E2  method x split x budget:  F1, steps, tokens/question, unanswered
  E3  three-layer scores and the per-category profile for the trained store
Numbers are printed to the precision the paper reports; nothing is inferred.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ARMS = [                       # label, m15 dir, m8 dir
    ("B1 flat",        "v10L_b1_m15",       "v10L_b1_m8"),
    ("B3 GraphRAG",    "v10L_graphrag_m15", "v10L_graphrag_m8"),
    ("B5 HippoRAG2",   "v10L_hipporag_m15", "v10L_hipporag_m8"),
    ("Ours (trained)", "v10L_trained_m15",  "v10L_trained_m8"),
]
RUNS = Path("runs")


def load(run):
    p = RUNS / run / "test_log.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p)]


def agg(rows, split):
    rs = [r for r in rows if r["split"] == split]
    if not rs:
        return None
    n = len(rs)
    return dict(n=n,
                f1=sum(r["f1"] for r in rs) / n,
                steps=sum(r["steps"] for r in rs) / n,
                tok=sum(r["tokens_in"] + r["tokens_out"] for r in rs) / n,
                unans=sum(1 for r in rs if r["status"] == "unanswered"))


def e2():
    print("E2  baseline table (150 questions: 100 test_in + 50 test_out)\n")
    hdr = f"{'method':<16}{'budget':<8}{'split':<10}{'n':>4}{'F1':>7}{'steps':>7}{'tok/q':>8}{'unans':>7}"
    print(hdr)
    print("-" * len(hdr))
    for label, d15, d8 in ARMS:
        for budget, d in (("M=15", d15), ("M=8", d8)):
            rows = load(d)
            if rows is None:
                print(f"{label:<16}{budget:<8}{'(pending)':<10}")
                continue
            for split in ("test_in", "test_out"):
                a = agg(rows, split)
                if a:
                    print(f"{label:<16}{budget:<8}{split:<10}{a['n']:>4}"
                          f"{a['f1']:>7.3f}{a['steps']:>7.1f}{a['tok']:>8.0f}{a['unans']:>7}")
        print()


def e3(train_run="v10L_dedup"):
    print("\nE3  three-layer generalization\n")
    tl = [json.loads(l) for l in open(RUNS / train_run / "train_log.jsonl")]
    for ep in sorted({r["epoch"] for r in tl}):
        rs = [r for r in tl if r["epoch"] == ep]
        print(f"  train-forward epoch {ep}: F1 {sum(r['f1'] for r in rs)/len(rs):.3f} "
              f"(n={len(rs)}; valid as retention from epoch 2)")
    rows = load("v10L_trained_m15")
    if rows:
        for split in ("test_in", "test_out"):
            a = agg(rows, split)
            print(f"  {split:<9}: F1 {a['f1']:.3f} (n={a['n']})")
        print("\n  per-category profile (trained vs B1, M=15)\n")
        base = load("v10L_b1_m15")
        bycat = defaultdict(lambda: {"t": [], "b": []})
        for r in rows:
            bycat[r["category"]]["t"].append(r["f1"])
        for r in base or []:
            bycat[r["category"]]["b"].append(r["f1"])
        print(f"  {'cat':<7}{'n':>4}{'B1':>8}{'trained':>9}{'delta':>8}")
        for cat in sorted(bycat):
            t, b = bycat[cat]["t"], bycat[cat]["b"]
            if not t or not b:
                continue
            tm, bm = sum(t) / len(t), sum(b) / len(b)
            print(f"  {cat:<7}{len(t):>4}{bm:>8.2f}{tm:>9.2f}{tm-bm:>+8.2f}")


def costs(train_run="v10L_dedup"):
    print("\nCosts\n")
    tl = [json.loads(l) for l in open(RUNS / train_run / "train_log.jsonl")]
    tin = sum(r["tokens_in"] for r in tl)
    tout = sum(r["tokens_out"] for r in tl)
    print(f"  training: {len(tl)} iterations, {tin+tout:,} tokens "
          f"({tin:,} in / {tout:,} out), {sum(r['seconds'] for r in tl)/3600:.1f} h")
    tr, b1 = load("v10L_trained_m15"), load("v10L_b1_m15")
    if tr and b1:
        ct = sum(r["tokens_in"] + r["tokens_out"] for r in tr) / len(tr)
        cb = sum(r["tokens_in"] + r["tokens_out"] for r in b1) / len(b1)
        print(f"  answering: B1 {cb:.0f} tok/q, trained {ct:.0f} tok/q, "
              f"saving {cb-ct:+.0f}")
        if cb > ct:
            print(f"  break-even N* = {(tin+tout)/(cb-ct):.0f} questions")
        else:
            print("  break-even: undefined (trained store is not cheaper per question)")


if __name__ == "__main__":
    e2()
    e3()
    costs()
