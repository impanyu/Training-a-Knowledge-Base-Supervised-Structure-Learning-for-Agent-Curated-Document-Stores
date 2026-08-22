"""The key-overlap gradient: steps as the endpoint, rho against B1.

Groups, in decreasing order of contact with the training set:
  exact    -- the question itself was trained on (train-split runs)
  share2   -- both keys appeared in some training question
  share1   -- exactly one key did
  share0   -- neither did
"""
import json, os, random

B = 20000
ARMS = [("b1", "B1 flat"), ("graphrag", "B2 GraphRAG-style"),
        ("hipporag", "B3 HippoRAG2-style"),
        ("trained", "Ours (trained)")]


def load(run, split=None):
    p = f"runs/{run}/test_log.jsonl"
    if not os.path.exists(p):
        return {}
    rows = (json.loads(l) for l in open(p))
    return {r["qid"]: r for r in rows if split is None or r.get("split") == split}


def ci(deltas, base):
    n = len(deltas)
    s = sorted(sum(random.choice(deltas) for _ in range(n)) / n for _ in range(B))
    return (base + s[int(.025 * B)]) / base, (base + s[int(.975 * B)]) / base


def show(title, runs, filt):
    random.seed(0)
    d = {k: {q: r for q, r in load(runs[k]).items() if filt(r)} for k, _ in ARMS}
    if not d["b1"]:
        print(f"\n{title}: pending"); return
    qids = [q for q in d["b1"] if all(q in d[k] for k, _ in ARMS if d[k])]
    if not qids:
        print(f"\n{title}: pending"); return
    n = len(qids)
    bs = sum(d["b1"][q]["steps"] for q in qids) / n
    bf = sum(d["b1"][q]["f1"] for q in qids) / n
    print(f"\n{title}   n={n}")
    print(f"  {'arm':<19}{'steps':>7}{'rho':>7}{'   95% CI':>16}{'F1':>8}")
    for k, label in ARMS:
        if not d[k]:
            print(f"  {label:<19}{'--':>7}  pending"); continue
        x = d[k]
        st = sum(x[q]["steps"] for q in qids) / n
        f1 = sum(x[q]["f1"] for q in qids) / n
        lo, hi = ((1.0, 1.0) if k == "b1" else
                  ci([x[q]["steps"] - d["b1"][q]["steps"] for q in qids], bs))
        mark = "*" if k != "b1" and hi < 1.0 else " "
        print(f"  {label:<19}{st:>7.1f}{st/bs:>7.3f}{mark}  [{lo:.2f}, {hi:.2f}]{f1:>8.3f}")


def main():
    LABEL = {"exact": "EXACT   (the question itself was trained on)",
             2: "SHARE 2 (both keys seen in training)",
             1: "SHARE 1 (one key seen in training)",
             0: "SHARE 0 (neither key seen in training)"}
    for k in ("exact", 2, 1, 0):
        split = "exact" if k == "exact" else f"share{k}"
        show(LABEL[k], {a: f"grad_{a}" for a, _ in ARMS},
             lambda r, s=split: r.get("split") == s)


if __name__ == "__main__":
    main()
