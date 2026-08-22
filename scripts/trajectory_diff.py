"""Side-by-side reader trajectories on the same question, two stores.

The gradient table says how many actions each arm spends; this says what it
spends them on. For a question in a given group it prints the trained
store's trajectory next to the untrained one, so the mechanism -- a search
that lands on an index and reads its targets, versus a search that pages
through similar-looking source documents -- is visible rather than inferred.

    python3 scripts/trajectory_diff.py --group share2 --n 2
    python3 scripts/trajectory_diff.py --qid g0031
"""
import argparse
import json
import re
from pathlib import Path

W = 62


def traces(run):
    out = {}
    p = Path(f"runs/{run}/trace.jsonl")
    if not p.exists():
        return out
    for line in open(p):
        d = json.loads(line)
        out.setdefault(d["qid"], []).append(d)
    return out


def rows(run):
    p = Path(f"runs/{run}/test_log.jsonl")
    return ({json.loads(l)["qid"]: json.loads(l) for l in open(p)}
            if p.exists() else {})


def fmt(step):
    a, inp = step["action"], step["input"]
    if isinstance(inp, str):
        try:
            inp = json.loads(inp.replace("'", '"'))
        except Exception:
            inp = {"_": inp}
    if a == "search":
        head = f'search "{inp.get("query","")}"'
        if int(inp.get("page", 1)) > 1:
            head += f' p{inp["page"]}'
    elif a == "read":
        head = f'read {inp.get("id","")}'
    elif a == "answer":
        head = f'answer "{str(inp.get("text",""))[:34]}"'
    else:
        head = a
    res = re.sub(r"\s+", " ", str(step.get("result", "")))[:W - 4]
    return head[:W], res


def show(qid, text, gold, A, B, ra, rb):
    print("=" * 132)
    print(f"{qid}   {text}")
    print(f"gold: {gold}")
    print("-" * 132)
    print(f"{'OURS (trained store)':<64} | {'B1 (untrained flat store)':<64}")
    print(f"{'steps ' + str(ra['steps']) + ',  F1 ' + f'{ra[chr(102)+chr(49)]:.2f}':<64} | "
          f"{'steps ' + str(rb['steps']) + ',  F1 ' + f'{rb[chr(102)+chr(49)]:.2f}':<64}")
    print("-" * 132)
    for i in range(max(len(A), len(B))):
        for half in (A, B):
            pass
        la = fmt(A[i]) if i < len(A) else ("", "")
        lb = fmt(B[i]) if i < len(B) else ("", "")
        print(f"{i+1:>2} {la[0]:<61} | {i+1 if i < len(B) else '':>2} {lb[0]:<61}")
        if la[1] or lb[1]:
            print(f"   {('-> ' + la[1])[:61]:<61} |    {('-> ' + lb[1])[:61]:<61}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None, help="exact | share2 | share1 | share0")
    ap.add_argument("--qid", default=None)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--rank", choices=["median", "best"], default="median")
    ap.add_argument("--universe", default="data/grad_b1/universe.json")
    a = ap.parse_args()

    uni = json.load(open(a.universe))
    Q = {q["qid"]: q for q in uni["questions"]}
    TA, TB = traces("grad_trained"), traces("grad_b1")
    RA, RB = rows("grad_trained"), rows("grad_b1")

    if a.qid:
        picks = [a.qid]
    else:
        cand = [q for q in RA if q in RB and RA[q].get("split") == a.group]
        # the MEDIAN case, not the best one: sorting by step gap and taking
        # the extreme would show the group at its most flattering, which is
        # exactly the thing a qualitative example should not do
        cand.sort(key=lambda q: RA[q]["steps"] - RB[q]["steps"])
        mid = len(cand) // 2
        picks = cand[mid:mid + a.n] if a.n == 1 else cand[
            max(0, mid - a.n // 2):max(0, mid - a.n // 2) + a.n]

    for qid in picks:
        if qid not in TA or qid not in TB:
            print(f"{qid}: no trace"); continue
        show(qid, Q[qid]["text"], ", ".join(Q[qid]["golds"][:1]),
             TA[qid], TB[qid], RA[qid], RB[qid])


if __name__ == "__main__":
    main()
