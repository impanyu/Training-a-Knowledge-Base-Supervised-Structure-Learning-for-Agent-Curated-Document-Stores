"""Live quality read on a running training job, from its trace alone.

Progress counters say nothing about whether the store is getting better, and
every failure in this project looked fine in the counters: indexes that were
built and left empty, sixty targets attached to a ten-member key, a budget
spent rephrasing instead of paging. This reconstructs the store from the
trace - no API calls, no waiting for a snapshot - and reports the handful of
numbers that would have caught each of them.

    python3 scripts/watch_quality.py runs/v11_main [runs/pw1_main ...]
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

NAME = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
# what healthy looks like: B3 HippoRAG2 reaches 16.9 out-degree with none
# empty, the exhaustive construction 9.95.
GOOD_DEGREE, MAX_EMPTY, MIN_PRECISION = 6.0, 0.15, 0.80


def rebuild(run: Path):
    """Index documents and their links as the agent has them right now."""
    idx, links, rejected = {}, defaultdict(set), 0
    pages, searches, passes = [], 0, set()
    for line in open(run / "trace.jsonl"):
        r = json.loads(line)
        if r.get("phase") != 2:
            continue
        passes.add((r["epoch"], r["qid"]))
        a, res = r["action"], str(r["result"])
        if a == "search":
            searches += 1
            pages.append(r["input"].get("page", 1))
        if res.startswith("ERROR"):
            rejected += 1
            continue
        if a == "add" and res.startswith("added"):
            idx[res.split()[1]] = r["input"]["text"]
        elif a == "delete":
            idx.pop(str(r["input"].get("id")), None)
        elif a == "link":
            links[r["input"]["a"]].add(r["input"]["b"])
        elif a == "link_many":
            links[r["input"]["a"]].update(r["input"]["targets"])
    return idx, links, pages, searches, len(passes), rejected


def precision(idx, links, universe):
    """Share of linked documents that actually mention their index's key."""
    text = {n["id"]: n["text"] for n in universe["nodes"]}
    v = universe["vocab"]
    vocab = list(v.get("jobs", [])) + list(v.get("hobbies", [])) + list(v.get("cities", []))
    hit = tot = 0
    for i, label in idx.items():
        names = NAME.findall(label)
        key = names[0] if len(names) == 1 else next(
            (x for x in vocab if re.search(rf"\b{re.escape(x)}\b", label, re.I)), None)
        targets = [t for t in links.get(i, ()) if t in text]
        if not key or not targets:
            continue
        tot += len(targets)
        hit += sum(1 for t in targets if re.search(re.escape(key), text[t], re.I))
    return (hit / tot, tot) if tot else (None, 0)


def report(run: Path):
    log = run / "train_log.jsonl"
    done = sum(1 for _ in open(log)) if log.exists() else 0
    idx, links, pages, searches, passes, rejected = rebuild(run)
    print(f"\n{run.name}  {done} iterations, {passes} backward passes")
    if not idx:
        print("  nothing built yet")
        return
    deg = [len(links.get(i, ())) for i in idx]
    mean = sum(deg) / len(deg)
    empty = sum(1 for d in deg if d == 0) / len(deg)
    uni = json.loads((Path("data/v10L/universe.json") if "v11" in run.name
                      else Path("data/pw1/universe.json")).read_text())
    prec, scored = precision(idx, links, uni)
    p1 = Counter(pages)[1] / len(pages) if pages else 0

    def verdict(ok):
        return "ok " if ok else "LOW"
    print(f"  {verdict(mean >= GOOD_DEGREE)} out-degree   {mean:>6.2f}   (want >= {GOOD_DEGREE})")
    print(f"  {verdict(empty <= MAX_EMPTY)} empty        {empty:>6.0%}   (want <= {MAX_EMPTY:.0%})")
    if prec is not None:
        print(f"  {verdict(prec >= MIN_PRECISION)} precision    {prec:>6.0%}   "
              f"(want >= {MIN_PRECISION:.0%}, {scored} links scored)")
    print(f"      indexes      {len(idx):>6}      edges {sum(deg)}")
    print(f"      page-1 share {p1:>6.0%}      {searches / max(passes, 1):.1f} searches/pass")
    if rejected:
        print(f"      {rejected} rejected edits (oversized link batches)")


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["runs/v11_main", "runs/pw1_main"]:
        report(Path(arg))
