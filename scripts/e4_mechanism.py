"""E4 (iv) structure-semantics alignment and (v) mechanism attribution.

Both are computed from artifacts already on disk -- the trained store, the
universe's (invisible-to-the-agent) support sets, and the exam traces -- so
neither costs an API call.

(iv) alignment: a built link (u, v) is "aligned" if u and v co-occur in the
     support set of some question in the universe. Supports are never shown to
     the training agent, so a high share means the agent recovered genuine
     reasoning adjacency rather than surface similarity. Chance level is
     estimated by sampling random statement pairs.

(v)  attribution: per exam question, compare the trained reader's steps to
     B1's on the SAME qid, and split questions by whether the trained
     trajectory touched agent-built structure (an authored navigation document,
     or a statement that gained links during training).

    python3 scripts/e4_mechanism.py [--run v10L_trained_m15] [--base v10L_b1_m15]
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

SID = re.compile(r"\b([su]\d{3,})\b")


def load_store(path):
    d = json.load(open(path))
    return d["store"]["nodes"] if isinstance(d.get("store"), dict) else d["nodes"]


def alignment(nodes, questions, seed=0):
    """(iv) share of built links whose endpoints co-occur in some support set."""
    pairs = set()
    for q in questions:
        sup = q.get("support") or []
        for i, a in enumerate(sup):
            for b in sup[i + 1:]:
                pairs.add((min(a, b), max(a, b)))

    links, aligned, cross = 0, 0, 0
    authored = {n["id"] for n in nodes if n.get("flag") == "authored"}
    for n in nodes:
        for t in n.get("links", []):
            links += 1
            u, v = n["id"], t
            if u in authored or v in authored:
                cross += 1          # nav-doc links have no support analogue
                continue
            if (min(u, v), max(u, v)) in pairs:
                aligned += 1
    stmt_links = links - cross

    # chance level: random statement pairs
    rng = random.Random(seed)
    sids = [n["id"] for n in nodes if n.get("flag") != "authored"]
    hits = sum(1 for _ in range(20000)
               if (lambda a, b: (min(a, b), max(a, b)) in pairs)(
                   rng.choice(sids), rng.choice(sids)))
    return dict(links=links, nav_links=cross, statement_links=stmt_links,
                aligned=aligned,
                rate=aligned / stmt_links if stmt_links else 0.0,
                chance=hits / 20000, support_pairs=len(pairs))


def read_trace(path):
    """qid -> set of node ids the reader saw (search results and reads)."""
    seen = defaultdict(set)
    if not Path(path).exists():
        return seen
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") != "test":
            continue
        qid = r["qid"]
        blob = f"{r.get('input')} {r.get('result')}"
        seen[qid].update(SID.findall(str(blob)))
    return seen


def attribution(run, base, nodes, orig_ids):
    authored = {n["id"] for n in nodes if n.get("flag") == "authored"}
    linked = {n["id"] for n in nodes
              if n.get("flag") != "authored" and n.get("links")}

    tr = {json.loads(l)["qid"]: json.loads(l)
          for l in open(f"runs/{run}/test_log.jsonl")}
    b1 = {json.loads(l)["qid"]: json.loads(l)
          for l in open(f"runs/{base}/test_log.jsonl")}
    seen = read_trace(f"runs/{run}/trace.jsonl")

    groups = {"touched": [], "untouched": []}
    nav_hits = 0
    for qid, r in tr.items():
        if qid not in b1:
            continue
        s = seen.get(qid, set())
        hit_nav = bool(s & authored)
        hit_link = bool(s & linked)
        nav_hits += hit_nav
        key = "touched" if (hit_nav or hit_link) else "untouched"
        groups[key].append((r["steps"] - b1[qid]["steps"],
                            r["f1"] - b1[qid]["f1"], hit_nav, hit_link))

    out = {}
    for k, rows in groups.items():
        if not rows:
            out[k] = None
            continue
        n = len(rows)
        out[k] = dict(n=n,
                      d_steps=sum(r[0] for r in rows) / n,
                      d_f1=sum(r[1] for r in rows) / n,
                      nav=sum(r[2] for r in rows),
                      links=sum(r[3] for r in rows))
    out["nav_hit_questions"] = nav_hits
    out["authored_alive"] = len(authored)
    out["statements_with_links"] = len(linked)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="runs/v10L_dedup/kb_epoch_2.json")
    ap.add_argument("--universe", default="data/v10L/universe.json")
    ap.add_argument("--run", default="v10L_trained_m15")
    ap.add_argument("--base", default="v10L_b1_m15")
    a = ap.parse_args()

    uni = json.load(open(a.universe))
    nodes = load_store(a.kb)
    orig = {n["id"] for n in uni["nodes"]}

    al = alignment(nodes, uni["questions"])
    print("E4 (iv) structure-semantics alignment")
    print(f"  built links               {al['links']:>6}")
    print(f"    involving a nav doc     {al['nav_links']:>6}")
    print(f"    statement-to-statement  {al['statement_links']:>6}")
    print(f"  aligned with a support set{al['aligned']:>6}"
          f"   ({al['rate']:.1%})")
    print(f"  chance level                     {al['chance']:.2%}"
          f"   (random statement pairs)")
    if al["chance"]:
        print(f"  lift over chance          {al['rate']/al['chance']:>6.0f}x")

    at = attribution(a.run, a.base, nodes, orig)
    print("\nE4 (v) mechanism attribution "
          f"({a.run} vs {a.base}, per-question paired)")
    print(f"  authored nav docs alive   {at['authored_alive']:>6}")
    print(f"  statements with links     {at['statements_with_links']:>6}")
    for k in ("touched", "untouched"):
        g = at[k]
        if not g:
            continue
        print(f"  {k:<10} n={g['n']:>3}  dsteps {g['d_steps']:+.2f}"
              f"  dF1 {g['d_f1']:+.3f}"
              f"  (nav {g['nav']}, links {g['links']})")


if __name__ == "__main__":
    main()
