"""Profile what training added to the store, by the only two kinds of
structure the action set can produce:

  (i)  INDEX notes  - agent-authored notes pointing at the notes that belong
       to their key. Their value is the completeness of their links.
  (ii) EDGES between existing notes - fact-to-fact (a hop a chain must make)
       and index-to-index (navigation levels).

Reported per class, because "how many notes were added" says nothing about
whether the store became navigable. An index carrying two links is not an
index. The v10L run is the reference point: 359 authored notes averaging
1.4 outgoing links, and 127 fact-to-fact edges over 5,864 statements.

    python3 scripts/note_character.py --kb runs/X/kb_epoch_N.json \
        [--baseline runs/v10L_dedup/kb_epoch_2.json]
"""
import argparse
import json
import statistics as S
from collections import Counter


def load(path):
    d = json.load(open(path))
    return d["store"]["nodes"] if "store" in d else d["nodes"]


def profile(path):
    nodes = load(path)
    by_id = {n["id"]: n for n in nodes}
    idx = {n["id"] for n in nodes if n.get("flag") == "authored"}

    out_deg, edges = {}, Counter()
    for n in nodes:
        src_is_idx = n["id"] in idx
        links = n.get("links", [])
        if src_is_idx:
            out_deg[n["id"]] = len(links)
        for t in links:
            if t not in by_id:
                continue
            kind = ("index->index" if src_is_idx and t in idx else
                    "index->fact" if src_is_idx else
                    "fact->index" if t in idx else "fact->fact")
            edges[kind] += 1

    degs = sorted(out_deg.values())
    # how much of the store is reachable from an index in one read
    covered = {t for n in nodes if n["id"] in idx
               for t in n.get("links", []) if t in by_id and t not in idx}
    facts = len(nodes) - len(idx)
    return dict(nodes=len(nodes), indexes=len(idx), facts=facts,
                degs=degs, edges=edges,
                covered=len(covered),
                coverage=len(covered) / facts if facts else 0.0)


def show(label, p):
    print(f"{label}")
    print(f"  store            {p['nodes']:>6} notes = {p['facts']} facts "
          f"+ {p['indexes']} indexes")
    if p["degs"]:
        d = p["degs"]
        print(f"  index out-degree  mean {S.mean(d):>5.2f}  median {S.median(d):>3.0f}"
              f"  max {max(d):>3}   empty(0 links) {sum(1 for x in d if x == 0)}"
              f"  thin(<5) {sum(1 for x in d if x < 5)}")
    for k in ("index->fact", "index->index", "fact->fact", "fact->index"):
        if p["edges"][k]:
            print(f"  {k:<14} {p['edges'][k]:>6}")
    print(f"  facts reachable from some index in ONE read: "
          f"{p['covered']} / {p['facts']} = {p['coverage']:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--baseline", default=None)
    a = ap.parse_args()
    if a.baseline:
        show("baseline " + a.baseline, profile(a.baseline))
        print()
    show("run      " + a.kb, profile(a.kb))
