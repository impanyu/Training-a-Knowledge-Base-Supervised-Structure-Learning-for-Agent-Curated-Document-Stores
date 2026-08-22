"""Index construction quality, by structural family.

Scores every authored index against the universe semantically (see
scripts/index_precision2.py for why string matching is wrong for relational
keys), and groups the result by what kind of key the index is built on.
Precision = links pointing at a document about a genuine member of the key.
Recall = members of the key reachable from the index.

    python3 scripts/index_quality_table.py --kb runs/v11_main/kb_epoch_2.json
    python3 scripts/index_quality_table.py --kb ... --latex
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from index_precision2 import NAME, members, universe_relations

FAMILY = [
    ("attribute: city",   lambda t: t.lower().startswith("residents of")),
    ("attribute: hobby / job",
     lambda t: t.lower().startswith(("people whose hobby", "people whose job"))),
    ("relation: 1-hop",
     lambda t: t.lower().startswith(("children of", "friends of", "parents of",
                                     "spouse of", "spouses of"))),
    ("relation: 2-hop",
     lambda t: t.lower().startswith(("grandchildren of", "siblings of"))),
    ("single-entity hub", lambda t: len(NAME.findall(t)) == 1 and " of " not in t.lower()),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--universe", default="data/v10L/universe.json")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()

    uni = json.load(open(a.universe))
    kids, parents, friends, spouse, attr = universe_relations(uni["nodes"])
    d = json.load(open(a.kb))
    nodes = d["store"]["nodes"] if "store" in d else d["nodes"]
    by_id = {n["id"]: n for n in nodes}
    about = {n["id"]: set(NAME.findall(n["text"])) for n in uni["nodes"]}
    idx = [n for n in nodes if n.get("flag") == "authored"]

    buckets = {name: [] for name, _ in FAMILY}
    buckets["_dropped"] = []      # unclassifiable keys, not reported
    for n in idx:
        text = n["text"]
        want = members(text, kids, parents, friends, spouse, attr)
        links = [t for t in n.get("links", []) if t in by_id]
        fam = next((name for name, pred in FAMILY if pred(text)), None)
        if fam is None or want is None or not links:
            buckets["_dropped"].append((n, None, links))
            continue
        buckets[fam].append((n, want, links))

    rows = []
    for name in [f for f, _ in FAMILY]:
        items = buckets[name]
        if not items:
            continue
        n_idx = len(items)
        scored = [(n, w, l) for n, w, l in items if w is not None]
        if not scored:
            rows.append((name, n_idx, 0.0, None, None, sum(
                1 for n, _, l in items if not l)))
            continue
        L = sum(len(l) for _, _, l in scored)
        ok = sum(1 for _, w, l in scored for t in l if about.get(t, set()) & w)
        found = sum(len({p for t in l for p in about.get(t, set()) & w})
                    for _, w, l in scored)
        total = sum(len(w) for _, w, _ in scored)
        empty = sum(1 for n, _, _ in items if not n.get("links"))
        rows.append((name, n_idx, L / n_idx, ok / L, found / max(1, total), empty))

    if a.latex:
        for name, n_idx, deg, p, r, empty in rows:
            pp = f"{p:.0%}".replace("%", r"\%") if p is not None else "---"
            rr = f"{r:.0%}".replace("%", r"\%") if r is not None else "---"
            print(f"{name} & {n_idx} & {deg:.1f} & {pp} & {rr} \\\\")
    else:
        print(f"{'family':<26}{'n':>4}{'mean deg':>10}{'precision':>11}"
              f"{'recall':>9}")
        for name, n_idx, deg, p, r, empty in rows:
            pp = f"{p:.0%}" if p is not None else "--"
            rr = f"{r:.0%}" if r is not None else "--"
            print(f"{name:<26}{n_idx:>4}{deg:>10.1f}{pp:>11}{rr:>9}")
        tot = sum(r[1] for r in rows)
        print(f"{'':<26}{tot:>4}  scored, of {len(idx)} authored index documents")


if __name__ == "__main__":
    main()
