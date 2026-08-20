"""What kinds of index did training actually build?

Classifies every agent-authored index note by the KEY it is built on and by
what it points at, and prints real examples of each kind. The taxonomy is
read off the notes themselves rather than imposed: an index is grouped by
whether its key is one entity, one attribute value, a relation, or a
combination, and separately by whether its links reach facts, other indexes,
or both. Sizes come with it, because an index carrying one link is a
different object from one carrying twenty.

    python3 scripts/index_taxonomy.py --kb runs/v11_main/kb_epoch_2.json \
        [--universe data/v10L/universe.json] [--examples 2]
"""
import argparse
import json
import re
from collections import defaultdict

NAME = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
RELATION = re.compile(r"\b(child|parent|father|mother|son|daughter|spouse|"
                      r"marriage|married|marries|friend|grandchild|sibling|"
                      r"family|families|ancestor|descendant)\w*\b", re.I)


def load(path):
    d = json.load(open(path))
    return d["store"]["nodes"] if "store" in d else d["nodes"]


def classify(note, vocab):
    """Key kind, from the note's own text."""
    t = note["text"]
    names = set(NAME.findall(t))
    attrs = {k for k in vocab if k.lower() in t.lower()}
    has_rel = bool(RELATION.search(t))
    if has_rel and len(names) == 1:
        return "relation of one entity"     # "Grandchildren of Petra Dunmore"
    if has_rel:
        return "relation, no entity named"  # "Marriages index"
    if len(attrs) >= 2:
        return "composite of attributes"    # "Coopers in Fenmarch"
    if attrs and not names:
        return "attribute value"            # "Residents of Fenmarch"
    if len(names) == 1:
        return "single entity"              # "Petra Dunmore directory"
    if names:
        return "several entities named"
    return "unclassified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--universe", default="data/v10L/universe.json")
    ap.add_argument("--examples", type=int, default=2)
    a = ap.parse_args()

    nodes = load(a.kb)
    by_id = {n["id"]: n for n in nodes}
    idx = {n["id"] for n in nodes if n.get("flag") == "authored"}
    v = json.load(open(a.universe))["vocab"]
    vocab = set(v.get("jobs", [])) | set(v.get("hobbies", [])) | set(v.get("cities", []))

    groups = defaultdict(list)
    for n in nodes:
        if n["id"] in idx:
            groups[classify(n, vocab)].append(n)

    total = sum(len(g) for g in groups.values())
    print(f"{total} authored index notes\n")
    for kind, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        degs = [len(x.get("links", [])) for x in g]
        to_idx = sum(1 for x in g for t in x.get("links", []) if t in idx)
        to_fact = sum(len(x.get("links", [])) for x in g) - to_idx
        print(f"{kind:<24} {len(g):>4} notes  ({len(g)/total:>4.0%})  "
              f"links: mean {sum(degs)/len(degs):>5.1f} max {max(degs):>3}  "
              f"empty {sum(1 for d in degs if d == 0):>3}  "
              f"-> {to_fact} facts / {to_idx} indexes")
        for x in sorted(g, key=lambda y: -len(y.get("links", [])))[:a.examples]:
            links = x.get("links", [])
            sample = ", ".join(by_id[t]["text"][:38] for t in links[:2] if t in by_id)
            print(f"     \"{x['text'][:78]}\"  [{len(links)} links]")
            if sample:
                print(f"        -> {sample}")
        print()


if __name__ == "__main__":
    main()
