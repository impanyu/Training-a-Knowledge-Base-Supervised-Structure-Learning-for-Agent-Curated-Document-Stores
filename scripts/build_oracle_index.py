"""An oracle access layer: the structure our training is trying to discover,
built directly from the universe so we can measure what it is worth.

This is not a baseline we compete with - it is a CEILING. If a perfectly
built index layer does not help the fixed reader, no training procedure can,
and the hypothesis is wrong for a reason that has nothing to do with the
agent. If it helps a lot, the gap between it and the trained store is the
part of the problem that is about learning rather than about structure.

Shape (the one the design converged on):
  person node   - one per person, linked to every note about them
  attribute idx - one per city / job / hobby, linked to the PERSON nodes
  relation idx  - "friends of X", "children of X", "grandchildren of X",
                  linked to the person nodes of the members
Every index is complete by construction, and its members live in its links,
never in its text.

    python3 scripts/build_oracle_index.py --out data/v10L_oracle
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

PATTERNS = [
    (re.compile(r"^(.+?)'s job is (.+)\.$"), "job"),
    (re.compile(r"^(.+?)'s hobby is (.+)\.$"), "hobby"),
    (re.compile(r"^(.+?) lives in the city of (.+)\.$"), "city"),
    (re.compile(r"^(.+?) is married to (.+)\.$"), "spouse"),
    (re.compile(r"^(.+?) is a friend of (.+)\.$"), "friend"),
    (re.compile(r"^(.+?) is the (?:father|mother) of (.+)\.$"), "parent"),
    (re.compile(r"^(.+?) is a child of (.+)\.$"), "child"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="data/v10L/universe.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    uni = json.load(open(a.universe))
    nodes = [dict(n) for n in uni["nodes"]]
    for n in nodes:
        n.setdefault("links", [])
        n.setdefault("flag", None)
    by_person = defaultdict(list)          # person -> their note ids
    attr = defaultdict(lambda: defaultdict(set))   # kind -> value -> persons
    rel = defaultdict(lambda: defaultdict(set))    # kind -> person -> persons

    for n in nodes:
        for pat, kind in PATTERNS:
            m = pat.match(n["text"])
            if not m:
                continue
            subj, obj = m.group(1), m.group(2)
            by_person[subj].append(n["id"])
            if kind in ("job", "hobby", "city"):
                attr[kind][obj].add(subj)
            else:
                by_person[obj].append(n["id"])
                if kind == "friend":
                    rel["friends"][subj].add(obj)
                    rel["friends"][obj].add(subj)
                elif kind == "spouse":
                    rel["spouse"][subj].add(obj)
                    rel["spouse"][obj].add(subj)
                elif kind == "parent":
                    rel["children"][subj].add(obj)
                    rel["parents"][obj].add(subj)
                elif kind == "child":
                    rel["parents"][subj].add(obj)
                    rel["children"][obj].add(subj)
            break

    nid = max(int(n["id"][1:]) for n in nodes)

    def new(text, targets):
        nonlocal nid
        nid += 1
        node = {"id": f"s{nid}", "text": text, "origin": None,
                "flag": "authored", "links": sorted(set(targets)), "absorbed": []}
        nodes.append(node)
        return node["id"]

    # layer 1: one node per person, linked to every note about them
    person_node = {p: new(f"{p}", sorted(set(ids)))
                   for p, ids in sorted(by_person.items())}

    # layer 2: attribute indexes, linked to person nodes
    n_attr = 0
    for kind, values in attr.items():
        label = {"job": "People whose job is", "hobby": "People whose hobby is",
                 "city": "People who live in the city of"}[kind]
        for value, people in sorted(values.items()):
            new(f"{label} {value}", [person_node[p] for p in people if p in person_node])
            n_attr += 1

    # layer 3: relation indexes, linked to person nodes
    n_rel = 0
    for p, friends in sorted(rel["friends"].items()):
        if p in person_node:
            new(f"Friends of {p}", [person_node[q] for q in friends if q in person_node])
            n_rel += 1
    for p, kids in sorted(rel["children"].items()):
        if p in person_node:
            new(f"Children of {p}", [person_node[q] for q in kids if q in person_node])
            n_rel += 1
            grand = {g for k in kids for g in rel["children"].get(k, ())}
            if grand:
                new(f"Grandchildren of {p}",
                    [person_node[q] for q in grand if q in person_node])
                n_rel += 1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    uni_out = dict(uni)
    uni_out["nodes"] = nodes
    json.dump(uni_out, open(out / "universe.json", "w"))
    authored = [n for n in nodes if n.get("flag") == "authored"]
    degs = [len(n["links"]) for n in authored]
    meta = {"person_nodes": len(person_node), "attribute_indexes": n_attr,
            "relation_indexes": n_rel, "index_nodes": len(authored),
            "links": sum(degs), "mean_out_degree": round(sum(degs) / len(degs), 2),
            "empty_indexes": sum(1 for d in degs if d == 0)}
    json.dump(meta, open(out / "build_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
