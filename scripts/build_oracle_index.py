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
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kb.baseline_common import extract_all, entity_mentions  # noqa: E402

# Which relations make two people neighbours, and which make a person an
# instance of an attribute. Written to cover both arms: KBGym renders
# "X's job is cooper" while PhantomWiki renders "The occupation of X is
# cooper", and the oracle should not care which universe it is given.
PERSON_REL = {"spouse", "married to", "friend", "friend of", "friends",
              "parent of", "child of", "mother", "father", "husband", "wife",
              "son", "sons", "daughter", "daughters", "brother", "brothers",
              "sister", "sisters"}
ATTR_LABEL = {"job": "People whose job is", "occupation": "People whose job is",
              "hobby": "People whose hobby is",
              "lives in": "People who live in the city of",
              "city": "People who live in the city of"}
CHILD_REL = {"parent of", "son", "sons", "daughter", "daughters"}
PARENT_REL = {"child of", "mother", "father"}
FRIEND_REL = {"friend", "friend of", "friends"}


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
    rel_map = defaultdict(lambda: defaultdict(set))  # kind -> person -> persons

    for n in nodes:
        for subj, rel, obj in extract_all(n["text"]):
            by_person[subj].append(n["id"])
            if rel in ATTR_LABEL:
                attr[rel][obj].add(subj)
                continue
            if rel not in PERSON_REL:
                continue                       # birth dates, gender: no index
            by_person[obj].append(n["id"])
            if rel in FRIEND_REL:
                rel_map["friends"][subj].add(obj)
                rel_map["friends"][obj].add(subj)
            elif rel in CHILD_REL:
                rel_map["children"][subj].add(obj)
                rel_map["parents"][obj].add(subj)
            elif rel in PARENT_REL:
                rel_map["parents"][subj].add(obj)
                rel_map["children"][obj].add(subj)
            else:                              # spouse-like, symmetric
                rel_map["spouse"][subj].add(obj)
                rel_map["spouse"][obj].add(subj)

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
        label = ATTR_LABEL[kind]
        for value, people in sorted(values.items()):
            new(f"{label} {value}", [person_node[p] for p in people if p in person_node])
            n_attr += 1

    # layer 3: relation indexes, linked to person nodes
    n_rel = 0
    for p, friends in sorted(rel_map["friends"].items()):
        if p in person_node:
            new(f"Friends of {p}", [person_node[q] for q in friends if q in person_node])
            n_rel += 1
    for p, kids in sorted(rel_map["children"].items()):
        if p in person_node:
            new(f"Children of {p}", [person_node[q] for q in kids if q in person_node])
            n_rel += 1
            grand = {g for k in kids for g in rel_map["children"].get(k, ())}
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
