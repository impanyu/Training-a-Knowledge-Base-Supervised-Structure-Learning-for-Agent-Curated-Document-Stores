"""Semantic index scoring: does a link point at a document that belongs?

The string-match scorer (scripts/index_precision.py) asks whether the target
document mentions the index's key. That test is sound for attribute indexes
("Residents of Eastmere" -> documents naming Eastmere) but wrong for
relational ones: the grandchildren of Ivor Yarrow are recorded as "<person>
is a child of <parent>", and never name the grandparent, so a perfectly
correct two-hop index scores zero. This scorer resolves the key's true
member set from the universe and asks whether the target document is about a
member.

    python3 scripts/index_precision2.py --kb runs/v11_main/kb_epoch_2.json
"""
import argparse
import json
import re
from collections import defaultdict

NAME = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")


def universe_relations(nodes):
    """child->parents, parent->children, friends, and attribute maps."""
    kids, parents, friends = defaultdict(set), defaultdict(set), defaultdict(set)
    attr = defaultdict(set)                       # value -> people
    for n in nodes:
        t = n["text"].rstrip(".")
        if (m := re.match(r"(.+) is a child of (.+)$", t)):
            kids[m.group(2)].add(m.group(1)); parents[m.group(1)].add(m.group(2))
        elif (m := re.match(r"(.+) is a friend of (.+)$", t)):
            friends[m.group(1)].add(m.group(2)); friends[m.group(2)].add(m.group(1))
        elif (m := re.match(r"(.+) lives in the city of (.+)$", t)):
            attr[m.group(2)].add(m.group(1))
        elif (m := re.match(r"(.+)'s hobby is (.+)$", t)):
            attr[m.group(2)].add(m.group(1))
        elif (m := re.match(r"(.+)'s job is (.+)$", t)):
            attr[m.group(2)].add(m.group(1))
    return kids, parents, friends, attr


def members(text, kids, parents, friends, attr):
    """The set of PEOPLE this index claims to gather, or None if unparsable."""
    t = text.rstrip(".")
    names = NAME.findall(t)
    who = names[0] if len(names) == 1 else None
    if who:
        low = t.lower()
        if low.startswith("children of"):
            return kids.get(who, set())
        if low.startswith("grandchildren of"):
            return {g for c in kids.get(who, set()) for g in kids.get(c, set())}
        if low.startswith("siblings of"):
            s = {k for p in parents.get(who, set()) for k in kids.get(p, set())}
            return s - {who}
        if low.startswith("friends of"):
            return friends.get(who, set())
        if low.startswith("parents of"):
            return parents.get(who, set())
        # a bare person hub: documents about that person
        return {who}
    for value, people in attr.items():
        if re.search(rf"\b{re.escape(value)}\b", t, re.I):
            return people
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--universe", default="data/v10L/universe.json")
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()

    uni = json.load(open(a.universe))
    kids, parents, friends, attr = universe_relations(uni["nodes"])
    d = json.load(open(a.kb))
    nodes = d["store"]["nodes"] if "store" in d else d["nodes"]
    by_id = {n["id"]: n for n in nodes}
    idx = [n for n in nodes if n.get("flag") == "authored"]

    # every universe document, and which people it is about
    about = {n["id"]: set(NAME.findall(n["text"])) for n in uni["nodes"]}

    rows, unscored = [], 0
    for n in idx:
        want = members(n["text"], kids, parents, friends, attr)
        links = [t for t in n.get("links", []) if t in by_id]
        if want is None or not links:
            unscored += 1
            continue
        hit = sum(1 for t in links if about.get(t, set()) & want)
        reachable = {p for t in links for p in about.get(t, set()) & want}
        rows.append((n["text"][:44], len(links), hit, len(want), len(reachable)))

    P = sum(r[2] for r in rows) / max(1, sum(r[1] for r in rows))
    R = sum(r[4] for r in rows) / max(1, sum(r[3] for r in rows))
    print(f"{len(rows)} scorable indexes ({unscored} unscorable), "
          f"precision {P:.0%}, member recall {R:.0%}\n")
    if a.show:
        print(f"{'index':<46}{'links':>6}{'ok':>5}{'members':>9}{'found':>7}{'prec':>7}")
        for r in sorted(rows, key=lambda r: -r[1])[:a.show]:
            print(f"{r[0]:<46}{r[1]:>6}{r[2]:>5}{r[3]:>9}{r[4]:>7}{r[2]/r[1]:>7.0%}")


if __name__ == "__main__":
    main()
