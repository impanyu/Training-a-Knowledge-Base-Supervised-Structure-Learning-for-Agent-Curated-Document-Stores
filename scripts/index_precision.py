"""How much of what an index links actually belongs under its key?

Out-degree alone cannot tell a good index from a dumped search result: a
person node carrying forty links looks healthier than one carrying ten and
is worse, because thirty of them are about other people. For an index whose
text names an entity or an attribute value, a linked note belongs if its
text mentions that key. Reported alongside recall against the universe, so
over-linking and under-linking are visible separately.

    python3 scripts/index_precision.py --kb runs/v11_pilot/kb_epoch_1.json
"""
import argparse
import json
import re

NAME = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")


def key_of(text, vocab):
    """The thing this index claims to be about, if it is stated plainly."""
    names = NAME.findall(text)
    if len(names) == 1:
        return names[0]
    for v in vocab:
        if re.search(rf"\b{re.escape(v)}\b", text, re.I):
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--universe", default="data/v10L/universe.json")
    a = ap.parse_args()

    uni = json.load(open(a.universe))
    v = uni["vocab"]
    vocab = list(v.get("jobs", [])) + list(v.get("hobbies", [])) + list(v.get("cities", []))
    d = json.load(open(a.kb))
    nodes = d["store"]["nodes"] if "store" in d else d["nodes"]
    by_id = {n["id"]: n for n in nodes}
    idx = [n for n in nodes if n.get("flag") == "authored"]

    universe_text = {n["id"]: n["text"] for n in uni["nodes"]}
    rows, unscored = [], 0
    for n in idx:
        k = key_of(n["text"], vocab)
        links = [t for t in n.get("links", []) if t in by_id]
        if not k or not links:
            unscored += 1
            continue
        hit = sum(1 for t in links if re.search(re.escape(k), by_id[t]["text"], re.I))
        total_in_universe = sum(1 for t in universe_text.values()
                                if re.search(re.escape(k), t, re.I))
        rows.append((n["text"][:40], k, len(links), hit, total_in_universe))

    if not rows:
        print("no scorable indexes")
        return
    P = sum(r[3] for r in rows) / sum(r[2] for r in rows)
    R = sum(min(r[3], r[4]) for r in rows) / sum(r[4] for r in rows)
    print(f"{len(rows)} scorable indexes ({unscored} unscorable), "
          f"precision {P:.0%}, recall {R:.0%}\n")
    print(f"{'index':<42}{'links':>6}{'belong':>8}{'in universe':>13}{'prec':>7}")
    for text, k, n_l, hit, tot in sorted(rows, key=lambda r: -r[2])[:15]:
        print(f"{text:<42}{n_l:>6}{hit:>8}{tot:>13}{hit/n_l:>7.0%}")


if __name__ == "__main__":
    main()
