"""Build ablated versions of the trained store for a causal test of what the
agent built.

The observational touched/untouched split is not identifying: whether a
trajectory "touches" built structure is downstream of how long it searched
(P(touched) rises from 0% at one search to 63% at nine). The intervention is
on the STORE, with reader, questions and budget fixed:

  --drop nav    remove agent-authored navigation documents (keep the links
                that connect original statements to each other)
  --drop links  remove every link, keep the authored documents
  --drop both   both of the above (should reproduce B1 up to edited text)

    python3 scripts/make_ablations.py --kb runs/v10L_dedup/kb_epoch_2.json \
        --drop nav --out /tmp/kb_no_nav.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="runs/v10L_dedup/kb_epoch_2.json")
    ap.add_argument("--drop", choices=["nav", "links", "both"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = json.load(open(a.kb))
    nodes = d["store"]["nodes"]
    authored = {n["id"] for n in nodes if n.get("flag") == "authored"}

    if a.drop in ("nav", "both"):
        nodes = [n for n in nodes if n["id"] not in authored]
    for n in nodes:
        if a.drop in ("links", "both"):
            n["links"] = []
        else:
            # dangling links to removed nav docs would crash traversal
            n["links"] = [t for t in n.get("links", []) if t not in authored] \
                if a.drop == "nav" else n.get("links", [])

    d["store"]["nodes"] = nodes
    json.dump(d, open(a.out, "w"))
    print(f"{a.out}: {len(nodes)} nodes, "
          f"{sum(len(n.get('links', [])) for n in nodes)} links, "
          f"{sum(1 for n in nodes if n.get('flag') == 'authored')} authored")


if __name__ == "__main__":
    main()
