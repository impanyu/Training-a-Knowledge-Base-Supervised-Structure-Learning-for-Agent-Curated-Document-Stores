"""B5 — HippoRAG2-style offline structuring baseline (T40).

Builds a KG layer over the statement graph, adapted to OUR store format so
the fixed reader (kb.test) runs on the output unchanged:

1. (subject, relation, object) triples per statement — LEXICAL (see
   baseline_common; documented deviation from the original's LLM openIE:
   the statements are template-shaped, a regex extractor is exact and free,
   so the build costs ZERO LLM tokens);
2. "synapse" edges: every pair of statements sharing a PERSON entity is
   bidirectionally linked (the passage-graph edges HippoRAG's random walk
   would traverse). Person entities mentioned by more than --synapse-cap
   statements are skipped (clique guard; the entity index still covers
   them);
3. per-entity INDEX nodes (HippoRAG's phrase-node layer; --no-entity-index
   disables): one authored node per entity — persons AND attribute values
   (job / hobby / city) — mentioned by >= 2 statements, linked to every
   mentioning statement, with a backlink from each statement.

DOCUMENTED DEVIATION — no Personalized PageRank at query time: our reader is
fixed (budgeted search/read/answer), so the baseline's power must come
through structure the reader can traverse. We evaluate HippoRAG2's
STRUCTURE under a common reader, not its retrieval algorithm.

    python -m kb.baseline_hipporag --universe data/v10L/universe.json \
        --out data/v10L_hipporag [--synapse-cap 40] [--no-entity-index]
"""
import argparse
import json
import sys
import time

from kb.baseline_common import (extract_pw, PW_PERSON_RELATIONS,
                                copy_universe, index_statements, next_sid,
                                write_output)

SYNAPSE_CAP = 40

_INDEX_TEXT = {
    "person": ("Entity index for {name}: every note mentioning {name} is "
               "linked from this note."),
    "job": ("Entity index for the job {name}: every note about a person "
            "whose job is {name} is linked from this note."),
    "hobby": ("Entity index for the hobby {name}: every note about a person "
              "whose hobby is {name} is linked from this note."),
    "city": ("Entity index for the city {name}: every note about a person "
             "who lives in {name} is linked from this note."),
}


def build_hipporag(universe: dict, entity_index: bool = True,
                   synapse_cap: int = SYNAPSE_CAP) -> tuple[dict, dict]:
    """(universe copy with synapse links + entity index nodes added, meta)."""
    t0 = time.perf_counter()
    out = copy_universe(universe)
    by_entity, triples = index_statements(out["nodes"])
    node_by_id = {n["id"]: n for n in out["nodes"]}
    have = {n["id"]: set(n["links"]) for n in out["nodes"]}

    def _link(a: str, b: str) -> int:
        if b in have[a] or a == b:
            return 0
        node_by_id[a]["links"].append(b)
        have[a].add(b)
        return 1

    synapse_links = 0
    capped = 0
    for (kind, name), sids in sorted(by_entity.items()):
        if kind != "person" or len(sids) < 2:
            continue
        if len(sids) > synapse_cap:
            capped += 1
            continue
        for a in sids:
            for b in sids:
                synapse_links += _link(a, b)

    nodes_added = index_links = 0
    if entity_index:
        sid_n = next_sid(out["nodes"])
        for (kind, name), sids in sorted(by_entity.items()):
            if len(sids) < 2:
                continue
            nid = f"s{sid_n:04d}"
            sid_n += 1
            node = {"id": nid, "text": _INDEX_TEXT[kind].format(name=name),
                    "origin": None, "flag": "authored", "links": list(sids)}
            out["nodes"].append(node)
            node_by_id[nid] = node
            have[nid] = set(sids)
            nodes_added += 1
            index_links += len(sids)
            for s in sids:
                index_links += _link(s, nid)

    meta = {"baseline": "hipporag",
            "n_triples": len(triples),
            "n_entities": len(by_entity),
            "synapse_links": synapse_links,
            "capped_person_entities": capped,
            "entity_index": entity_index,
            "index_nodes": nodes_added,
            "index_links": index_links,
            "nodes_added": nodes_added,
            "links_added": synapse_links + index_links,
            "build_tokens_in": 0, "build_tokens_out": 0,   # lexical build
            "seconds": time.perf_counter() - t0}
    out["meta"] = {**out.get("meta", {}), "baseline": meta}
    return out, meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synapse-cap", type=int, default=SYNAPSE_CAP)
    ap.add_argument("--no-entity-index", action="store_true")
    args = ap.parse_args(argv)
    with open(args.universe) as f:
        universe = json.load(f)
    built, meta = build_hipporag(universe,
                                 entity_index=not args.no_entity_index,
                                 synapse_cap=args.synapse_cap)
    meta["universe"] = args.universe
    write_output(args.out, built, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    sys.exit(main())
