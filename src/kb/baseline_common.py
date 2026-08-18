"""Shared machinery for the T40 offline-structuring baselines (B3 GraphRAG /
B5 HippoRAG2): lexical extraction and store-copy plumbing.

DOCUMENTED CHOICE — lexical extraction, not LLM extraction: every build-time
statement in this corpus is rendered through one of eight fixed templates
(kb.build), so a regex per template recovers the (subject, relation, object)
triple EXACTLY and for free. The original GraphRAG / HippoRAG pipelines spend
LLM tokens on this step because their corpora are open prose; ours is not.
The baselines' LLM budget is reserved for the one step that genuinely needs
prose (GraphRAG's community summaries).

Both baselines write a FULL universe copy (questions / splits / vocab intact,
original file never touched) so kb.test runs on the output unchanged; added
nodes carry origin None + flag "authored", exactly like agent-authored notes.
"""
import json
import re
from pathlib import Path

_NAME = r"[A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+"

# One pattern per kb.build statement template, tried in order.
TEMPLATES: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"^({_NAME})'s job is (.+)\.$"), "job"),
    (re.compile(rf"^({_NAME})'s hobby is (.+)\.$"), "hobby"),
    (re.compile(rf"^({_NAME}) lives in the city of (.+)\.$"), "lives in"),
    (re.compile(rf"^({_NAME}) was born on (.+)\.$"), "born on"),
    (re.compile(rf"^({_NAME}) is married to ({_NAME})\.$"), "married to"),
    (re.compile(rf"^({_NAME}) is a child of ({_NAME})\.$"), "child of"),
    (re.compile(rf"^({_NAME}) is the (?:father|mother) of ({_NAME})\.$"),
     "parent of"),
    (re.compile(rf"^({_NAME}) is a friend of ({_NAME})\.$"), "friend of"),
]

PERSON_RELATIONS = frozenset({"married to", "child of", "parent of",
                              "friend of"})
ATTR_KIND = {"job": "job", "hobby": "hobby", "lives in": "city"}


def extract_triple(text: str) -> tuple[str, str, str] | None:
    """(subject, relation, object) for a template-shaped statement, else
    None (agent-authored / summary nodes do not parse, by design)."""
    for pat, rel in TEMPLATES:
        m = pat.match(text.strip())
        if m:
            return m.group(1), rel, m.group(2)
    return None


def entity_mentions(triple: tuple[str, str, str]) -> list[tuple[str, str]]:
    """(kind, name) entities a statement mentions. Kinds: person / job /
    hobby / city. Birth dates are globally unique by construction (exactly
    one statement each) so they are useless as shared entities — excluded."""
    subj, rel, obj = triple
    out = [("person", subj)]
    if rel in PERSON_RELATIONS:
        out.append(("person", obj))
    elif rel in ATTR_KIND:
        out.append((ATTR_KIND[rel], obj))
    return out


def index_statements(nodes: list[dict]) -> tuple[dict, dict]:
    """(entity -> sorted sid list, sid -> triple) over the template-shaped
    nodes; entity keys are (kind, name) tuples."""
    by_entity: dict[tuple[str, str], list[str]] = {}
    triples: dict[str, tuple[str, str, str]] = {}
    for n in nodes:
        t = extract_triple(n["text"])
        if t is None:
            continue
        triples[n["id"]] = t
        for ent in entity_mentions(t):
            by_entity.setdefault(ent, []).append(n["id"])
    for sids in by_entity.values():
        sids.sort()
    return by_entity, triples


def copy_universe(universe: dict) -> dict:
    """Deep-enough copy: node dicts and their link lists are fresh objects,
    so the input dict (and the file it came from) is never mutated."""
    return {**universe,
            "nodes": [{**n, "links": list(n.get("links", []))}
                      for n in universe["nodes"]]}


def next_sid(nodes: list[dict]) -> int:
    return max((int(n["id"][1:]) for n in nodes), default=0) + 1


def write_output(out_dir, universe: dict, meta: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "universe.json", "w") as f:
        json.dump(universe, f, ensure_ascii=False)
    with open(out / "build_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
