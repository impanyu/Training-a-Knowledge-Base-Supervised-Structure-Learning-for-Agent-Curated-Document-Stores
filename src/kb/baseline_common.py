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
    None (agent-authored / summary nodes do not parse, by design).

    Tries the KBGym templates first, then the PhantomWiki grammar, so a
    baseline builds on either arm without being told which it is looking at.
    A plural PhantomWiki sentence names several objects; only the first is
    returned here, and callers wanting all of them use extract_all()."""
    for pat, rel in TEMPLATES:
        m = pat.match(text.strip())
        if m:
            return m.group(1), rel, m.group(2)
    pw = extract_pw(text.strip())
    return pw[0] if pw else None


def extract_all(text: str) -> list[tuple[str, str, str]]:
    """Every triple in a statement. Identical to extract_triple on KBGym,
    where a sentence carries one fact; on PhantomWiki a friends list yields
    one triple per friend, which is what makes those people co-occur."""
    t = extract_triple(text.strip())
    if t is None:
        return []
    pw = extract_pw(text.strip())
    return pw if len(pw) > 1 else [t]


def entity_mentions(triple: tuple[str, str, str]) -> list[tuple[str, str]]:
    """(kind, name) entities a statement mentions. Kinds: person / job /
    hobby / city. Birth dates are globally unique by construction (exactly
    one statement each) so they are useless as shared entities — excluded."""
    subj, rel, obj = triple
    out = [("person", subj)]
    if rel in PERSON_RELATIONS or rel in PW_PERSON_RELATIONS:
        out.append(("person", obj))
    elif rel in ATTR_KIND:
        out.append((ATTR_KIND[rel], obj))
    elif rel in ("occupation", "hobby"):        # PhantomWiki attribute names
        out.append(("job" if rel == "occupation" else "hobby", obj))
    return out


def index_statements(nodes: list[dict]) -> tuple[dict, dict]:
    """(entity -> sorted sid list, sid -> triple) over the template-shaped
    nodes; entity keys are (kind, name) tuples."""
    by_entity: dict[tuple[str, str], list[str]] = {}
    triples: dict[str, tuple[str, str, str]] = {}
    for n in nodes:
        ts = extract_all(n["text"])
        if not ts:
            continue
        triples[n["id"]] = ts[0]
        for t in ts:
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

# ---------------------------------------------------------------- PhantomWiki
# Their articles are template-rendered too, but with a different grammar:
# "The <relation> of <NAME> is <VALUE>." and its plural "... are <V1>, <V2>."
# Two patterns cover 3,403 of 3,403 notes, so the baselines can be built on
# this arm with the same free, exact extraction used on KBGym rather than
# being skipped for want of an extractor.
_PW_NAME = r"[A-Z][a-zA-Z'\-]+(?: [A-Z][a-zA-Z'\-]+)*"
PW_SINGULAR = re.compile(rf"^The ([\w ]+?) of ({_PW_NAME}) is (.+)\.$")
PW_PLURAL = re.compile(rf"^The ([\w ]+?) of ({_PW_NAME}) are (.+)\.$")

PW_PERSON_RELATIONS = frozenset({
    "mother", "father", "husband", "wife", "son", "sons", "daughter",
    "daughters", "brother", "brothers", "sister", "sisters", "friend",
    "friends",
})


def extract_pw(text: str) -> list[tuple[str, str, str]]:
    """(subject, relation, object) triples from one PhantomWiki sentence.

    A plural sentence names several objects and yields one triple each, so a
    friends list becomes edges rather than a single opaque fact."""
    m = PW_SINGULAR.match(text)
    if m:
        return [(m.group(2), m.group(1), m.group(3))]
    m = PW_PLURAL.match(text)
    if m:
        rel, subj = m.group(1), m.group(2)
        return [(subj, rel, o.strip()) for o in m.group(3).split(",") if o.strip()]
    return []
