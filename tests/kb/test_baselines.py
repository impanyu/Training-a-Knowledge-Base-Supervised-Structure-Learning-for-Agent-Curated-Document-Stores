"""T40 offline-structuring baselines: lexical extractor correctness,
community / entity-index construction determinism, store-copy integrity,
CLI smoke with the stub summarizer. Offline only — no LLM, hash EF."""
import copy
import json

from kb.baseline_common import entity_mentions, extract_triple
from kb.baseline_graphrag import StubSummarizer, build_graphrag
from kb.baseline_graphrag import main as graphrag_main
from kb.baseline_hipporag import build_hipporag
from kb.baseline_hipporag import main as hipporag_main
from kb.build import Universe
from kb.store import Store

from tests.kb.fixtures import HashEmbedding, mini_universe

# ---------------- extractor ----------------

def test_extractor_covers_every_statement_template():
    cases = [
        ("Adelia Yarrow's job is falconer.",
         ("Adelia Yarrow", "job", "falconer")),
        ("Silas Kestrel's hobby is kite flying.",
         ("Silas Kestrel", "hobby", "kite flying")),
        ("Petra Foxglove lives in the city of Pinehaven.",
         ("Petra Foxglove", "lives in", "Pinehaven")),
        ("Rufus Grimsby was born on March 7, 1952.",
         ("Rufus Grimsby", "born on", "March 7, 1952")),
        ("Hazel Jessup is married to Caleb Jessup.",
         ("Hazel Jessup", "married to", "Caleb Jessup")),
        ("Fern Osprey is a child of Hazel Jessup.",
         ("Fern Osprey", "child of", "Hazel Jessup")),
        ("Caleb Jessup is the father of Fern Osprey.",
         ("Caleb Jessup", "parent of", "Fern Osprey")),
        ("Hazel Jessup is the mother of Fern Osprey.",
         ("Hazel Jessup", "parent of", "Fern Osprey")),
        ("Thea Vexley is a friend of Junia Dovecote.",
         ("Thea Vexley", "friend of", "Junia Dovecote")),
    ]
    for text, want in cases:
        assert extract_triple(text) == want, text


def test_extractor_rejects_non_template_text():
    assert extract_triple("Community summary: a family of potters.") is None
    assert extract_triple("Entity index for Petra Foxglove: notes.") is None
    assert extract_triple("Navigation note for q0001.") is None


def test_extractor_is_exact_on_a_real_build():
    u = mini_universe()
    for n in u.nodes:
        assert extract_triple(n["text"]) is not None, n["text"]


def test_entity_mentions_kinds():
    assert entity_mentions(("A B", "married to", "C D")) == [
        ("person", "A B"), ("person", "C D")]
    assert entity_mentions(("A B", "job", "weaver")) == [
        ("person", "A B"), ("job", "weaver")]
    assert entity_mentions(("A B", "lives in", "Otterby")) == [
        ("person", "A B"), ("city", "Otterby")]
    # birth dates are unique per person: never a shared entity
    assert entity_mentions(("A B", "born on", "May 1, 1950")) == [
        ("person", "A B")]


# ---------------- B3 GraphRAG ----------------

def test_graphrag_build_is_deterministic():
    u = mini_universe().to_json()
    a, ma = build_graphrag(copy.deepcopy(u), StubSummarizer(), max_size=6)
    b, mb = build_graphrag(copy.deepcopy(u), StubSummarizer(), max_size=6)
    ma.pop("seconds"), mb.pop("seconds")        # same dict object as in meta
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert ma == mb


def test_graphrag_communities_partition_persons_within_cap():
    u = mini_universe().to_json()
    built, meta = build_graphrag(copy.deepcopy(u), StubSummarizer(),
                                 max_size=6)
    summaries = [n for n in built["nodes"] if n.get("flag") == "authored"]
    assert len(summaries) == meta["n_communities"] == meta["nodes_added"]
    # stub text names the members: recover the partition from it
    seen = []
    for s in summaries:
        members = s["text"].split(" about ", 1)[1].rstrip(".").split(", ")
        assert 1 <= len(members) <= 6
        seen += members
    persons = {name for n in u["nodes"]
               for kind, name in entity_mentions(extract_triple(n["text"]))
               if kind == "person"}
    assert sorted(seen) == sorted(persons)      # exactly one community each


def test_graphrag_summary_links_reach_every_member_statement():
    u = mini_universe().to_json()
    built, meta = build_graphrag(copy.deepcopy(u), StubSummarizer())
    ids = {n["id"] for n in built["nodes"]}
    linked = set()
    for s in (n for n in built["nodes"] if n.get("flag") == "authored"):
        assert s["origin"] is None
        assert s["links"] == sorted(set(s["links"]))
        assert set(s["links"]) <= ids
        linked |= set(s["links"])
    # every template statement mentions a person, so every one is reachable
    assert linked == {n["id"] for n in u["nodes"]}
    assert meta["links_added"] == sum(
        len(n["links"]) for n in built["nodes"] if n.get("flag") == "authored")
    assert meta["build_tokens_in"] == 0        # stub: no LLM tokens


def test_graphrag_copy_integrity():
    u = mini_universe().to_json()
    before = copy.deepcopy(u)
    built, _ = build_graphrag(u, StubSummarizer())
    assert u == before                          # input never mutated
    originals = built["nodes"][:len(u["nodes"])]
    for orig, node in zip(u["nodes"], originals):
        assert node["id"] == orig["id"]
        assert node["text"] == orig["text"]
        assert node["origin"] == orig["id"]     # origins preserved
        assert node.get("flag") is None
        assert node["links"] == []              # graphrag links live on summaries
    assert built["questions"] == u["questions"]
    assert built["splits"] == u["splits"]


# ---------------- B5 HippoRAG2 ----------------

def test_hipporag_build_is_deterministic():
    u = mini_universe().to_json()
    a, ma = build_hipporag(copy.deepcopy(u))
    b, mb = build_hipporag(copy.deepcopy(u))
    ma.pop("seconds"), mb.pop("seconds")        # same dict object as in meta
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert ma == mb


def test_hipporag_synapses_link_statements_sharing_a_person():
    u = mini_universe().to_json()
    built, meta = build_hipporag(copy.deepcopy(u), entity_index=False)
    links = {n["id"]: set(n["links"]) for n in built["nodes"]}
    # gather each person's statements from the ORIGINAL nodes
    by_person: dict[str, list[str]] = {}
    for n in u["nodes"]:
        for kind, name in entity_mentions(extract_triple(n["text"])):
            if kind == "person":
                by_person.setdefault(name, []).append(n["id"])
    for name, sids in by_person.items():
        if len(sids) < 2 or len(sids) > 40:
            continue
        for a in sids:
            for b in sids:
                if a != b:
                    assert b in links[a], (name, a, b)
                    assert a in links[b], (name, a, b)   # symmetric
    assert meta["nodes_added"] == 0             # index disabled
    assert meta["links_added"] == meta["synapse_links"] == sum(
        len(v) for v in links.values())
    assert meta["build_tokens_in"] == meta["build_tokens_out"] == 0


def test_hipporag_synapse_cap_skips_big_cliques():
    u = mini_universe().to_json()
    built, meta = build_hipporag(copy.deepcopy(u), entity_index=False,
                                 synapse_cap=2)
    assert meta["capped_person_entities"] > 0
    # capped entities contribute no clique: far fewer links than uncapped
    _, full = build_hipporag(copy.deepcopy(u), entity_index=False)
    assert meta["synapse_links"] < full["synapse_links"]


def test_hipporag_entity_index_nodes_and_backlinks():
    u = mini_universe().to_json()
    built, meta = build_hipporag(copy.deepcopy(u))
    index = [n for n in built["nodes"] if n.get("flag") == "authored"]
    assert len(index) == meta["index_nodes"] == meta["nodes_added"] > 0
    by_id = {n["id"]: n for n in built["nodes"]}
    kinds_seen = set()
    for node in index:
        assert node["origin"] is None
        assert node["text"].startswith("Entity index for ")
        assert len(node["links"]) >= 2
        for s in node["links"]:
            assert node["id"] in by_id[s]["links"]      # backlink
        kinds_seen.add(node["text"].split("Entity index for ")[1].split()[0])
    # attribute-value entities are indexed too, not just persons
    assert "the" in kinds_seen                  # "the job/hobby/city ..."


def test_hipporag_copy_integrity():
    u = mini_universe().to_json()
    before = copy.deepcopy(u)
    built, _ = build_hipporag(u)
    assert u == before                          # input never mutated
    originals = built["nodes"][:len(u["nodes"])]
    for orig, node in zip(u["nodes"], originals):
        assert node["id"] == orig["id"]
        assert node["text"] == orig["text"]     # texts untouched
        assert node["origin"] == orig["id"]     # origins preserved
        assert node.get("flag") is None         # originals never flagged
    assert built["questions"] == u["questions"]
    assert built["splits"] == u["splits"]


# ---------------- CLI smoke (stub summarizer, offline) ----------------

def _write_mini(tmp_path):
    src = tmp_path / "universe.json"
    mini_universe().save(src)
    return src


def test_graphrag_cli_smoke(tmp_path):
    src = _write_mini(tmp_path)
    raw = src.read_bytes()
    out = tmp_path / "graphrag"
    graphrag_main(["--universe", str(src), "--out", str(out), "--stub"])
    assert src.read_bytes() == raw              # original file untouched
    meta = json.loads((out / "build_meta.json").read_text())
    assert meta["model"] == "stub" and meta["nodes_added"] > 0
    data = json.loads((out / "universe.json").read_text())
    u = Universe.from_json(data)                # kb.test loads it unchanged
    store = Store.from_nodes(u.nodes, embedding_function=HashEmbedding())
    stats = store.stats()
    assert stats["authored_statements"] == meta["nodes_added"]
    assert stats["coverage"] == 1.0             # every origin still alive
    assert store.search("Community summary", k=3)


def test_hipporag_cli_smoke(tmp_path):
    src = _write_mini(tmp_path)
    raw = src.read_bytes()
    out = tmp_path / "hipporag"
    hipporag_main(["--universe", str(src), "--out", str(out)])
    assert src.read_bytes() == raw              # original file untouched
    meta = json.loads((out / "build_meta.json").read_text())
    assert meta["nodes_added"] > 0 and meta["synapse_links"] > 0
    data = json.loads((out / "universe.json").read_text())
    u = Universe.from_json(data)
    store = Store.from_nodes(u.nodes, embedding_function=HashEmbedding())
    stats = store.stats()
    assert stats["authored_statements"] == meta["nodes_added"]
    assert stats["coverage"] == 1.0
    assert stats["n_links"] == meta["links_added"]
    # the reader can traverse: read any linked statement, links render
    some = next(n for n in u.nodes if n.get("links"))
    node = store.read(some["id"])
    assert node.links
