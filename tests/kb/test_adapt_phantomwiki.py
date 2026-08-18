"""T41 official-PhantomWiki adapter tests — offline, against the tiny
checked-in fixture in tests/kb/pw_fixture/ (three articles + 16 questions in
phantom-wiki 1.0.3 json output format; hand-constructed, with a pronoun
section and a multi-sentence line that real 1.0.3 output never emits, to
exercise the normalization guards)."""
from pathlib import Path

from kb.adapt_phantomwiki import adapt, main, normalize, split_article
from kb.build import Universe
from kb.store import Store

from .fixtures import HashEmbedding

SRC = Path(__file__).parent / "pw_fixture"
SIZES = (4, 3, 3)
EVAL = (2, 2)


def _adapt(seed=0):
    return adapt(SRC, seed, SIZES, EVAL)


# ---------------- sentence splitting ----------------

def test_sentence_splitting_headers_blanks_multisentence():
    stmts = split_article("Ryan Wang", (
        "# Ryan Wang\n\n## Family\n"
        "The wife of Ryan Wang is Aida Wang. The daughter of Ryan Wang is "
        "Johnetta Wang.\n\n## Attributes\n"
        "The occupation of Ryan Wang is coroner.\n"))
    assert stmts == ["The wife of Ryan Wang is Aida Wang.",
                     "The daughter of Ryan Wang is Johnetta Wang.",
                     "The occupation of Ryan Wang is coroner."]


def test_fixture_node_count_and_ids():
    u = _adapt()
    # Aida 7 lines, Ryan 5 lines (first splits in two -> 6), Shelli 3
    assert len(u.nodes) == 16
    assert [n["id"] for n in u.nodes] == [f"s{i:04d}" for i in range(1, 17)]
    assert all(n["origin"] == n["id"] for n in u.nodes)
    assert all("links" not in n for n in u.nodes)          # zero links


# ---------------- self-containedness normalization ----------------

def test_normalization_pronouns_and_last_resort_prefix():
    assert (normalize("Ryan Wang", "He enjoys quiet evenings.")
            == "Ryan Wang enjoys quiet evenings.")
    assert (normalize("Ryan Wang", "His hobby is birdwatching.")
            == "Ryan Wang's hobby is birdwatching.")
    assert (normalize("Ryan Wang", "The gender of this person is male.")
            == "Ryan Wang: The gender of this person is male.")
    # already self-contained -> untouched
    assert (normalize("Ryan Wang", "The occupation of Ryan Wang is coroner.")
            == "The occupation of Ryan Wang is coroner.")


def test_every_statement_names_its_person():
    u = _adapt()
    titles = ["Aida Wang", "Ryan Wang", "Shelli Beltran"]
    for n in u.nodes:
        assert any(t in n["text"] for t in titles), n["text"]


# ---------------- questions, golds, categories ----------------

def test_golds_list_answers_and_numeric_variants():
    u = _adapt()
    by_text = {q.text: q for q in u.questions.values()}
    sis = by_text["Who is the sister of Aida Wang?"]
    assert sis.golds == ["Jeannine Wexler, Vicki Hackworth",
                         "Jeannine Wexler", "Vicki Hackworth"]
    agg = by_text["How many sisters does Aida Wang have?"]
    assert agg.golds == ["2", "two"]
    assert not agg.unanswerable
    assert agg.hops == 1


def test_categories_and_templates_from_pw_taxonomy():
    u = _adapt()
    for q in u.questions.values():
        t = u.vocab["templates"][q.template]
        assert q.category == t["category"]
    assert u.vocab["templates"]["pw_type1"] == {
        "category": "PW_REL", "hops": None, "reserved": False, "pw_type": 1,
        "pw_template": "Who is the <relation>_3 of <name>_4 ?",
        "is_aggregation": False}
    assert u.vocab["templates"]["pw_type0"]["reserved"] is True
    assert u.vocab["templates"]["pw_type6"]["reserved"] is True


def test_supports_empty_and_noted():
    u = _adapt()
    assert all(q.support == [] for q in u.questions.values())
    assert "repair diagnostics unavailable" in u.meta["supports"]


# ---------------- splits ----------------

def test_split_sizes_disjointness_and_reservation():
    u = _adapt()
    assert {s: len(q) for s, q in u.splits.items()} == {
        "train": 4, "test_in": 3, "test_out": 3, "eval": 4}
    all_qids = [qid for qids in u.splits.values() for qid in qids]
    assert len(all_qids) == len(set(all_qids))
    texts = [u.questions[qid].text for qid in all_qids]
    assert len(texts) == len(set(texts))       # duplicate b5 was dropped
    trained = {"pw_type1", "pw_type7"}
    for qid in u.splits["train"] + u.splits["test_in"]:
        assert u.questions[qid].template in trained
    for qid in u.splits["test_out"]:
        assert u.questions[qid].template in {"pw_type0", "pw_type6"}
    flavors = [u.questions[qid].eval_flavor for qid in u.splits["eval"]]
    assert flavors.count("in") == 2 and flavors.count("out") == 2
    for qid in u.splits["eval"]:
        q = u.questions[qid]
        assert q.template in (trained if q.eval_flavor == "in"
                              else {"pw_type0", "pw_type6"})


# ---------------- determinism + downstream compatibility ----------------

def test_determinism_and_seed_sensitivity():
    a, b = _adapt(seed=0), _adapt(seed=0)
    assert a.to_json() == b.to_json()
    c = _adapt(seed=1)
    assert {q.text for q in a.questions.values()} != \
        {u.text for u in c.questions.values()} or \
        a.to_json() != c.to_json()             # different draw order at least


def test_universe_round_trips_and_loads_into_store(tmp_path):
    main(["--src", str(SRC), "--out", str(tmp_path), "--train", "4",
          "--test-in", "3", "--test-out", "3", "--eval-in", "2",
          "--eval-out", "2"])
    u = Universe.load(tmp_path / "universe.json")
    assert len(u.nodes) == 16 and len(u.questions) == 14
    store = Store.from_nodes(u.nodes, embedding_function=HashEmbedding())
    assert store.stats()["n_nodes"] == 16
    assert store.stats()["coverage"] == 1.0
    hits = store.search("occupation of Ryan Wang coroner", k=3)
    assert any("coroner" in text for _, text in hits)
