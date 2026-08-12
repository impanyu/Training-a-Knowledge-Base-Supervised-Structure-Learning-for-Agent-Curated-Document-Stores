"""Short-term memories plus the v5 vector-backed AgentMemory.

The vector store is exercised with a cheap deterministic embedding stub (a
normalized bag-of-words hash, `fixtures.HashEmbedding`), so the unit tests
never touch the ONNX model. Corpus seeding goes through the SAME
`seed_corpus` path production uses, with stub-computed embeddings.
"""
import json
from pathlib import Path

import pytest
from fixtures import DEMO_CORPUS, HashEmbedding, demo_corpus_embeddings

from ca.memory import AgentMemory, FifoMemory, GoalStack, load_corpus

V5 = Path(__file__).resolve().parents[1] / "data" / "v5"


def mem(**kw) -> AgentMemory:
    return AgentMemory(embedding_function=HashEmbedding(), **kw)


def seeded(agents=("a", "b"), **kw) -> AgentMemory:
    m = mem(**kw)
    m.seed_corpus(DEMO_CORPUS, demo_corpus_embeddings(), list(agents))
    return m


# ---------------- short-term (unchanged) ----------------

def test_fifo_rolls_over():
    m = FifoMemory(k=2)
    m.add("a1", "r1"); m.add("a2", "r2"); m.add("a3", "r3")
    out = m.render()
    assert "a1" not in out and "a2" in out and "a3" in out


def test_fifo_pair_based_keeps_results_full():
    """K is the pair budget: 4 pairs added to k=3, oldest evicted, 3 remain IN FULL."""
    m = FifoMemory(k=3)
    big1 = "x" * 2000 + " MARKER_1"
    big2 = "y" * 2000 + " MARKER_2"
    big3 = "z" * 2000 + " MARKER_3"
    big4 = "w" * 2000 + " MARKER_4"

    m.add("act1", big1)
    m.add("act2", big2)
    m.add("act3", big3)
    m.add("act4", big4)  # oldest (big1) should be evicted

    out = m.render()
    assert "MARKER_1" not in out
    assert "MARKER_2" in out and "MARKER_3" in out and "MARKER_4" in out
    assert m.items[0][1] == big2
    assert m.items[1][1] == big3
    assert m.items[2][1] == big4


def test_goal_stack_root_protected():
    g = GoalStack("maximize tokens")
    g.push("do q1")
    assert g.pop() == "do q1"
    with pytest.raises(IndexError):
        g.pop()
    assert "maximize tokens" in g.render()


# ---------------- AgentMemory: notes ----------------

def test_search_is_semantic_and_scoped_per_agent():
    m = mem()
    m.write("a", "paris is the capital of france")
    m.write("a", "tokyo is in japan")
    m.write("b", "secret of b")
    hits = m.search("a", "capital france", k=1)
    assert [h["text"] for h in hits] == ["paris is the capital of france"]
    assert hits[0]["kind"] == "note" and hits[0]["qid"] is None
    assert m.search("b", "capital france", k=3) == [
        {"text": "secret of b", "kind": "note", "qid": None, "f1": None,
         "title": None}]
    assert m.search("c", "anything") == []


def test_search_honours_k_and_never_exceeds_the_store():
    m = mem()
    for i in range(4):
        m.write("a", f"note number {i} about france")
    assert len(m.search("a", "france", k=2)) == 2
    assert len(m.search("a", "france", k=99)) == 4


def test_answers_are_ordinary_entries_found_by_meaning_too():
    m = mem()
    m.write("a", '[q0042] which river flows through orleans -> "Loire" (F1 1.00)',
            kind="answer", qid="q0042", f1=1.0)
    hit = m.search("a", "river orleans", k=1)[0]
    assert hit["qid"] == "q0042" and hit["kind"] == "answer" and hit["f1"] == 1.0


# ---------------- v5: the corpus is memory ----------------

def test_seed_corpus_entries_are_searchable_with_their_title():
    m = seeded()
    hit = m.search("a", "capital of France.", k=1)[0]
    assert hit == {"text": "Paris is the capital of France.", "kind": "corpus",
                   "qid": None, "f1": None, "title": "Paris"}


def test_seeding_never_re_embeds():
    class Exploding(HashEmbedding):
        def __call__(self, input):
            raise AssertionError("seed_corpus must not embed")

    m = AgentMemory(embedding_function=Exploding())
    m.seed_corpus(DEMO_CORPUS, demo_corpus_embeddings(), ["a"])   # no raise
    assert m.n_entries("a") == len(DEMO_CORPUS)


def test_one_search_spans_corpus_notes_and_answers():
    m = seeded()
    m.write("a", "note: chalk quarries are in kent")
    m.write("a", '[q0005] which rock type is chalk? -> "sedimentary" (F1 1.00)',
            kind="answer", qid="q0005", f1=1.0)
    kinds = {h["kind"] for h in m.search("a", "chalk sedimentary rock type", k=6)}
    assert kinds == {"corpus", "note", "answer"}


def test_corpus_entries_are_not_answers():
    m = seeded()
    assert m.answer("a", "q0001") is None
    assert m.n_answers("a") == 0
    assert m.n_notes("a") == 0
    assert m.n_entries("a") == len(DEMO_CORPUS)


def test_seeding_is_per_bucket_and_shared_seeds_once():
    m = seeded(agents=("agent_1", "agent_2"))
    assert m.n_entries("agent_1") == m.n_entries("agent_2") == len(DEMO_CORPUS)
    s = seeded(agents=("agent_1", "agent_2"), shared=True)
    assert s.n_entries("agent_9") == len(DEMO_CORPUS)   # one shared bucket


def test_notes_ids_continue_after_the_corpus_in_every_bucket():
    """Corpus consumes seq 1..N first, identically in every bucket, so
    note/answer ids are deterministic regardless of who writes first."""
    m = seeded(agents=("a", "b"))
    m.write("b", "first note")
    got = m._col("b").get(where={"kind": "note"})
    assert got["ids"] == [f"m{len(DEMO_CORPUS) + 1}"]
    assert got["metadatas"][0]["seq"] == len(DEMO_CORPUS) + 1


# ---------------- AgentMemory: answers by qid ----------------

def test_answer_is_a_metadata_lookup_not_a_substring_match():
    m = mem()
    m.write("a", "remember to look at q0042 tomorrow")          # a note mentioning it
    assert m.answer("a", "q0042") is None
    m.write("a", 'q0042 -> "Loire"', kind="answer", qid="q0042", f1=0.5)
    rec = m.answer("a", "q0042")
    assert rec["text"] == 'q0042 -> "Loire"' and rec["f1"] == 0.5
    assert m.answer("a", "q0001") is None


def test_memory_is_append_only_and_answer_returns_the_best_f1():
    m = mem()
    m.write("a", "first try", kind="answer", qid="q1", f1=0.3)
    m.write("a", "second try", kind="answer", qid="q1", f1=0.9)
    m.write("a", "third try", kind="answer", qid="q1", f1=0.4)
    assert m.n_answers("a") == 3                       # nothing overwritten
    assert m.answer("a", "q1")["text"] == "second try"
    assert len(m.search("a", "try", k=10)) == 3


def test_best_f1_ties_are_broken_by_recency():
    m = mem()
    m.write("a", "older", kind="answer", qid="q1", f1=0.5)
    m.write("a", "newer", kind="answer", qid="q1", f1=0.5)
    assert m.answer("a", "q1")["text"] == "newer"


def test_a_graded_answer_beats_an_ungraded_one_and_ungraded_falls_back_to_latest():
    m = mem()
    m.write("a", "from a contract", kind="answer", qid="q1")       # F1 unknown
    assert m.answer("a", "q1") == {"text": "from a contract", "kind": "answer",
                                   "qid": "q1", "f1": None, "title": None}
    m.write("a", "graded", kind="answer", qid="q1", f1=0.1)
    assert m.answer("a", "q1")["text"] == "graded"
    m.write("a", "another contract", kind="answer", qid="q1")
    assert m.answer("a", "q1")["text"] == "graded"


def test_n_answers_counts_only_answers_and_is_per_agent():
    m = mem()
    m.write("a", "a note")
    m.write("a", "an answer", kind="answer", qid="q1", f1=1.0)
    assert m.n_answers("a") == 1 and m.n_answers("b") == 0
    assert m.n_notes("a") == 1 and m.n_notes("b") == 0


# ---------------- C2: one shared bucket ----------------

def test_shared_bucket_pools_notes_and_answers_across_agents():
    m = mem(shared=True)
    m.write("agent_1", "paris is the capital of france")
    m.write("agent_1", "answer text", kind="answer", qid="q1", f1=1.0)
    assert m.answer("agent_2", "q1")["f1"] == 1.0
    assert m.n_answers("agent_2") == 1
    assert m.search("agent_2", "capital france", k=1)[0]["text"] == \
        "paris is the capital of france"


def test_private_buckets_do_not_leak():
    m = mem()
    m.write("agent_1", "answer text", kind="answer", qid="q1", f1=1.0)
    assert m.answer("agent_2", "q1") is None
    assert m.n_answers("agent_2") == 0


def test_corpus_seeded_private_buckets_still_do_not_leak_notes_or_answers():
    """v5 twist on the privacy invariant: every private store holds the SAME
    corpus, yet notes/answers written into one must stay invisible to the
    others."""
    m = seeded(agents=("agent_1", "agent_2"))
    m.write("agent_1", "private note about chalk")
    m.write("agent_1", "an answer", kind="answer", qid="q1", f1=1.0)
    assert m.answer("agent_2", "q1") is None
    assert m.n_notes("agent_2") == 0
    texts = [h["text"] for h in m.search("agent_2", "private note about chalk", k=10)]
    assert "private note about chalk" not in texts
    assert m.n_entries("agent_2") == len(DEMO_CORPUS)   # corpus, nothing else


# ---------------- checkpoint (T29) ----------------

def test_state_roundtrips_through_json_and_reindexes():
    m = mem()
    m.write("agent_1", "paris is the capital of france")
    m.write("agent_1", "first", kind="answer", qid="q1", f1=0.2)
    m.write("agent_1", "second", kind="answer", qid="q1", f1=0.8)
    m.write("agent_2", "tokyo is in japan")
    state = json.loads(json.dumps(m.to_state()))

    fresh = mem()
    fresh.from_state(state)
    assert fresh.n_answers("agent_1") == 2
    assert fresh.answer("agent_1", "q1")["text"] == "second"
    assert fresh.search("agent_1", "capital france", k=1)[0]["text"] == \
        "paris is the capital of france"
    assert fresh.search("agent_2", "japan", k=1)[0]["text"] == "tokyo is in japan"
    assert fresh.to_state() == state
    fresh.write("agent_1", "later", kind="answer", qid="q1", f1=0.9)
    assert fresh.answer("agent_1", "q1")["text"] == "later"   # ids keep growing


def test_corpus_is_excluded_from_state_and_survives_restore():
    m = seeded()
    m.write("a", "a note about chalk")
    m.write("a", "an answer", kind="answer", qid="q1", f1=1.0)
    state = json.loads(json.dumps(m.to_state()))
    assert all(row[2]["kind"] != "corpus"
               for rows in state.values() for row in rows)
    assert sum(len(rows) for rows in state.values()) == 2   # not 2 + corpus

    fresh = seeded()                       # freshly re-seeded store, then restore
    fresh.from_state(state)
    assert fresh.n_entries("a") == len(DEMO_CORPUS) + 2
    assert fresh.answer("a", "q1")["f1"] == 1.0
    assert fresh.search("a", "capital of France.", k=1)[0]["kind"] == "corpus"
    assert fresh.to_state() == state       # byte-parity across the round-trip


def test_seq_determinism_survives_save_restore():
    """A restored store must hand out the SAME ids a never-stopped run would:
    the corpus consumes seq 1..N, dumped rows keep theirs, and the next write
    continues from the maximum."""
    m = seeded()
    m.write("a", "note one")
    state = json.loads(json.dumps(m.to_state()))

    fresh = seeded()
    fresh.from_state(state)
    m.write("a", "note two")
    fresh.write("a", "note two")
    assert fresh.to_state() == m.to_state()


def test_from_state_replaces_previous_notes_but_never_the_corpus():
    m = seeded()
    m.write("a", "stale note")
    m.from_state({})
    texts = [h["text"] for h in m.search("a", "stale note", k=10)]
    assert "stale note" not in texts
    assert m.n_entries("a") == len(DEMO_CORPUS)


def test_shared_state_roundtrips():
    m = mem(shared=True)
    m.write("agent_1", "shared answer", kind="answer", qid="q1", f1=1.0)
    fresh = mem(shared=True)
    fresh.from_state(json.loads(json.dumps(m.to_state())))
    assert fresh.answer("agent_9", "q1")["text"] == "shared answer"


# ---------------- the real corpus ----------------

@pytest.mark.skipif(not ((V5 / "corpus.jsonl").exists()
                         and (V5 / "corpus_emb.npy").exists()),
                    reason="v5 corpus not built yet")
def test_real_v5_corpus_loads_aligned_with_its_embeddings():
    paras, emb = load_corpus(V5 / "corpus.jsonl", V5 / "corpus_emb.npy")
    assert len(paras) == len(emb) > 10000
    assert emb.dtype.name == "float32" and emb.ndim == 2
    assert {"title", "text"} <= set(paras[0])
