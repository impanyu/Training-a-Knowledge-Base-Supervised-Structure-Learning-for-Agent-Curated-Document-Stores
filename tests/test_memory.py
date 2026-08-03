"""Short-term memories plus the v4 vector-backed AgentMemory.

The vector store is exercised with a cheap deterministic embedding stub (a
normalized bag-of-words hash), so the unit tests never touch the ONNX model --
same trick as KeywordBackend standing in for ChromaBackend in test_retrieval.
"""
import json
import math
import zlib

import pytest
from chromadb.api.types import EmbeddingFunction
from ca.memory import AgentMemory, FifoMemory, GoalStack

DIM = 64


class HashEmbedding(EmbeddingFunction):
    """Deterministic bag-of-words embedding: cosine order == token overlap."""

    def __init__(self):
        pass

    @staticmethod
    def name() -> str:
        return "hash-stub"

    def get_config(self) -> dict:
        return {}

    def __call__(self, input):
        out = []
        for doc in input:
            v = [0.0] * DIM
            for w in str(doc).lower().split():
                v[zlib.crc32(w.encode()) % DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def mem(**kw) -> AgentMemory:
    return AgentMemory(embedding_function=HashEmbedding(), **kw)


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
        {"text": "secret of b", "kind": "note", "qid": None, "f1": None}]
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
                                   "qid": "q1", "f1": None}
    m.write("a", "graded", kind="answer", qid="q1", f1=0.1)
    assert m.answer("a", "q1")["text"] == "graded"
    m.write("a", "another contract", kind="answer", qid="q1")
    assert m.answer("a", "q1")["text"] == "graded"


def test_n_answers_counts_only_answers_and_is_per_agent():
    m = mem()
    m.write("a", "a note")
    m.write("a", "an answer", kind="answer", qid="q1", f1=1.0)
    assert m.n_answers("a") == 1 and m.n_answers("b") == 0


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


def test_shared_state_roundtrips():
    m = mem(shared=True)
    m.write("agent_1", "shared answer", kind="answer", qid="q1", f1=1.0)
    fresh = mem(shared=True)
    fresh.from_state(json.loads(json.dumps(m.to_state())))
    assert fresh.answer("agent_9", "q1")["text"] == "shared answer"


def test_from_state_replaces_previous_content():
    m = mem()
    m.write("agent_1", "stale note")
    m.from_state({})
    assert m.search("agent_1", "stale note") == []
