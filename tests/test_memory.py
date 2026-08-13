import json

from fixtures import DEMO_CORPUS, HashEmbedding, demo_corpus_embeddings

from ca.memory import AgentMemory, FifoMemory, GoalStack, load_corpus


def make(seeded=True):
    mem = AgentMemory(embedding_function=HashEmbedding())
    if seeded:
        mem.seed_corpus(DEMO_CORPUS, demo_corpus_embeddings())
    return mem


# ---------------- short-term ----------------

def test_fifo_rolls_over():
    f = FifoMemory(2)
    f.add("a", "1")
    f.add("b", "2")
    f.add("c", "3")
    assert [a for a, _ in f.items] == ["b", "c"]
    assert "a" not in f.render()


def test_fifo_pair_based_keeps_results_full():
    f = FifoMemory(3)
    f.add("act", "R" * 500)
    assert f.render().count("R") == 500        # never truncated


def test_goal_stack_root_protected():
    g = GoalStack("root")
    g.push("sub")
    assert g.pop() == "sub"
    try:
        g.pop()
        assert False, "root must be unpoppable"
    except IndexError:
        pass


# ---------------- the shared KB ----------------

def test_one_physical_store_for_every_agent():
    mem = make()
    mem.write("agent_1", "France's fastest train is the TGV")
    hits = mem.search("fastest train France")
    assert any("TGV" in h["text"] for h in hits)      # no per-caller scoping
    assert mem.n_entries() == len(DEMO_CORPUS) + 1


def test_search_is_semantic_over_corpus_notes_and_answers_alike():
    mem = make()
    mem.write("agent_1", "note: chalk cliffs are in Normandy")
    mem.write("agent_2", '[q0001] capital of France? -> "Paris" (F1 1.00)',
              kind="answer", qid="q0001", f1=1.0)
    hits = mem.search("capital of France", k=6)
    kinds = {h["kind"] for h in hits}
    assert {"corpus", "answer"} <= kinds


def test_search_honours_k_and_never_exceeds_the_store():
    mem = make()
    assert len(mem.search("France", k=2)) == 2
    assert len(mem.search("France", k=100)) == len(DEMO_CORPUS)


def test_every_write_carries_its_author():
    mem = make()
    mem.write("agent_1", "a note")
    mem.write("agent_2", "Q: q?\nA: a", kind="selfqa")
    assert mem.count("note", "agent_1") == 1
    assert mem.count("note", "agent_2") == 0
    assert mem.count("selfqa", "agent_2") == 1
    assert mem.count("selfqa") == 1


def test_counts_by_kind_never_include_the_corpus():
    mem = make()
    assert mem.count("note") == 0 and mem.count("answer") == 0
    assert mem.count("selfqa") == 0
    assert mem.n_entries() == len(DEMO_CORPUS)


def test_seed_corpus_entries_are_searchable_with_their_title():
    mem = make()
    hits = mem.search("longest river of France", k=1)
    assert hits[0]["kind"] == "corpus" and hits[0]["title"] == "Loire"
    assert hits[0]["agent"] is None            # the corpus has no author


def test_seeding_never_re_embeds():
    class Exploding(HashEmbedding):
        def __call__(self, input):
            raise AssertionError("seeding must use the precomputed embeddings")

    mem = AgentMemory(embedding_function=Exploding())
    mem.seed_corpus(DEMO_CORPUS, demo_corpus_embeddings())
    assert mem.n_entries() == len(DEMO_CORPUS)


def test_notes_ids_continue_after_the_corpus():
    mem = make()
    mem.write("agent_1", "first note")
    got = mem._col().get(where={"kind": "note"})
    assert got["ids"] == [f"m{len(DEMO_CORPUS) + 1}"]
    assert got["metadatas"][0]["seq"] == len(DEMO_CORPUS) + 1


def test_memory_is_append_only():
    mem = make()
    mem.write("agent_1", '[q0001] x -> "wrong" (F1 0.00)', kind="answer",
              qid="q0001", f1=0.0)
    mem.write("agent_2", '[q0001] x -> "Paris" (F1 1.00)', kind="answer",
              qid="q0001", f1=1.0)
    assert mem.count("answer") == 2            # both attempts on record


# ---------------- checkpoint ----------------

def test_state_roundtrips_through_json_and_reindexes():
    mem = make()
    mem.write("agent_1", "note one")
    mem.write("agent_2", "Q: q?\nA: a", kind="selfqa")
    mem.write("agent_1", '[q0003] largest city? -> "Paris" (F1 1.00)',
              kind="answer", qid="q0003", f1=1.0)
    state = json.loads(json.dumps(mem.to_state()))

    fresh = make()                             # re-seeded, then restored
    fresh.from_state(state)
    assert fresh.count("note", "agent_1") == 1
    assert fresh.count("selfqa", "agent_2") == 1
    assert fresh.count("answer") == 1
    hits = fresh.search("largest city Paris", k=2)
    assert any(h["qid"] == "q0003" for h in hits)


def test_corpus_is_excluded_from_state_and_survives_restore():
    mem = make()
    mem.write("agent_1", "a note")
    state = mem.to_state()
    assert len(state) == 1                     # the note, not 1 + corpus
    fresh = make()
    fresh.from_state(state)
    assert fresh.n_entries() == len(DEMO_CORPUS) + 1
    assert fresh.search("capital of France", k=1)[0]["kind"] == "corpus"


def test_seq_determinism_survives_save_restore():
    mem = make()
    mem.write("agent_1", "n1")
    fresh = make()
    fresh.from_state(mem.to_state())
    fresh.write("agent_2", "n2")
    mem.write("agent_2", "n2")
    assert mem.to_state() == fresh.to_state()  # ids/seq identical either way


def test_from_state_replaces_previous_writes_but_never_the_corpus():
    mem = make()
    mem.write("agent_1", "stale note")
    mem.from_state([])                         # empty snapshot
    assert mem.count("note") == 0
    assert mem.n_entries() == len(DEMO_CORPUS)


def test_real_v5_corpus_loads_aligned_with_its_embeddings(tmp_path):
    import numpy as np
    paras = [{"title": "A", "text": "alpha"}, {"title": "B", "text": "beta"}]
    with open(tmp_path / "corpus.jsonl", "w") as f:
        for p in paras:
            f.write(json.dumps(p) + "\n")
    np.save(tmp_path / "corpus_emb.npy",
            np.asarray(HashEmbedding()([p["text"] for p in paras]), dtype=np.float32))
    got, emb = load_corpus(tmp_path / "corpus.jsonl", tmp_path / "corpus_emb.npy")
    assert got == paras and emb.shape[0] == 2

    np.save(tmp_path / "corpus_emb.npy", emb[:1])
    try:
        load_corpus(tmp_path / "corpus.jsonl", tmp_path / "corpus_emb.npy")
        assert False, "misaligned embeddings must be rejected"
    except ValueError:
        pass
